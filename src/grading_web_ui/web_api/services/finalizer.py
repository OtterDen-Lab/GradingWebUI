"""
Service for finalizing grading: annotating PDFs and uploading to Canvas.
"""
import asyncio
import base64
import html
import io
import json
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import fitz  # PyMuPDF
from PIL import Image

from ..repositories import SessionRepository, SubmissionRepository, ProblemRepository
from ..services.problem_service import ProblemService
from lms_interface.canvas_interface import CanvasInterface
from lms_interface.classes import Feedback
from .. import sse

log = logging.getLogger(__name__)


class FinalizationService:
  """Handles finalization of grading sessions"""

  def __init__(self, session_id: int, temp_dir: Path, stream_id: str,
               event_loop):
    self.session_id = session_id
    self.temp_dir = temp_dir
    self.stream_id = stream_id
    self.event_loop = event_loop  # Store event loop reference for thread communication
    self.canvas_interface = None
    self.course = None
    self.assignment = None
    self.total_submissions = 0
    self.current_submission = 0
    # Step-based progress tracking (3 steps per submission: PDF, comments, upload)
    self.steps_per_submission = 3
    self.total_steps = 0
    self.current_step = 0
    self.quiz_yaml_text = None

  def finalize(self):
    """Main finalization process (runs in thread to avoid blocking event loop)"""
    # Get session info and initialize Canvas
    session_info = self._get_session_info()
    self._init_canvas(session_info)

    # Get all submissions
    submissions = self._get_submissions()
    for submission in submissions:
      submission["assignment_name"] = session_info.get("assignment_name")
      submission["session_name"] = session_info.get("session_name")
    self.total_submissions = len(submissions)
    # Add 1 for final cleanup step
    self.total_steps = (self.total_submissions * self.steps_per_submission) + 1

    log.info(
      f"Finalizing {len(submissions)} submissions ({self.total_steps} total steps)"
    )

    # Process each submission
    for i, submission in enumerate(submissions, 1):
      self.current_submission = i
      student_name = submission['student_name'] or 'Unknown'

      try:
        # Generate annotated PDF
        self._update_progress(
          f"Processing {i}/{len(submissions)}: Generating PDF for {student_name}"
        )
        pdf_path = self._create_annotated_pdf(submission)

        # Generate comments
        self._update_progress(
          f"Processing {i}/{len(submissions)}: Preparing comments for {student_name}"
        )
        comments = self._generate_comments(submission)

        # Upload to Canvas
        self._update_progress(
          f"Processing {i}/{len(submissions)}: Uploading to Canvas for {student_name}"
        )
        self._upload_to_canvas(submission, pdf_path, comments)

        log.info(
          f"Successfully uploaded submission {i}/{len(submissions)} for {student_name}"
        )

      except Exception as e:
        log.error(f"Failed to process submission {submission['id']}: {e}",
                  exc_info=True)
        self._update_progress(
          f"Processing {i}/{len(submissions)}: ERROR - Failed for {student_name}: {str(e)}"
        )
        # Continue with other submissions

    # Final cleanup step
    self._update_progress(
      f"Finalization complete - all {len(submissions)} submissions processed")

  def _get_session_info(self) -> Dict:
    """Get session information from database"""
    session_repo = SessionRepository()
    session = session_repo.get_by_id(self.session_id)

    if not session:
      raise ValueError(f"Session {self.session_id} not found")

    # Handle older sessions without use_prod_canvas column
    # Note: SQLite stores booleans as INTEGER (0 or 1)
    use_prod = session.use_prod_canvas if session.use_prod_canvas is not None else 0

    log.info(
      f"Session {self.session_id}: use_prod_canvas from DB = {session.use_prod_canvas} (type: {type(session.use_prod_canvas)}), computed use_prod = {use_prod}"
    )
    if session.metadata and isinstance(session.metadata.get("quiz_yaml_text"), str):
      yaml_text = session.metadata.get("quiz_yaml_text")
      self.quiz_yaml_text = yaml_text if yaml_text.strip() else None

    return {
      "course_id": session.course_id,
      "assignment_id": session.assignment_id,
      "assignment_name": session.assignment_name,
      "session_name": getattr(session, "session_name", None),
      "canvas_points": session.canvas_points,
      "use_prod_canvas": use_prod
    }

  def _init_canvas(self, session_info: Dict):
    """Initialize Canvas interface"""
    use_prod = bool(session_info.get("use_prod_canvas", 0))
    log.info(
      f"Initializing Canvas interface: session_info['use_prod_canvas'] = {session_info.get('use_prod_canvas')} → use_prod = {use_prod}"
    )
    log.info(f"Calling CanvasInterface(prod={use_prod})")
    self.canvas_interface = CanvasInterface(prod=use_prod)
    log.info(
      f"Canvas interface initialized with URL: {self.canvas_interface.canvas_url}"
    )
    self.course = self.canvas_interface.get_course(session_info["course_id"])
    self.assignment = self.course.get_assignment(session_info["assignment_id"])

  def _get_submissions(self) -> List[Dict]:
    """Get all submissions for the session"""
    submission_repo = SubmissionRepository()
    problem_repo = ProblemRepository()

    # Get all submissions for this session
    submissions_list = submission_repo.get_by_session(self.session_id)

    submissions = []
    for sub in submissions_list:
      # Get problems for this submission
      problems_list = problem_repo.get_by_submission(sub.id)

      problems = []
      for prob in problems_list:
        problems.append({
          "problem_number": prob.problem_number,
          "score": prob.score or 0.0,
          "feedback": prob.feedback or '',
          "ai_reasoning": prob.ai_reasoning or '',
          "max_points": prob.max_points,
          "region_coords": prob.region_coords,
          "qr_encrypted_data": prob.qr_encrypted_data
        })

      submissions.append({
        "id": sub.id,
        "student_name": sub.student_name,
        "canvas_user_id": sub.canvas_user_id,
        "page_mappings": sub.page_mappings,
        "exam_pdf_data": sub.exam_pdf_data,
        "problems": problems
      })

    return submissions

  def _create_annotated_pdf(self, submission: Dict) -> Path:
    """
        Create annotated PDF by adding score stickers to the original PDF.
        Works directly with the original exam PDF instead of reconstructing from images.
        """
    output_path = self.temp_dir / f"exam_{submission['id']}.pdf"

    # Check if we have the original PDF data
    if not submission.get("exam_pdf_data"):
      log.error(
        f"No exam_pdf_data for submission {submission['id']}, cannot create annotated PDF"
      )
      raise ValueError(
        f"Missing exam_pdf_data for submission {submission['id']}")

    # Decode and open the original PDF
    pdf_bytes = base64.b64decode(submission["exam_pdf_data"])
    pdf_doc = fitz.open("pdf", pdf_bytes)

    # Add score stickers to each problem region
    for problem in submission["problems"]:
      region_coords = problem.get("region_coords")

      if not region_coords:
        log.warning(
          f"No region_coords for problem {problem['problem_number']}, skipping annotation"
        )
        continue

      page_number = region_coords["page_number"]
      region_y_start = region_coords["region_y_start"]
      region_y_end = region_coords["region_y_end"]

      # Get the page
      if page_number >= len(pdf_doc):
        log.warning(
          f"Page {page_number} out of range for submission {submission['id']}")
        continue

      page = pdf_doc[page_number]

      # Add score sticker in the problem region (upper right corner of the region)
      self._add_score_sticker_at_region(page, problem["score"], region_y_start,
                                        region_y_end)

    # Save the annotated PDF with compression
    pdf_doc.save(
      str(output_path),
      garbage=4,  # Maximum garbage collection
      deflate=True,  # Compress content streams
      clean=True  # Clean up unused objects
    )
    pdf_doc.close()

    log.info(
      f"Created annotated PDF for submission {submission['id']} at {output_path}"
    )
    return output_path

  def _add_score_sticker_at_region(self, page: fitz.Page, score: float,
                                   region_y_start: int, region_y_end: int):
    """
        Add a score sticker to the upper right corner of a problem region.

        Args:
            page: The PDF page to annotate
            score: The score to display
            region_y_start: Top Y coordinate of the problem region
            region_y_end: Bottom Y coordinate of the problem region
        """
    # Define sticker dimensions and position
    sticker_width = 60
    sticker_height = 30
    margin = 10

    # Position in upper right corner of the region
    page_width = page.rect.width
    x0 = page_width - sticker_width - margin
    y0 = region_y_start + margin
    x1 = page_width - margin
    y1 = region_y_start + margin + sticker_height

    # Create rectangle for background (black with 90% opacity)
    rect = fitz.Rect(x0, y0, x1, y1)
    page.draw_rect(rect, color=None, fill=(0, 0, 0), fill_opacity=0.9)

    # Add score text (blue, fully opaque)
    score_text = f"{score:.1f}"
    text_point = fitz.Point(x0 + sticker_width / 2,
                            y0 + sticker_height / 2 + 5)

    # Insert text centered in the sticker
    page.insert_text(
      text_point,
      score_text,
      fontsize=16,
      fontname="Helvetica-Bold",
      color=(0.2, 0.5, 1.0),  # Blue color
      render_mode=0  # Fill text (fully opaque)
    )

  def _generate_comments(self, submission: Dict) -> str:
    """Generate feedback comments for Canvas as a self-contained HTML document"""
    quiz_name = (submission.get("session_name")
                 or submission.get("assignment_name")
                 or "Grading Feedback")
    student_name = submission.get("student_name") or "Unknown Student"
    total_score = sum(p["score"] for p in submission["problems"])
    total_max = sum(
      p["max_points"] for p in submission["problems"] if p.get("max_points") is not None
    )
    score_summary = (f"{total_score:.2f} / {total_max:.2f}"
                     if total_max > 0 else f"{total_score:.2f}")

    problem_service = ProblemService()
    pdf_document = None
    if submission.get("exam_pdf_data"):
      try:
        pdf_bytes = base64.b64decode(submission["exam_pdf_data"])
        pdf_document = fitz.open("pdf", pdf_bytes)
      except Exception as e:
        log.warning(f"Failed to open PDF for submission {submission['id']}: {e}")

    problem_sections = []
    for problem in sorted(submission["problems"], key=lambda p: p["problem_number"]):
      score_display = (f"{problem['score']:.2f} / {problem['max_points']:.2f}"
                       if problem.get("max_points") is not None
                       else f"{problem['score']:.2f}")

      image_html = ""
      if pdf_document and problem.get("region_coords"):
        region = problem["region_coords"]
        try:
          image_base64, _ = problem_service.extract_image_from_document(
            pdf_document=pdf_document,
            start_page=region["page_number"],
            start_y=region["region_y_start"],
            end_page=region.get("end_page_number", region["page_number"]),
            end_y=region.get("end_region_y", region["region_y_end"]),
            start_y_pct=region.get("region_y_start_pct"),
            end_y_pct=region.get("region_y_end_pct"),
            end_page_y_pct=region.get("end_region_y_pct"),
            page_transforms=region.get("page_transforms"),
            dpi=150
          )
          image_html = (
            "<div class=\"problem-image\">"
            f"<img src=\"data:image/png;base64,{image_base64}\" "
            f"alt=\"Problem {problem['problem_number']} submission\" />"
            "</div>"
          )
        except Exception as e:
          log.warning(
            f"Failed to extract image for submission {submission['id']} "
            f"problem {problem['problem_number']}: {e}"
          )

      feedback_html = self._render_text_or_html(problem.get("feedback"))
      explanation_html = self._render_text_or_html(
        self._get_explanation_html(problem)
      )

      explanation_section = ""
      if explanation_html:
        explanation_section = (
          "<div class=\"problem-explanation auto-generated\">"
          "<div class=\"auto-badge\">Auto-generated</div>"
          "<h4>Explanation</h4>"
          f"{explanation_html}"
          "</div>"
        )

      problem_sections.append(
        "<details class=\"problem\">"
        f"<summary>Problem {problem['problem_number']} - Score {score_display}</summary>"
        "<div class=\"problem-body\">"
        f"{image_html}"
        "<div class=\"problem-meta\">"
        f"<div class=\"problem-score\">Score: {score_display}</div>"
        "</div>"
        "<div class=\"problem-feedback\">"
        "<h4>Feedback</h4>"
        f"{feedback_html if feedback_html else '<p>No feedback provided.</p>'}"
        "</div>"
        f"{explanation_section}"
        "</div>"
        "</details>"
      )

    if pdf_document:
      pdf_document.close()

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(quiz_name)} Feedback</title>
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']]
      }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <style>
    :root {{
      color-scheme: light;
      --ink: #111827;
      --muted: #6b7280;
      --accent: #2563eb;
      --border: #e5e7eb;
      --panel: #f9fafb;
    }}
    body {{
      margin: 0;
      padding: 24px;
      font-family: "Georgia", "Times New Roman", serif;
      color: var(--ink);
      background: #ffffff;
    }}
    header {{
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 2px solid var(--border);
    }}
    header h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      color: var(--accent);
    }}
    header .meta {{
      font-size: 14px;
      color: var(--muted);
    }}
    .summary {{
      margin-top: 12px;
      font-size: 16px;
      font-weight: 600;
    }}
    .problem {{
      border: 1px solid var(--border);
      border-radius: 12px;
      margin-bottom: 16px;
      overflow: hidden;
      background: var(--panel);
    }}
    .problem summary {{
      cursor: pointer;
      list-style: none;
      padding: 12px 16px;
      font-weight: 600;
      background: #ffffff;
      border-bottom: 1px solid var(--border);
    }}
    .problem summary::-webkit-details-marker {{
      display: none;
    }}
    .problem-body {{
      padding: 16px;
    }}
    .problem-image img {{
      max-width: 100%;
      height: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #ffffff;
    }}
    .problem-meta {{
      margin: 12px 0;
      font-size: 14px;
      color: var(--muted);
    }}
    .problem-feedback, .problem-explanation {{
      margin-top: 12px;
      padding: 12px;
      background: #ffffff;
      border-radius: 8px;
      border: 1px solid var(--border);
    }}
    .problem-explanation.auto-generated {{
      background: #fef9c3;
      border-color: #facc15;
    }}
    .auto-badge {{
      display: inline-block;
      padding: 2px 8px;
      margin-bottom: 6px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #713f12;
      background: #fde68a;
      border-radius: 999px;
    }}
    .problem-feedback h4, .problem-explanation h4 {{
      margin: 0 0 8px;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }}
    img {{
      max-width: 100%;
    }}
    code {{
      background: #f3f4f6;
      padding: 2px 4px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(quiz_name)}</h1>
    <div class="meta">Student: {html.escape(student_name)}</div>
    <div class="summary">Total Score: {html.escape(score_summary)}</div>
  </header>
  {"".join(problem_sections)}
</body>
</html>
"""

    return html_doc

  def _render_text_or_html(self, text: str) -> str:
    if not text:
      return ""
    if self._looks_like_html(text):
      return self._inline_local_images(text)
    rendered = self._render_markdown(text)
    if rendered:
      return self._inline_local_images(rendered)
    return f"<p>{html.escape(text).replace(chr(10), '<br>')}</p>"

  def _looks_like_html(self, text: str) -> bool:
    return bool(re.search(r"<(p|div|span|ul|ol|li|br|strong|em|table|img|h[1-6])[\s/>]", text, re.IGNORECASE))

  def _render_markdown(self, text: str) -> str:
    try:
      import markdown  # type: ignore
      return markdown.markdown(text, extensions=["extra", "sane_lists"])
    except Exception:
      return ""

  def _get_explanation_html(self, problem: Dict) -> str:
    qr_data = problem.get("qr_encrypted_data")
    if qr_data:
      try:
        from QuizGenerator.regenerate import regenerate_from_encrypted
        result = regenerate_from_encrypted(encrypted_data=qr_data,
                                           points=problem.get("max_points") or 0.0,
                                           image_mode="inline",
                                           yaml_text=self.quiz_yaml_text)
        explanation_html = result.get("explanation_html") or result.get("explanation_markdown")
        if explanation_html:
          return explanation_html
      except Exception as e:
        log.warning(
          f"Failed to regenerate explanation for problem {problem.get('problem_number')}: {e}"
        )
    return ""

  def _inline_local_images(self, html_text: str) -> str:
    def inline_src(src: str) -> str:
      if not src:
        return src
      lowered = src.lower()
      if lowered.startswith("data:") or lowered.startswith("http://") or lowered.startswith("https://"):
        return src
      path = src
      if lowered.startswith("file://"):
        path = src[7:]
      candidate_paths = [path]
      if not os.path.isabs(path):
        candidate_paths.append(os.path.join(os.getcwd(), path))
      file_path = next((p for p in candidate_paths if os.path.exists(p)), None)
      if not file_path:
        return src
      try:
        with open(file_path, "rb") as f:
          data = f.read()
        mime_type = mimetypes.guess_type(file_path)[0] or "image/png"
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
      except Exception:
        return src

    def replace_double(match):
      prefix, src, suffix = match.group(1), match.group(2), match.group(3)
      return f"{prefix}{inline_src(src)}{suffix}"

    def replace_single(match):
      prefix, src, suffix = match.group(1), match.group(2), match.group(3)
      return f"{prefix}{inline_src(src)}{suffix}"

    updated = re.sub(r'(<img[^>]+src=")([^"]*)(")', replace_double, html_text, flags=re.IGNORECASE)
    updated = re.sub(r"(<img[^>]+src=')([^']*)(')", replace_single, updated, flags=re.IGNORECASE)
    return updated

  def _upload_to_canvas(self, submission: Dict, pdf_path: Path, comments: str):
    """Upload graded exam and comments to Canvas"""
    # Create feedback object
    feedback = Feedback()
    feedback.comments = comments

    # Add PDF as attachment
    with open(pdf_path, 'rb') as f:
      pdf_bytes = f.read()

    # Canvas expects file-like objects
    pdf_file = io.BytesIO(pdf_bytes)
    pdf_file.name = f"graded_exam_{submission['student_name']}.pdf"

    # Upload to Canvas
    self.assignment.push_feedback(score=sum(p["score"]
                                            for p in submission["problems"]),
                                  comments=comments,
                                  attachments=[pdf_file],
                                  user_id=submission["canvas_user_id"],
                                  keep_previous_best=True,
                                  clobber_feedback=False)

  def _update_progress(self, message: str):
    """Update progress message in database and send SSE event"""
    # Increment step counter
    self.current_step += 1

    # Update database
    session_repo = SessionRepository()
    session = session_repo.get_by_id(self.session_id)
    if session:
      session.processing_message = message
      session_repo.update(session)

    # Send SSE progress event based on steps completed
    if self.total_steps > 0:
      progress_percent = min(100,
                             int((self.current_step / self.total_steps) * 100))
      try:
        # Use stored event loop reference (we're in a thread)
        asyncio.run_coroutine_threadsafe(
          sse.send_event(
            self.stream_id, "progress", {
              "total": self.total_submissions,
              "current": self.current_submission,
              "progress": progress_percent,
              "current_step": self.current_step,
              "total_steps": self.total_steps,
              "message": message
            }), self.event_loop)
      except Exception as e:
        log.error(f"Failed to send SSE event: {e}")

    log.info(message)
