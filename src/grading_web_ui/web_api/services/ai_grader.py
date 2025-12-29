"""
Service for AI-assisted grading of exam problems.
"""
import logging
import json
from typing import Dict, List, Optional, Tuple
from grading_web_ui.lms_interface.ai_helper import AI_Helper__Anthropic

from ..repositories import ProblemRepository, ProblemMetadataRepository, SubmissionRepository
from .problem_service import ProblemService

log = logging.getLogger(__name__)


class AIGraderService:
  """Handles AI-assisted autograding of exam problems"""

  def __init__(self):
    self.ai_helper = AI_Helper__Anthropic()
    self.problem_service = ProblemService()

  def extract_question_text(self, image_base64: str) -> str:
    """Extract question text from a problem image, ignoring handwritten content.

        Args:
            image_base64: Base64-encoded PNG image of the problem

        Returns:
            Extracted question text
        """
    message = (
      "Please extract the question text from this exam problem image. "
      "Ignore all handwritten text - only extract the printed/typed question text. "
      "Return only the question text without any additional commentary.")

    attachments = [("png", image_base64)]
    question_text, usage = self.ai_helper.query_ai(message,
                                                   attachments,
                                                   max_response_tokens=2000)

    log.info(
      f"Extracted question text ({usage['total_tokens']} tokens): {question_text[:100]}..."
    )
    return question_text.strip()

  def decipher_handwriting(self, image_base64: str) -> str:
    """Extract handwritten answer from a problem image.

        Args:
            image_base64: Base64-encoded PNG image of the problem

        Returns:
            Extracted handwritten text
        """
    message = (
      "Please extract ONLY the handwritten text from this exam problem image. "
      "Ignore the printed/typed question text - focus only on what the student wrote. "
      "Return only the handwritten text without any additional commentary.")

    attachments = [("png", image_base64)]
    handwriting_text, usage = self.ai_helper.query_ai(message,
                                                      attachments,
                                                      max_response_tokens=2000)

    log.info(
      f"Deciphered handwriting ({usage['total_tokens']} tokens): {handwriting_text[:100]}..."
    )
    return handwriting_text.strip()

  def generate_rubric(self,
                      question_text: str,
                      max_points: float,
                      example_answers: List[Dict] = None) -> str:
    """Generate a grading rubric for a question using AI and representative student answers.

        Args:
            question_text: The exam question
            max_points: Maximum points for this problem
            example_answers: Optional list of dicts with 'answer', 'score', 'feedback'
                           from manually graded examples

        Returns:
            Generated rubric text
        """
    # Build examples section
    examples_section = ""
    if example_answers and len(example_answers) > 0:
      examples_section = "\n\nRepresentative student answers with your manual grades:\n\n"
      for i, example in enumerate(example_answers, 1):
        examples_section += (f"Example {i}:\n"
                             f"Student Answer: {example['answer']}\n"
                             f"Your Score: {example['score']}/{max_points}\n"
                             f"Your Feedback: {example['feedback']}\n\n")

    message = (
      f"Create a grading rubric for this {max_points}-point exam problem.\n\n"
      f"Question:\n{question_text}"
      f"{examples_section}\n\n"
      f"Requirements:\n"
      f"- Break down into key components with integer point values (sum = {max_points})\n"
      f"- Be concise and specific (brief descriptions, no extra commentary)\n"
      f"- Align with the grading standards in the examples above\n"
      f"- Return ONLY valid JSON (no markdown, no code blocks):\n\n"
      f"{{\n"
      f'  "items": [\n'
      f'    {{"points": 2, "description": "Correct identification of X"}},\n'
      f'    {{"points": 3, "description": "Shows calculation for Y"}}\n'
      f"  ]\n"
      f"}}")

    response, usage = self.ai_helper.query_ai(message, [],
                                              max_response_tokens=2000)

    log.info(
      f"Generated rubric ({usage['total_tokens']} tokens): {response[:200]}..."
    )

    # Parse and validate JSON, then re-serialize to ensure clean format
    import json
    try:
      # Try to extract JSON if the AI wrapped it in markdown code blocks
      response_clean = response.strip()
      if response_clean.startswith('```'):
        # Extract content between code fences
        lines = response_clean.split('\n')
        json_lines = []
        in_code_block = False
        for line in lines:
          if line.startswith('```'):
            in_code_block = not in_code_block
            continue
          if in_code_block:
            json_lines.append(line)
        response_clean = '\n'.join(json_lines)

      rubric_data = json.loads(response_clean)

      # Validate structure
      if 'items' not in rubric_data or not isinstance(rubric_data['items'],
                                                      list):
        raise ValueError("Invalid rubric structure: missing 'items' array")

      # Return clean JSON string
      return json.dumps(rubric_data)

    except json.JSONDecodeError as e:
      log.error(f"Failed to parse rubric JSON: {e}. Raw response: {response}")
      # Fallback: return a simple valid JSON structure
      return json.dumps({
        "items": [{
          "points": max_points,
          "description": "Complete and correct answer"
        }]
      })

  def grade_problem(self,
                    question_text: str,
                    student_answer: str,
                    max_points: float,
                    grading_examples: List[Dict] = None,
                    rubric: str = None,
                    grading_notes: Optional[str] = None) -> Tuple[int, str]:
    """Grade a student's answer using AI.

        Args:
            question_text: The exam question
            student_answer: The student's handwritten answer
            max_points: Maximum points for this problem
            grading_examples: Optional list of dicts with 'answer', 'score', 'feedback' for few-shot prompting
            rubric: Optional grading rubric to follow

        Returns:
            Tuple of (score, feedback)
        """
    # Build rubric section
    rubric_section = ""
    if rubric:
      # Try to parse as JSON, fall back to treating as text
      import json
      try:
        rubric_data = json.loads(rubric)
        if 'items' in rubric_data and isinstance(rubric_data['items'], list):
          # Convert JSON rubric to readable format
          rubric_text = "Grading Rubric:\n"
          for item in rubric_data['items']:
            points = item.get('points', 0)
            description = item.get('description', '')
            rubric_text += f"- {description} ({points} points)\n"
          rubric_section = f"\n\n{rubric_text}\nPlease follow this rubric when grading.\n"
        else:
          rubric_section = f"\n\nGrading Rubric:\n{rubric}\n\nPlease follow this rubric when grading.\n"
      except json.JSONDecodeError:
        # Not JSON, treat as plain text
        rubric_section = f"\n\nGrading Rubric:\n{rubric}\n\nPlease follow this rubric when grading.\n"

    # Build grading notes section
    notes_section = ""
    if grading_notes:
      notes_section = (f"\n\nAdditional grading instructions:\n"
                       f"{grading_notes}\n")

    # Build few-shot examples section
    examples_section = ""
    if grading_examples and len(grading_examples) > 0:
      examples_section = "\n\nHere are examples of how you previously graded similar answers to this question:\n\n"
      for i, example in enumerate(grading_examples, 1):
        examples_section += (f"Example {i}:\n"
                             f"Student Answer: {example['answer']}\n"
                             f"Your Score: {example['score']}/{max_points}\n"
                             f"Your Feedback: {example['feedback']}\n\n")
      examples_section += "Please grade the current answer in a similar style and with similar standards.\n"

    message = (
      f"You are grading an exam problem worth {max_points} points.\n\n"
      f"Question:\n{question_text}"
      f"{rubric_section}"
      f"{examples_section}\n"
      f"{notes_section}\n"
      f"Current Student's Answer:\n{student_answer}\n\n"
      f"Please grade this answer and provide:\n"
      f"1. An INTEGER score out of {max_points} points (no decimals, round to nearest integer)\n"
      f"2. Clear and constructive feedback for the student\n\n"
      f"IMPORTANT: The score must be a whole number (integer) between 0 and {int(max_points)}.\n"
      f"IMPORTANT: The feedback should be addressed to the student, explain where their work went wrong, and suggest how to improve. Do NOT provide a full solution.\n"
      f"IMPORTANT: If the answer is blank, minimal, or shows no understanding, score it 0 and provide constructive feedback on how to approach the problem correctly. Focus on what a correct answer would include.\n\n"
      f"Format your response as:\n"
      f"SCORE: [integer]\n"
      f"FEEDBACK: [clear and constructive feedback for the student]")

    response, usage = self.ai_helper.query_ai(message, [],
                                              max_response_tokens=1000)

    log.info(
      f"AI grading response ({usage['total_tokens']} tokens): {response[:200]}..."
    )

    # Parse score and feedback from response
    score = 0  # Default to 0 if parsing fails
    feedback = response
    score_found = False

    try:
      lines = response.split('\n')
      for line in lines:
        if line.startswith('SCORE:'):
          score_str = line.replace('SCORE:', '').strip()
          # Extract number from string (handles "5" or "5.0" or "5 out of 10")
          import re
          score_match = re.search(r'(\d+\.?\d*)', score_str)
          if score_match:
            # Convert to int (round if decimal was provided)
            score = int(round(float(score_match.group(1))))
            score_found = True
        elif line.startswith('FEEDBACK:'):
          feedback = line.replace('FEEDBACK:', '').strip()
          # Get the rest of the response after FEEDBACK:
          feedback_start = response.find('FEEDBACK:') + len('FEEDBACK:')
          feedback = response[feedback_start:].strip()
          break
    except Exception as e:
      log.error(f"Failed to parse AI grading response: {e}")
      feedback = response

    # Ensure score is within valid range (0 to max_points)
    score = max(0, min(int(max_points), score))

    if not score_found:
      log.warning(
        f"No score found in AI response, defaulting to 0. Response: {response[:200]}"
      )

    return score, feedback

  def get_or_extract_question(self, session_id: int, problem_number: int,
                              sample_image_base64: str) -> str:
    """Get question text from metadata or extract it from a sample image.

        Args:
            session_id: Grading session ID
            problem_number: Problem number
            sample_image_base64: Sample problem image to extract from if not cached

        Returns:
            Question text
        """
    metadata_repo = ProblemMetadataRepository()

    # Check if question already extracted
    question_text = metadata_repo.get_question_text(session_id, problem_number)

    if question_text:
      log.info(f"Using cached question text for problem {problem_number}")
      return question_text

    # Extract question text
    log.info(f"Extracting question text for problem {problem_number}")
    question_text = self.extract_question_text(sample_image_base64)

    # Store in metadata
    metadata_repo.upsert_question_text(session_id, problem_number, question_text)

    return question_text

  def get_grading_examples(self,
                           session_id: int,
                           problem_number: int,
                           limit: int = 3) -> List[Dict]:
    """Fetch examples of previously graded submissions for few-shot prompting.

        Args:
            session_id: Grading session ID
            problem_number: Problem number
            limit: Maximum number of examples to return

        Returns:
            List of dicts with 'answer', 'score', 'feedback'
        """
    examples = []
    problem_repo = ProblemRepository()
    submission_repo = SubmissionRepository()

    # Get graded problems (exclude blanks and problems without feedback)
    rows = problem_repo.get_grading_examples(session_id, problem_number, limit)

    if not rows:
      log.info(f"No graded examples found for problem {problem_number}")
      return examples

    log.info(
      f"Found {len(rows)} graded examples for problem {problem_number}, deciphering..."
    )

    for row in rows:
      try:
        # Get image data - either directly or extract from PDF
        image_data = None
        if row["image_data"]:
          # Legacy: image_data is stored
          image_data = row["image_data"]
        elif row["region_coords"]:
          import json

          region_data = row["region_coords"]
          if isinstance(region_data, str):
            region_data = json.loads(region_data)

          submission = submission_repo.get_by_id(row["submission_id"])
          if submission and submission.exam_pdf_data:
            image_data = self.problem_service.extract_image_from_pdf_data(
              pdf_base64=submission.exam_pdf_data,
              page_number=region_data["page_number"],
              region_y_start=region_data["region_y_start"],
              region_y_end=region_data["region_y_end"],
              end_page_number=region_data.get("end_page_number"),
              end_region_y=region_data.get("end_region_y"),
              region_y_start_pct=region_data.get("region_y_start_pct"),
              region_y_end_pct=region_data.get("region_y_end_pct"),
              end_region_y_pct=region_data.get("end_region_y_pct"),
              page_transforms=region_data.get("page_transforms"),
              dpi=150
            )

        if not image_data:
          log.warning(
            f"No image data available for problem {row['id']}, skipping")
          continue

        # Decipher the handwriting from the example
        student_answer = self.decipher_handwriting(image_data)

        examples.append({
          'answer': student_answer,
          'score': row["score"],
          'feedback': row["feedback"]
        })
      except Exception as e:
        log.warning(f"Failed to decipher example submission: {e}")
        continue

    log.info(f"Successfully prepared {len(examples)} grading examples")
    return examples

  def autograde_problem(self,
                        session_id: int,
                        problem_number: int,
                        max_points: float = None,
                        progress_callback=None) -> Dict:
    """Autograde all ungraded submissions for a specific problem number.

        Args:
            session_id: Grading session ID
            problem_number: Problem number to grade
            max_points: Maximum points (optional, will query DB if not provided)
            progress_callback: Optional callback function(current, total, message)

        Returns:
            Dictionary with grading results
        """
    metadata_repo = ProblemMetadataRepository()
    problem_repo = ProblemRepository()

    # If max_points not provided, try to get from database
    if max_points is None:
      # Get max points for this problem - first check metadata
      max_points = metadata_repo.get_max_points(session_id, problem_number)

      if not max_points:
        # Fall back to max_points from problems table
        sample_problem = problem_repo.get_sample_for_problem_number(
          session_id, problem_number)

        if sample_problem and sample_problem.max_points:
          max_points = sample_problem.max_points

          # Save to metadata for future use
          metadata_repo.upsert_max_points(session_id, problem_number,
                                          max_points)

      if not max_points:
        raise ValueError(f"Max points not set for problem {problem_number}")

    # Get all ungraded problems for this problem number (include blanks for feedback)
    problems = problem_repo.get_ungraded_for_problem_number(
      session_id, problem_number)
    total = len(problems)

    if total == 0:
      return {"graded": 0, "message": "No ungraded problems found"}

    log.info(
      f"Autograding {total} problems for problem number {problem_number}")

    submission_repo = SubmissionRepository()

    # Get question text (use first problem's image as sample)
    # Extract image from first problem
    first_problem = problems[0]
    first_image_data = None
    if first_problem["image_data"]:
      first_image_data = first_problem["image_data"]
    elif first_problem["region_coords"]:
      import json

      region_data = first_problem["region_coords"]
      if isinstance(region_data, str):
        region_data = json.loads(region_data)

      submission = submission_repo.get_by_id(first_problem["submission_id"])
      if submission and submission.exam_pdf_data:
        first_image_data = self.problem_service.extract_image_from_pdf_data(
          pdf_base64=submission.exam_pdf_data,
          page_number=region_data["page_number"],
          region_y_start=region_data["region_y_start"],
          region_y_end=region_data["region_y_end"],
          end_page_number=region_data.get("end_page_number"),
          end_region_y=region_data.get("end_region_y"),
          region_y_start_pct=region_data.get("region_y_start_pct"),
          region_y_end_pct=region_data.get("region_y_end_pct"),
          end_region_y_pct=region_data.get("end_region_y_pct"),
          page_transforms=region_data.get("page_transforms"),
          dpi=150
        )

    question_text = self.get_or_extract_question(session_id, problem_number,
                                                 first_image_data)
    grading_notes = metadata_repo.get_ai_grading_notes(session_id,
                                                       problem_number)

    if progress_callback:
      progress_callback(0, total,
                        f"Extracted question for problem {problem_number}")

    # Get rubric from metadata if available
    rubric = metadata_repo.get_grading_rubric(session_id, problem_number)
    if rubric:
      log.info(f"Using rubric for problem {problem_number}")

      # Get grading examples for few-shot prompting
      if progress_callback:
        progress_callback(
          0, total, f"Fetching grading examples for problem {problem_number}")

      grading_examples = self.get_grading_examples(session_id,
                                                   problem_number,
                                                   limit=3)

      if progress_callback:
        if len(grading_examples) > 0:
          progress_callback(0, total,
                            f"Found {len(grading_examples)} grading examples")
        else:
          progress_callback(0, total,
                            f"No grading examples found, proceeding without")

      # Grade each problem
      graded_count = 0
      for idx, problem in enumerate(problems, 1):
        try:
          if progress_callback:
            progress_callback(
              idx, total,
              f"Autograding problem {problem_number}, submission {idx}/{total}"
            )

          # Get image data - either directly or extract from PDF
          image_data = None
          if problem["image_data"]:
            image_data = problem["image_data"]
          elif problem["region_coords"]:
            import json

            region_data = problem["region_coords"]
            if isinstance(region_data, str):
              region_data = json.loads(region_data)

            submission = submission_repo.get_by_id(problem["submission_id"])
            if submission and submission.exam_pdf_data:
              image_data = self.problem_service.extract_image_from_pdf_data(
                pdf_base64=submission.exam_pdf_data,
                page_number=region_data["page_number"],
                region_y_start=region_data["region_y_start"],
                region_y_end=region_data["region_y_end"],
                end_page_number=region_data.get("end_page_number"),
                end_region_y=region_data.get("end_region_y"),
                region_y_start_pct=region_data.get("region_y_start_pct"),
                region_y_end_pct=region_data.get("region_y_end_pct"),
                end_region_y_pct=region_data.get("end_region_y_pct"),
                page_transforms=region_data.get("page_transforms"),
                dpi=150
              )

          if not image_data:
            log.warning(
              f"No image data available for problem {problem['id']}, skipping")
            continue

          # For blank submissions, use placeholder text instead of deciphering
          if problem["is_blank"]:
            student_answer = "[No answer provided]"
            log.info(
              f"Problem {problem['id']} marked as blank, skipping handwriting extraction"
            )
          else:
            # Decipher handwriting for non-blank submissions
            student_answer = self.decipher_handwriting(image_data)

          # Grade the answer with rubric and examples
          score, feedback = self.grade_problem(
            question_text,
            student_answer,
            max_points,
            grading_examples=grading_examples,
            rubric=rubric,
            grading_notes=grading_notes)

          # Update problem with AI suggestion (score and feedback ready for instructor review)
          problem_repo.update_ai_grade(problem["id"], score, feedback)

          graded_count += 1
          log.info(f"AI graded problem {problem['id']}: {score}/{max_points}")

        except Exception as e:
          log.error(f"Failed to autograde problem {problem['id']}: {e}",
                    exc_info=True)
          continue

      if progress_callback:
        progress_callback(total, total,
                          f"Completed autograding {graded_count} problems")

    return {
      "graded": graded_count,
      "total": total,
      "question_text": question_text,
      "message": f"AI graded {graded_count}/{total} problems"
    }

  def autograde_problem_image_only(self,
                                   session_id: int,
                                   problem_number: int,
                                   settings: Dict,
                                   progress_callback=None) -> Dict:
    """Autograde submissions using images directly (no text extraction)."""
    metadata_repo = ProblemMetadataRepository()
    problem_repo = ProblemRepository()
    submission_repo = SubmissionRepository()

    max_points = metadata_repo.get_max_points(session_id, problem_number)
    if not max_points:
      sample_problem = problem_repo.get_sample_for_problem_number(
        session_id, problem_number)
      if sample_problem and sample_problem.max_points:
        max_points = sample_problem.max_points
        metadata_repo.upsert_max_points(session_id, problem_number, max_points)

    if not max_points:
      raise ValueError(f"Max points not set for problem {problem_number}")

    question_text = metadata_repo.get_question_text(session_id, problem_number)
    default_feedback = None
    if settings.get("include_default_feedback"):
      feedback, _ = metadata_repo.get_default_feedback(session_id,
                                                       problem_number)
      default_feedback = feedback
    grading_notes = metadata_repo.get_ai_grading_notes(session_id,
                                                       problem_number)

    problems = problem_repo.get_ungraded_for_problem_number(
      session_id, problem_number)
    total = len(problems)
    if total == 0:
      return {"graded": 0, "total": 0, "message": "No ungraded problems found"}

    batch_size = self._parse_batch_size(settings.get("batch_size"), total)
    dpi = self._map_image_quality(settings.get("image_quality"))
    dry_run = bool(settings.get("dry_run"))

    if progress_callback:
      progress_callback(
        0, total,
        f"Image-only autograding for problem {problem_number} (batch size {batch_size})"
      )

    graded_count = 0
    processed = 0

    if dry_run:
      sample_count = min(10, total)
      summary = {
        "total": total,
        "batch_size": batch_size,
        "image_quality": settings.get("image_quality"),
        "dpi": dpi,
        "include_answer": bool(settings.get("include_answer")),
        "include_default_feedback": bool(settings.get("include_default_feedback")),
        "sample_count": sample_count,
        "items": [{
          "problem_id": row["id"],
          "submission_id": row["submission_id"],
          "answer_available": bool(row.get("qr_encrypted_data")),
          "is_blank": bool(row.get("is_blank"))
        } for row in problems[:sample_count]]
      }
      log.info("Image-only autograding dry run: %s", summary)
      if progress_callback:
        progress_callback(total, total,
                          f"Dry run complete for {total} submissions")
      return {
        "graded": 0,
        "total": total,
        "message": f"Dry run complete for {total} submissions"
      }

    for batch_start in range(0, total, batch_size):
      batch = problems[batch_start:batch_start + batch_size]
      batch_items = []
      attachments = []

      for row in batch:
        image_data = self._extract_problem_image(row, submission_repo, dpi)
        if not image_data:
          log.warning(
            f"No image data available for problem {row['id']}, skipping")
          processed += 1
          if progress_callback:
            progress_callback(
              processed, total,
              f"Skipped problem {row['id']} (no image data)")
          continue

        answer_text = None
        if settings.get("include_answer"):
          answer_text = self._get_reference_answer(row)

        batch_items.append({
          "problem_id": row["id"],
          "submission_id": row["submission_id"],
          "answer_text": answer_text
        })
        attachments.append(("png", image_data))

      if not batch_items:
        continue

      if progress_callback:
        progress_callback(
          processed, total,
          f"Submitting batch {batch_start // batch_size + 1} ({len(batch_items)} items)"
        )

      results = self._grade_image_batch(problem_number, max_points,
                                        question_text, default_feedback,
                                        grading_notes,
                                        batch_items, attachments)
      if len(results) < len(batch_items) and len(batch_items) > 1:
        log.warning(
          "Batch returned %s/%s results; retrying individually for missing items",
          len(results), len(batch_items))
        returned_ids = {result.get("problem_id") for result in results if result.get("problem_id")}
        for idx, item in enumerate(batch_items, 1):
          if item["problem_id"] in returned_ids:
            continue
          retry_results = self._grade_image_batch(problem_number,
                                                  max_points,
                                                  question_text,
                                                  default_feedback,
                                                  grading_notes, [item],
                                                  [attachments[idx - 1]])
          results.extend(retry_results)

      seen_ids = set()
      for result in results:
        problem_id = result.get("problem_id")
        if not problem_id:
          continue
        seen_ids.add(problem_id)
        is_blank = bool(result.get("is_blank"))
        score_value = result.get("score")
        if score_value == "-":
          is_blank = True
        if is_blank:
          feedback = (result.get("feedback") or "").strip() or "No answer detected. Please show your work."
          problem_repo.update_ai_blank(problem_id, feedback=feedback)
        else:
          score = self._coerce_score(score_value, max_points)
          feedback = (result.get("feedback") or "").strip()
          problem_repo.update_ai_grade(problem_id, score, feedback)
        graded_count += 1
        processed += 1
        if progress_callback:
          progress_callback(
            processed, total,
            f"Image-only autograding {processed}/{total}")

      for item in batch_items:
        if item["problem_id"] not in seen_ids:
          processed += 1
          if progress_callback:
            progress_callback(
              processed, total,
              f"No AI result for problem {item['problem_id']}, skipping")

    if progress_callback:
      progress_callback(total, total,
                        f"Completed image-only autograding {graded_count} problems")

    return {
      "graded": graded_count,
      "total": total,
      "message": f"AI graded {graded_count}/{total} problems (image-only)"
    }

  def _extract_problem_image(self, problem_row: Dict,
                             submission_repo: SubmissionRepository,
                             dpi: int) -> Optional[str]:
    if problem_row.get("image_data"):
      return problem_row["image_data"]
    if problem_row.get("region_coords"):
      region_data = problem_row["region_coords"]
      if isinstance(region_data, str):
        region_data = json.loads(region_data)

      submission = submission_repo.get_by_id(problem_row["submission_id"])
      if submission and submission.exam_pdf_data:
        return self.problem_service.extract_image_from_pdf_data(
          pdf_base64=submission.exam_pdf_data,
          page_number=region_data["page_number"],
          region_y_start=region_data["region_y_start"],
          region_y_end=region_data["region_y_end"],
          end_page_number=region_data.get("end_page_number"),
          end_region_y=region_data.get("end_region_y"),
          region_y_start_pct=region_data.get("region_y_start_pct"),
          region_y_end_pct=region_data.get("region_y_end_pct"),
          end_region_y_pct=region_data.get("end_region_y_pct"),
          page_transforms=region_data.get("page_transforms"),
          dpi=dpi
        )
    return None

  def _get_reference_answer(self, problem_row: Dict) -> Optional[str]:
    encrypted_data = problem_row.get("qr_encrypted_data")
    if not encrypted_data:
      return None

    try:
      from QuizGenerator.regenerate import regenerate_from_encrypted
    except ImportError:
      log.warning("QuizGenerator not available; skipping reference answer")
      return None

    try:
      result = regenerate_from_encrypted(
        encrypted_data=encrypted_data,
        points=problem_row.get("max_points") or 0.0)
    except Exception as e:
      log.warning(f"Failed to regenerate answer: {e}")
      return None

    answers = []
    for key, answer_obj in result.get("answer_objects", {}).items():
      value = str(answer_obj.value)
      if hasattr(answer_obj, "tolerance") and answer_obj.tolerance is not None:
        value = f"{value} (tol={answer_obj.tolerance})"
      answers.append(f"{key}: {value}")
    if not answers:
      return None
    return "; ".join(answers)

  def _grade_image_batch(self, problem_number: int, max_points: float,
                         question_text: Optional[str],
                         default_feedback: Optional[str],
                         grading_notes: Optional[str],
                         batch_items: List[Dict],
                         attachments: List[Tuple[str, str]]) -> List[Dict]:
    message_lines = [
      f"You are grading {len(batch_items)} student submissions for problem {problem_number}.",
      f"The problem is worth {int(max_points)} points.",
      "Each submission is provided as an image attachment.",
      "Return ONLY valid JSON with this shape (no prose, no markdown, no code fences):",
      '{"results":[{"problem_id":123,"score":4,"feedback":"...","is_blank":false}]}',
      "Scores must be integers between 0 and the max points.",
      "If a reference answer is provided, treat it as authoritative and grade strictly against it.",
      "If the student's work matches the reference answer, award full credit.",
      "If the answer is blank or minimal, set is_blank=true and set score to \"-\".",
      "Equations are acceptable if they are correct.",
      "If the answer is correct, set feedback to an empty string.",
      "If the answer is incorrect, keep feedback concise (1-2 sentences, <= 200 characters), addressed to the student, and focused on what went wrong (no full solution).",
      "Images are attached in the SAME ORDER as the submission mapping below. Image 1 corresponds to mapping line 1, etc.",
      "Return results in the same order as the mapping.",
      "Do NOT reuse numbers from other submissions. Only use values visible in the current image and/or the provided reference answer.",
      "If the reference answer appears to be for a different question than the image, note the mismatch in feedback and score conservatively."
    ]

    if question_text:
      message_lines.append(f"\nQuestion text (if helpful):\n{question_text}")

    if default_feedback:
      message_lines.append(
        f"\nDefault feedback (use if appropriate):\n{default_feedback}")
    if grading_notes:
      message_lines.append(
        f"\nAdditional grading instructions:\n{grading_notes}")

    message_lines.append("\nSubmission mapping (image order):")
    for idx, item in enumerate(batch_items, 1):
      answer_text = item.get("answer_text")
      answer_line = "Reference answer: unavailable"
      if answer_text:
        answer_line = f"Reference answer (authoritative): {answer_text}"
      message_lines.append(
        f"Image {idx}: problem_id={item['problem_id']} submission_id={item['submission_id']} ({answer_line})"
      )

    response, usage = self.ai_helper.query_ai("\n".join(message_lines),
                                              attachments,
                                              max_response_tokens=8000,
                                              max_retries=0)
    log.info(
      f"Image-only grading response ({usage['total_tokens']} tokens): {response[:200]}..."
    )

    parsed = self._parse_json_response(response)
    results = parsed.get("results", []) if isinstance(parsed, dict) else []
    if not isinstance(results, list):
      return []
    return results

  def _parse_json_response(self, response: str) -> Dict:
    response_clean = response.strip()
    if "```" in response_clean:
      lines = response_clean.split("\n")
      json_lines = []
      in_code_block = False
      for line in lines:
        if line.startswith("```"):
          in_code_block = not in_code_block
          continue
        if in_code_block:
          json_lines.append(line)
      if json_lines:
        response_clean = "\n".join(json_lines).strip()

    try:
      return json.loads(response_clean)
    except json.JSONDecodeError:
      start = response_clean.find("{")
      if start != -1:
        balanced_end = self._find_last_balanced_brace(response_clean, start)
        if balanced_end is not None:
          try:
            return json.loads(response_clean[start:balanced_end + 1])
          except json.JSONDecodeError:
            pass
      end = response_clean.rfind("}")
      if start != -1 and end != -1 and end > start:
        try:
          return json.loads(response_clean[start:end + 1])
        except json.JSONDecodeError:
          pass
    log.warning(f"Failed to parse JSON response: {response[:200]}")
    return {}

  def _find_last_balanced_brace(self, text: str, start_index: int) -> Optional[int]:
    depth = 0
    last_balanced = None
    for idx in range(start_index, len(text)):
      char = text[idx]
      if char == "{":
        depth += 1
      elif char == "}":
        depth -= 1
        if depth == 0:
          last_balanced = idx
      if depth < 0:
        break
    return last_balanced

  def _parse_batch_size(self, value: Optional[str], total: int) -> int:
    if not value:
      return 1
    if isinstance(value, str) and value.lower() == "all":
      return max(1, total)
    try:
      parsed = int(value)
      return max(1, parsed)
    except (TypeError, ValueError):
      return 1

  def _map_image_quality(self, value: Optional[str]) -> int:
    quality = (value or "medium").lower()
    if quality == "low":
      return 150
    if quality == "high":
      return 600
    return 300

  def _coerce_score(self, score_value: Optional[object],
                    max_points: float) -> int:
    try:
      score = int(round(float(score_value)))
    except (TypeError, ValueError):
      score = 0
    return max(0, min(int(max_points), score))
