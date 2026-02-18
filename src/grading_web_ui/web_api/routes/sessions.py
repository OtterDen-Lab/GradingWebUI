"""
Session management endpoints.
"""
from fastapi import APIRouter, HTTPException, Response, UploadFile, File, Depends
from fastapi.responses import StreamingResponse
from typing import List, Optional
import json
import io
from datetime import datetime

import logging
import json
import base64
import fitz
from ..services.qr_scanner import QRScanner
from ..services.exam_processor import ExamProcessor

from ..models import (
  SessionCreate,
  SessionResponse,
  SessionStatsResponse,
  ProblemStatsResponse,
  SessionStatusUpdate,
  SessionStatusChange,
  SessionCloneRequest,
  SessionCompareExportRequest,
)
from ..database import get_db_connection  # Still needed for unrefactored endpoints
from ..repositories import SessionRepository, SubmissionRepository, ProblemRepository, ProblemMetadataRepository
from ..domain.common import SessionStatus as DomainSessionStatus
from ..domain.session import GradingSession
from lms_interface.canvas_interface import CanvasInterface
from ..services.qr_scanner import QRScanner
from ..auth import get_current_user, require_instructor, require_session_access
import os

router = APIRouter()
MOCK_ROSTER_ENV = "ALLOW_MOCK_ROSTER"


def mock_roster_enabled() -> bool:
  return os.getenv(MOCK_ROSTER_ENV, "").lower() in ("1", "true", "yes")


def session_to_response(session: GradingSession) -> SessionResponse:
  response = SessionResponse.model_validate(session)
  is_mock = bool(session.metadata and session.metadata.get("mock_roster"))
  ai_enabled = True
  if session.metadata and "ai_name_extraction" in session.metadata:
    ai_enabled = bool(session.metadata.get("ai_name_extraction"))
  session_name = None
  if session.metadata and session.metadata.get("session_name"):
    session_name = session.metadata.get("session_name")
  return response.model_copy(update={
    "mock_roster": is_mock,
    "ai_name_extraction": ai_enabled,
    "session_name": session_name
  })


@router.get("/mock-roster-enabled")
async def get_mock_roster_enabled(
  current_user: dict = Depends(get_current_user)
):
  return {"enabled": mock_roster_enabled()}


@router.post("", response_model=SessionResponse)
async def create_session(
  session: SessionCreate,
  current_user: dict = Depends(require_instructor)
):
  """Create a new grading session (instructor only)"""
  repo = SessionRepository()

  if session.use_mock_roster and not mock_roster_enabled():
    raise HTTPException(
      status_code=400,
      detail=f"Mock roster sessions are disabled. Set {MOCK_ROSTER_ENV}=true to enable."
    )

  # Create domain object
  metadata = {}
  if session.use_mock_roster or not session.use_ai_name_extraction:
    metadata["mock_roster"] = bool(session.use_mock_roster)
    metadata["ai_name_extraction"] = bool(session.use_ai_name_extraction)
  if session.session_name:
    metadata["session_name"] = session.session_name
  if not metadata:
    metadata = None
  new_session = GradingSession(
    id=0,  # Will be set by DB
    assignment_id=session.assignment_id,
    assignment_name=session.assignment_name,
    course_id=session.course_id,
    course_name=session.course_name,
    status=DomainSessionStatus.PREPROCESSING,
    canvas_points=session.canvas_points,
    use_prod_canvas=session.use_prod_canvas,
    metadata=metadata,
    created_at=datetime.now(),
    updated_at=datetime.now(),
    total_exams=0,
    processed_exams=0,
    matched_exams=0,
    processing_message=None
  )

  created_session = repo.create(new_session)
  return session_to_response(created_session)


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
  session_id: int,
  current_user: dict = Depends(require_session_access())
):
  """Get session details (requires session access)"""
  from ..repositories import SessionRepository

  repo = SessionRepository()
  session = repo.get_by_id(session_id)

  if not session:
    raise HTTPException(status_code=404, detail="Session not found")

  # Convert domain model to API response model
  return session_to_response(session)


@router.patch("/{session_id}/status")
async def update_session_status(
  session_id: int,
  status_update: SessionStatusChange,
  current_user: dict = Depends(require_session_access())
):
  """Update session status (requires session access)"""
  from ..repositories import SessionRepository, ProblemRepository, FeedbackTagRepository
  from ..domain.common import SessionStatus as DomainSessionStatus

  repo = SessionRepository()

  # Verify session exists
  if not repo.exists(session_id):
    raise HTTPException(status_code=404, detail="Session not found")

  # Update status using repository
  # Convert API enum to domain enum
  domain_status = DomainSessionStatus(status_update.status.value)
  repo.update_status(session_id, domain_status)

  # If transitioning to 'ready', create default feedback tags for all problems
  if domain_status == DomainSessionStatus.READY:
    problem_repo = ProblemRepository()
    tag_repo = FeedbackTagRepository()

    # Get all distinct problem numbers in this session
    problem_numbers = problem_repo.get_distinct_problem_numbers(session_id)

    # Create default "Show work" tag for each problem
    for problem_num in problem_numbers:
      try:
        tag_repo.create(
          session_id=session_id,
          problem_number=problem_num,
          short_name="Show work",
          comment_text="Please show your work, it helps me find partial credit."
        )
      except Exception as e:
        # Tag might already exist (e.g., if re-importing session) - that's okay, skip it
        import logging
        log = logging.getLogger(__name__)
        log.debug(f"Skipped creating default tag for problem {problem_num}: {e}")

  return {
    "status": "updated",
    "session_id": session_id,
    "new_status": status_update.status
  }


@router.get("", response_model=List[SessionResponse])
async def list_sessions(current_user: dict = Depends(get_current_user)):
  """List grading sessions (instructors see all, TAs see only assigned)"""
  repo = SessionRepository()

  # Instructors see all sessions
  if current_user["role"] == "instructor":
    sessions = repo.list_all()
  else:
    # TAs only see sessions they're assigned to
    from ..repositories.session_assignment_repository import SessionAssignmentRepository
    assignment_repo = SessionAssignmentRepository()
    assigned_session_ids = assignment_repo.get_assigned_sessions(current_user["user_id"])
    sessions = [repo.get_by_id(sid) for sid in assigned_session_ids]
    sessions = [s for s in sessions if s is not None]  # Filter out None values

  return [session_to_response(session) for session in sessions]


@router.get("/{session_id}/stats", response_model=SessionStatsResponse)
async def get_session_stats(
  session_id: int,
  current_user: dict = Depends(require_session_access())
):
  """Get grading statistics for a session (requires session access)"""
  import statistics
  import logging

  problem_repo = ProblemRepository()
  metadata_repo = ProblemMetadataRepository()
  log = logging.getLogger(__name__)

  # Get overall stats
  overall_stats = problem_repo.get_session_overall_stats(session_id)
  if overall_stats["total_problems"] == 0:
    session_repo = SessionRepository()
    if not session_repo.exists(session_id):
      raise HTTPException(status_code=404, detail="Session not found")
    return SessionStatsResponse(
      session_id=session_id,
      total_submissions=0,
      total_problems=0,
      problems_graded=0,
      problems_remaining=0,
      progress_percentage=0,
      problem_stats=[],
    )

  total_submissions = overall_stats["total_submissions"]
  total_problems = overall_stats["total_problems"]
  problems_graded = overall_stats["problems_graded"]
  problems_remaining = total_problems - problems_graded
  progress = (problems_graded / total_problems * 100) if total_problems > 0 else 0

  # Get per-problem stats
  problem_numbers = problem_repo.get_distinct_problem_numbers(session_id)
  problem_stats = []

  for problem_num in problem_numbers:
    # Get scores and blank count
    scores, num_blank = problem_repo.get_problem_scores_and_blanks(
      session_id, problem_num)

    # Get max_points for this problem (default to 8 if not set)
    max_points = metadata_repo.get_max_points(session_id, problem_num)
    if max_points is None:
      max_points = 8.0

    # Get counts
    counts = problem_repo.get_counts_for_problem_number(session_id, problem_num)
    num_total = counts["total"]
    num_graded = counts["graded"]
    num_blank_ungraded = counts["ungraded_blank"]
    num_blank_total = num_blank + num_blank_ungraded

    # Debug log
    log.info(
      f"[STATS] Problem {problem_num}: total={num_total}, graded={num_graded}, blank_ungraded={num_blank_ungraded}, blank_total={num_blank_total}"
    )

    # Calculate statistics
    avg_score = statistics.mean(scores) if scores else None
    min_score = min(scores) if scores else None
    max_score = max(scores) if scores else None
    median_score = statistics.median(scores) if scores else None
    stddev_score = statistics.stdev(scores) if len(scores) > 1 else None

    # Calculate normalized mean and stddev (0-1 scale based on max_points)
    mean_normalized = None
    stddev_normalized = None
    if avg_score is not None and max_points is not None and max_points > 0:
      mean_normalized = avg_score / max_points
    if stddev_score is not None and max_points is not None and max_points > 0:
      stddev_normalized = stddev_score / max_points

    # Calculate percentage blank
    pct_blank = (num_blank / num_graded * 100) if num_graded > 0 else None

    problem_stats.append(
      ProblemStatsResponse(
        problem_number=problem_num,
        avg_score=avg_score,
        min_score=min_score,
        max_score=max_score,
        median_score=median_score,
        stddev_score=stddev_score,
        mean_normalized=mean_normalized,
        stddev_normalized=stddev_normalized,
        pct_blank=pct_blank,
        num_blank=num_blank,
        num_blank_ungraded=num_blank_ungraded,
        num_graded=num_graded,
        num_total=num_total,
        max_points=max_points,
      ))

  return SessionStatsResponse(
    session_id=session_id,
    total_submissions=total_submissions,
    total_problems=total_problems,
    problems_graded=problems_graded,
    problems_remaining=problems_remaining,
    progress_percentage=progress,
    problem_stats=problem_stats,
  )


@router.get("/{session_id}/problem-numbers")
async def get_problem_numbers(
  session_id: int,
  current_user: dict = Depends(require_session_access())
):
  """Get list of distinct problem numbers for a session (requires session access)"""
  repo = ProblemRepository()
  problem_numbers = repo.get_distinct_problem_numbers(session_id)
  return {"problem_numbers": problem_numbers}


@router.get("/{session_id}/student-scores")
async def get_student_scores(
  session_id: int,
  current_user: dict = Depends(require_session_access())
):
  """Get aggregated scores for all students in a session (requires session access)"""
  submission_repo = SubmissionRepository()
  students = submission_repo.get_student_scores(session_id)
  return {"students": students}


@router.get("/{session_id}/submissions/{submission_id}/problems")
async def get_submission_problems(
  session_id: int,
  submission_id: int,
  current_user: dict = Depends(require_session_access())
):
  """Get all problems for a specific submission (requires session access)"""
  from ..models import ProblemResponse

  submission_repo = SubmissionRepository()
  submission = submission_repo.get_by_id(submission_id)

  if not submission or submission.session_id != session_id:
    raise HTTPException(status_code=404,
                        detail="Submission not found in this session")

  pdf_base64 = submission.exam_pdf_data

  # Get all problems for this submission
  problem_repo = ProblemRepository()
  problems_list = problem_repo.get_by_submission(submission_id)

  problems = []
  from ..services.problem_service import ProblemService
  problem_service = ProblemService()

  for problem in problems_list:
    # Extract image from PDF using region coords
    region_coords = problem.region_coords
    start_page = region_coords["page_number"]
    start_y = region_coords["region_y_start"]
    end_page = region_coords.get("end_page_number", start_page)
    end_y = region_coords["region_y_end"]

    try:
      problem_image_base64 = problem_service.extract_image_from_pdf_data(
        pdf_base64,
        start_page,
        start_y,
        end_y,
        end_page,
        end_y,
        region_y_start_pct=region_coords.get("region_y_start_pct"),
        region_y_end_pct=region_coords.get("region_y_end_pct"),
        end_region_y_pct=region_coords.get("end_region_y_pct"),
        page_transforms=region_coords.get("page_transforms"),
        dpi=150)
    except Exception as e:
      import logging
      log = logging.getLogger(__name__)
      log.error(f"Failed to extract image for problem {problem.id}: {e}")
      problem_image_base64 = ""

    problems.append(
      ProblemResponse(
        id=problem.id,
        problem_number=problem.problem_number,
        submission_id=problem.submission_id,
        image_data=problem_image_base64,
        score=problem.score,
        feedback=problem.feedback,
        graded=problem.graded,
        is_blank=problem.is_blank,
        blank_confidence=problem.blank_confidence,
        blank_method=problem.blank_method,
        blank_reasoning=problem.blank_reasoning,
        current_index=0,  # Not applicable for this endpoint
        total_count=0,  # Not applicable for this endpoint
        ungraded_blank=0,  # Not applicable for this endpoint
        ungraded_nonblank=0,  # Not applicable for this endpoint
        has_qr_data=False  # Not needed for debug view
      ))

  return problems


@router.get("/{session_id}/canvas-info")
async def get_canvas_info(
  session_id: int,
  current_user: dict = Depends(require_session_access())
):
  """Get Canvas course and assignment information (requires session access)"""
  repo = SessionRepository()
  session = repo.get_by_id(session_id)

  if not session:
    raise HTTPException(status_code=404, detail="Session not found")

  if session.metadata and session.metadata.get("mock_roster"):
    return {
      "course_id": session.course_id,
      "course_name": session.course_name,
      "assignment_id": session.assignment_id,
      "assignment_name": session.assignment_name,
      "canvas_url": "",
      "environment": "mock"
    }

  use_prod = session.use_prod_canvas
  canvas = CanvasInterface(prod=use_prod)

  # Get course and assignment to construct URL
  course = canvas.get_course(session.course_id)
  assignment = course.get_assignment(session.assignment_id)

  # Get base URL from Canvas interface
  # Remove trailing slash and /api/v1 if present
  base_url = str(canvas.canvas._Canvas__requester.base_url)
  if base_url.endswith('/api/v1'):
    base_url = base_url[:-7]
  base_url = base_url.rstrip('/')

  # Construct Canvas URL
  canvas_url = f"{base_url}/courses/{session.course_id}/assignments/{session.assignment_id}"

  return {
    "course_id": session.course_id,
    "course_name": session.course_name,
    "assignment_id": session.assignment_id,
    "assignment_name": session.assignment_name,
    "canvas_url": canvas_url,
    "environment": "production" if use_prod else "development"
  }


@router.put("/{session_id}/canvas-config")
async def update_canvas_config(
  session_id: int,
  course_id: int,
  assignment_id: int,
  use_prod: bool = False,
  current_user: dict = Depends(require_instructor)
):
  """Update Canvas configuration for a session (instructor only)"""
  session_repo = SessionRepository()
  session = session_repo.get_by_id(session_id)
  if not session:
    raise HTTPException(status_code=404, detail="Session not found")
  if session.metadata and session.metadata.get("mock_roster"):
    raise HTTPException(
      status_code=400,
      detail="Mock roster sessions cannot update Canvas configuration."
    )

  # Get course and assignment details from Canvas
  canvas_interface = CanvasInterface(prod=use_prod)
  try:
    course = canvas_interface.get_course(course_id)
    assignment = course.get_assignment(assignment_id)

    repo = SessionRepository()
    session = repo.get_by_id(session_id)
    if not session:
      raise HTTPException(status_code=404, detail="Session not found")

    # Update session fields
    session.course_id = course_id
    session.course_name = course.name
    session.assignment_id = assignment_id
    session.assignment_name = assignment.name
    session.use_prod_canvas = use_prod
    repo.update(session)

    return {
      "status": "updated",
      "course_id": course_id,
      "course_name": course.name,
      "assignment_id": assignment_id,
      "assignment_name": assignment.name,
      "environment": "production" if use_prod else "development"
    }

  except Exception as e:
    raise HTTPException(status_code=400,
                        detail=f"Failed to fetch Canvas data: {str(e)}")


@router.get("/{session_id}/problem-max-points-all")
async def get_all_problem_max_points(
  session_id: int,
  current_user: dict = Depends(require_session_access())
):
  """Get max points for all problems in a session (requires session access)"""
  session_repo = SessionRepository()
  if not session_repo.exists(session_id):
    raise HTTPException(status_code=404, detail="Session not found")

  metadata_repo = ProblemMetadataRepository()
  max_points = metadata_repo.get_all_max_points(session_id)

  return {"max_points": max_points}


@router.put("/{session_id}/problem-max-points")
async def update_problem_max_points(
  session_id: int,
  problem_number: int,
  max_points: float,
  current_user: dict = Depends(require_session_access())
):
  """Update max points for a specific problem number in a session (requires session access)"""
  session_repo = SessionRepository()
  if not session_repo.exists(session_id):
    raise HTTPException(status_code=404, detail="Session not found")

  # Update metadata
  metadata_repo = ProblemMetadataRepository()
  metadata_repo.upsert_max_points(session_id, problem_number, max_points)

  # Update all existing problems with this number
  problem_repo = ProblemRepository()
  problems_updated = problem_repo.update_max_points_bulk(
    session_id, problem_number, max_points)

  return {
    "status": "updated",
    "session_id": session_id,
    "problem_number": problem_number,
    "max_points": max_points,
    "problems_updated": problems_updated
  }


@router.get("/{session_id}/default-feedback/{problem_number}")
async def get_default_feedback(
  session_id: int,
  problem_number: int,
  current_user: dict = Depends(require_session_access())
):
  """Get default feedback for a specific problem number (requires session access)"""
  metadata_repo = ProblemMetadataRepository()
  result = metadata_repo.get_default_feedback(session_id, problem_number)
  if result:
    feedback, threshold = result
  else:
    feedback, threshold = None, None

  return {
    "default_feedback": feedback,
    "default_feedback_threshold": threshold if threshold is not None else 100.0
  }


@router.get("/{session_id}/ai-grading-notes/{problem_number}")
async def get_ai_grading_notes(
  session_id: int,
  problem_number: int,
  current_user: dict = Depends(require_session_access())
):
  """Get AI grading notes for a problem (requires session access)"""
  metadata_repo = ProblemMetadataRepository()
  notes = metadata_repo.get_ai_grading_notes(session_id, problem_number)
  return {"ai_grading_notes": notes}


@router.put("/{session_id}/ai-grading-notes")
async def update_ai_grading_notes(
  session_id: int,
  problem_number: int,
  ai_grading_notes: str = None,
  current_user: dict = Depends(require_instructor)
):
  """Update AI grading notes (instructor only)"""
  metadata_repo = ProblemMetadataRepository()
  notes = ai_grading_notes or ""
  metadata_repo.upsert_ai_grading_notes(session_id, problem_number, notes)
  return {
    "ai_grading_notes": notes
  }


@router.put("/{session_id}/default-feedback")
async def update_default_feedback(
  session_id: int,
  problem_number: int,
  default_feedback: str = None,
  threshold: float = 100.0,
  current_user: dict = Depends(require_session_access())
):
  """Update default feedback for a specific problem number (requires session access)"""
  session_repo = SessionRepository()
  if not session_repo.exists(session_id):
    raise HTTPException(status_code=404, detail="Session not found")

  metadata_repo = ProblemMetadataRepository()
  metadata_repo.upsert_default_feedback(session_id, problem_number,
                                        default_feedback, threshold)

  return {
    "status": "updated",
    "session_id": session_id,
    "problem_number": problem_number,
    "default_feedback": default_feedback,
    "threshold": threshold
  }


@router.delete("/{session_id}")
async def delete_session(
  session_id: int,
  current_user: dict = Depends(require_instructor)
):
  """Delete a grading session and all associated data (instructor only)"""
  session_repo = SessionRepository()
  deleted_count = session_repo.delete(session_id)

  if deleted_count == 0:
    raise HTTPException(status_code=404, detail="Session not found")

  return {"status": "deleted", "session_id": session_id}


@router.post("/{session_id}/clone", response_model=SessionResponse)
async def clone_session(
  session_id: int,
  request: SessionCloneRequest,
  current_user: dict = Depends(require_instructor)
):
  """Clone a grading session with optional data clearing (instructor only)"""
  from ..repositories import with_transaction
  from ..domain.submission import Submission
  from ..domain.problem import Problem
  from ..domain.common import SessionStatus as DomainSessionStatus

  with with_transaction() as repos:
    source_session = repos.sessions.get_by_id(session_id)
    if not source_session:
      raise HTTPException(status_code=404, detail="Session not found")

    metadata = dict(source_session.metadata) if source_session.metadata else {}
    metadata["session_name"] = request.new_name

    new_session = GradingSession(
      id=0,
      assignment_id=source_session.assignment_id,
      assignment_name=source_session.assignment_name,
      course_id=source_session.course_id,
      course_name=source_session.course_name,
      status=DomainSessionStatus.READY,
      created_at=datetime.now(),
      updated_at=datetime.now(),
      canvas_points=source_session.canvas_points,
      metadata=metadata,
      total_exams=0,
      processed_exams=0,
      matched_exams=0,
      processing_message="Cloned session",
      use_prod_canvas=source_session.use_prod_canvas
    )

    created_session = repos.sessions.create(new_session)

    submissions_list = repos.submissions.get_by_session(session_id)
    submission_map = {}
    new_submissions = []
    matched_count = 0

    for sub in submissions_list:
      if sub.canvas_user_id is not None:
        matched_count += 1
      new_submissions.append(
        Submission(
          id=0,
          session_id=created_session.id,
          document_id=sub.document_id,
          approximate_name=sub.approximate_name,
          name_image_data=sub.name_image_data,
          student_name=sub.student_name,
          display_name=sub.display_name,
          canvas_user_id=sub.canvas_user_id,
          page_mappings=sub.page_mappings,
          total_score=None if request.clear_scores else sub.total_score,
          graded_at=None if request.clear_scores else sub.graded_at,
          file_hash=sub.file_hash,
          original_filename=sub.original_filename,
          exam_pdf_data=sub.exam_pdf_data
        ))

    created_subs = repos.submissions.bulk_create(new_submissions)
    for old_sub, new_sub in zip(submissions_list, created_subs):
      submission_map[old_sub.id] = new_sub.id

    new_problems = []
    for old_sub in submissions_list:
      problems_list = repos.problems.get_by_submission(old_sub.id)
      for prob in problems_list:
        new_problems.append(
          Problem(
            id=0,
            session_id=created_session.id,
            submission_id=submission_map[old_sub.id],
            problem_number=prob.problem_number,
            score=None if request.clear_scores else prob.score,
            feedback=None if request.clear_scores else prob.feedback,
            graded=False if request.clear_scores else prob.graded,
            graded_at=None if request.clear_scores else prob.graded_at,
            is_blank=prob.is_blank,
            blank_confidence=prob.blank_confidence,
            blank_method=prob.blank_method,
            blank_reasoning=prob.blank_reasoning,
            max_points=prob.max_points,
            ai_reasoning=None if request.clear_scores else prob.ai_reasoning,
            region_coords=prob.region_coords,
            qr_encrypted_data=prob.qr_encrypted_data,
            transcription=prob.transcription,
            transcription_model=prob.transcription_model,
            transcription_cached_at=prob.transcription_cached_at
          ))

    if new_problems:
      repos.problems.bulk_create(new_problems)

    # Copy problem metadata with optional clearing
    with repos.sessions._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute("SELECT * FROM problem_metadata WHERE session_id = ?",
                     (session_id,))
      metadata_rows = cursor.fetchall()
      for row in metadata_rows:
        cursor.execute("""
          INSERT INTO problem_metadata
          (session_id, problem_number, max_points, question_text, grading_rubric,
           default_feedback, default_feedback_threshold, ai_grading_notes)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
          created_session.id,
          row["problem_number"],
          row["max_points"],
          row["question_text"],
          row["grading_rubric"],
          None if request.clear_default_feedback else row["default_feedback"],
          row["default_feedback_threshold"] if row["default_feedback_threshold"] is not None else 100.0,
          None if request.clear_ai_grading_notes else row["ai_grading_notes"]
        ))

      # Copy feedback tags
      cursor.execute("SELECT * FROM feedback_tags WHERE session_id = ?",
                     (session_id,))
      for tag in cursor.fetchall():
        cursor.execute("""
          INSERT INTO feedback_tags
          (session_id, problem_number, short_name, comment_text, use_count, created_at)
          VALUES (?, ?, ?, ?, ?, ?)
        """, (
          created_session.id,
          tag["problem_number"],
          tag["short_name"],
          tag["comment_text"],
          tag["use_count"],
          tag["created_at"]
        ))

    # Update counts
    created_session.total_exams = len(submissions_list)
    created_session.processed_exams = len(submissions_list)
    created_session.matched_exams = matched_count
    repos.sessions.update(created_session)

    return session_to_response(created_session)


@router.get("/{session_id}/export")
async def export_session(
  session_id: int,
  current_user: dict = Depends(require_instructor)
):
  """Export complete session data as JSON for checkpointing (instructor only)"""
  from dataclasses import asdict

  session_repo = SessionRepository()
  submission_repo = SubmissionRepository()
  problem_repo = ProblemRepository()

  # Get session metadata
  session = session_repo.get_by_id(session_id)
  if not session:
    raise HTTPException(status_code=404, detail="Session not found")

  session_data = asdict(session)
  session_data['status'] = session.status.value  # Convert enum to string

  # Get all submissions
  submissions_list = submission_repo.get_by_session(session_id)
  submissions = []
  for sub in submissions_list:
    sub_dict = asdict(sub)
    # Get all problems for this submission
    problems_list = problem_repo.get_by_submission(sub.id)
    sub_dict["problems"] = [asdict(p) for p in problems_list]
    submissions.append(sub_dict)

  # Get problem stats, metadata, and feedback tags using direct SQL
  # (These tables don't have repositories yet and are less critical)
  with get_db_connection() as conn:
    cursor = conn.cursor()

    # Get problem stats
    cursor.execute("SELECT * FROM problem_stats WHERE session_id = ?",
                   (session_id, ))
    problem_stats = [dict(row) for row in cursor.fetchall()]

    # Get problem metadata
    cursor.execute("SELECT * FROM problem_metadata WHERE session_id = ?",
                   (session_id, ))
    problem_metadata = [dict(row) for row in cursor.fetchall()]

    # Get feedback tags
    cursor.execute("SELECT * FROM feedback_tags WHERE session_id = ?",
                   (session_id, ))
    feedback_tags = [dict(row) for row in cursor.fetchall()]

  # Build export structure
  export_data = {
    "export_version": 1,
    "exported_at": datetime.now().isoformat(),
    "session": session_data,
    "submissions": submissions,
    "problem_stats": problem_stats,
    "problem_metadata": problem_metadata,
    "feedback_tags": feedback_tags
  }

  # Create JSON response
  json_str = json.dumps(export_data, indent=2, default=str)

  # Generate filename
  assignment_name = session.assignment_name.replace(" ", "_")
  filename = f"grading_session_{session_id}_{assignment_name}.json"

  # Return as downloadable file
  return StreamingResponse(
    io.BytesIO(json_str.encode()),
    media_type="application/json",
    headers={"Content-Disposition": f"attachment; filename={filename}"})


@router.post("/compare-export")
async def export_session_comparison(
  request: SessionCompareExportRequest,
  current_user: dict = Depends(require_instructor)
):
  """Export per-problem comparison CSV for multiple sessions (instructor only)"""
  import csv
  from collections import defaultdict

  session_ids = [sid for sid in request.session_ids if isinstance(sid, int)]
  if not session_ids or len(session_ids) < 2:
    raise HTTPException(status_code=400,
                        detail="At least two session_ids are required")

  repo = SessionRepository()
  sessions = []
  missing = []
  for session_id in session_ids:
    session = repo.get_by_id(session_id)
    if session is None:
      missing.append(session_id)
    else:
      sessions.append(session)
  if missing:
    raise HTTPException(status_code=404,
                        detail=f"Session(s) not found: {missing}")

  session_names = {}
  for session in sessions:
    metadata_name = None
    if session.metadata:
      metadata_name = session.metadata.get("session_name")
    session_names[session.id] = metadata_name or session.assignment_name or f"Session {session.id}"

  with get_db_connection() as conn:
    cursor = conn.cursor()

    data = defaultdict(lambda: defaultdict(dict))

    for session_id in session_ids:
      cursor.execute("""
        SELECT s.file_hash,
               p.problem_number,
               p.score,
               p.feedback,
               p.graded,
               p.is_blank,
               p.submission_id
        FROM problems p
        JOIN submissions s ON p.submission_id = s.id
        WHERE p.session_id = ? AND s.file_hash IS NOT NULL
      """, (session_id,))

      for row in cursor.fetchall():
        file_hash = row["file_hash"]
        problem_number = row["problem_number"]
        data[file_hash][problem_number][session_id] = {
          "submission_id": row["submission_id"],
          "score": row["score"],
          "feedback": row["feedback"],
          "graded": row["graded"],
          "is_blank": row["is_blank"],
        }

  # Build CSV
  output = io.StringIO()
  writer = csv.writer(output)

  header_top = ["file_hash", "problem_number"]
  header_bottom = ["", ""]
  for session_id in session_ids:
    label = f"{session_id} : {session_names.get(session_id, f'Session {session_id}')}"
    header_top.extend([label] * 5)
    header_bottom.extend([
      "submission_id",
      "score",
      "feedback",
      "graded",
      "is_blank",
    ])
  writer.writerow(header_top)
  writer.writerow(header_bottom)

  for file_hash in sorted(data.keys()):
    for problem_number in sorted(data[file_hash].keys()):
      row = [file_hash, problem_number]
      for session_id in session_ids:
        entry = data[file_hash][problem_number].get(session_id, {})
        row.extend([
          entry.get("submission_id"),
          entry.get("score"),
          entry.get("feedback"),
          entry.get("graded"),
          entry.get("is_blank"),
        ])
      writer.writerow(row)

  filename = f"session_comparison_{'_'.join(str(s) for s in session_ids)}.csv"
  return StreamingResponse(
    io.BytesIO(output.getvalue().encode()),
    media_type="text/csv",
    headers={"Content-Disposition": f"attachment; filename={filename}"})


@router.post("/import")
async def import_session(
  file: UploadFile = File(...),
  current_user: dict = Depends(require_instructor)
):
  """Import session data from JSON checkpoint file (instructor only)"""
  import logging
  from ..repositories import with_transaction
  from ..domain.submission import Submission
  from ..domain.problem import Problem

  log = logging.getLogger(__name__)

  try:
    # Read file content
    content = await file.read()

    # Parse JSON
    import_data = json.loads(content.decode())

    # Validate structure
    if import_data.get("export_version") != 1:
      raise HTTPException(status_code=400, detail="Unsupported export version")

    session_data = import_data["session"]
    submissions = import_data["submissions"]
    problem_stats = import_data.get("problem_stats", [])
    problem_metadata = import_data.get("problem_metadata", [])
    feedback_tags = import_data.get("feedback_tags", [])

    # Use transaction for atomic import
    with with_transaction() as repos:
      # Create new session
      new_session = GradingSession(
        id=0,  # Will be set by DB
        assignment_id=session_data["assignment_id"],
        assignment_name=session_data["assignment_name"],
        course_id=session_data["course_id"],
        course_name=session_data.get("course_name"),
        status=DomainSessionStatus(session_data["status"]),
        canvas_points=session_data.get("canvas_points"),
        use_prod_canvas=session_data.get("use_prod_canvas", False),
        created_at=datetime.fromisoformat(session_data.get("created_at")) if session_data.get("created_at") else datetime.now(),
        updated_at=datetime.now(),  # Use current time for updated_at
        total_exams=session_data.get("total_exams", 0),
        processed_exams=session_data.get("processed_exams", 0),
        matched_exams=session_data.get("matched_exams", 0),
        processing_message=session_data.get("processing_message"),
        metadata=session_data.get("metadata")
      )
      created_session = repos.sessions.create(new_session)
      new_session_id = created_session.id
      log.info(f"Created new session {new_session_id} from import")

      # Import submissions and problems
      for submission_data in submissions:
        # Create submission domain object
        new_submission = Submission(
          id=0,  # Will be set by DB
          session_id=new_session_id,
          document_id=submission_data["document_id"],
          approximate_name=submission_data.get("approximate_name"),
          name_image_data=submission_data.get("name_image_data"),
          student_name=submission_data.get("student_name"),
          display_name=submission_data.get("display_name"),
          canvas_user_id=submission_data.get("canvas_user_id"),
          page_mappings=submission_data["page_mappings"],
          total_score=submission_data.get("total_score"),
          graded_at=datetime.fromisoformat(submission_data.get("graded_at")) if submission_data.get("graded_at") else None,
          file_hash=submission_data.get("file_hash"),
          original_filename=submission_data.get("original_filename"),
          exam_pdf_data=submission_data.get("exam_pdf_data")
        )
        created_submission = repos.submissions.create(new_submission)

        # Import problems for this submission
        problems_to_create = []
        for problem_data in submission_data.get("problems", []):
          new_problem = Problem(
            id=0,  # Will be set by DB
            session_id=new_session_id,
            submission_id=created_submission.id,
            problem_number=problem_data["problem_number"],
            score=problem_data.get("score"),
            feedback=problem_data.get("feedback"),
            graded=bool(problem_data.get("graded", 0)),
            graded_at=datetime.fromisoformat(problem_data.get("graded_at")) if problem_data.get("graded_at") else None,
            is_blank=bool(problem_data.get("is_blank", 0)),
            blank_confidence=problem_data.get("blank_confidence", 0.0),
            blank_method=problem_data.get("blank_method"),
            blank_reasoning=problem_data.get("blank_reasoning"),
            max_points=problem_data.get("max_points"),
            ai_reasoning=problem_data.get("ai_reasoning"),
            region_coords=problem_data.get("region_coords"),
            qr_encrypted_data=problem_data.get("qr_encrypted_data"),
            transcription=problem_data.get("transcription"),
            transcription_model=problem_data.get("transcription_model"),
            transcription_cached_at=datetime.fromisoformat(problem_data.get("transcription_cached_at")) if problem_data.get("transcription_cached_at") else None
          )
          problems_to_create.append(new_problem)

        # Bulk create all problems for this submission
        if problems_to_create:
          repos.problems.bulk_create(problems_to_create)

      # Import problem stats, metadata, and feedback tags using direct SQL
      # (These tables don't have full repositories yet, okay for import)
      conn = repos.sessions._get_connection().__enter__()  # Get underlying connection
      cursor = conn.cursor()

      # Import problem stats
      for stat in problem_stats:
        cursor.execute(
          """
          INSERT INTO problem_stats
          (session_id, problem_number, avg_score, num_graded, num_total, updated_at)
          VALUES (?, ?, ?, ?, ?, ?)
          """,
          (new_session_id, stat["problem_number"], stat.get("avg_score"),
           stat.get("num_graded", 0), stat.get("num_total", 0), datetime.now()))

      # Import problem metadata (max_points, default_feedback, etc.)
      for metadata in problem_metadata:
        cursor.execute(
          """
          INSERT INTO problem_metadata
          (session_id, problem_number, max_points, default_feedback, default_feedback_threshold, ai_grading_notes)
          VALUES (?, ?, ?, ?, ?, ?)
          """,
          (new_session_id, metadata["problem_number"],
           metadata.get("max_points"), metadata.get("default_feedback"),
           metadata.get("default_feedback_threshold", 100.0),
           metadata.get("ai_grading_notes")))

      # Import feedback tags
      for tag in feedback_tags:
        cursor.execute(
          """
          INSERT INTO feedback_tags
          (session_id, problem_number, short_name, comment_text, use_count, created_at)
          VALUES (?, ?, ?, ?, ?, ?)
          """,
          (new_session_id, tag["problem_number"], tag["short_name"],
           tag["comment_text"], tag.get("use_count", 0),
           tag.get("created_at", datetime.now())))

      log.info(
        f"Imported {len(submissions)} submissions, {sum(len(s.get('problems', [])) for s in submissions)} problems, {len(problem_metadata)} metadata entries, and {len(feedback_tags)} feedback tags"
      )

    return {
      "status": "imported",
      "session_id": new_session_id,
      "assignment_name": session_data["assignment_name"],
      "submissions_imported": len(submissions)
    }

  except json.JSONDecodeError as e:
    raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
  except Exception as e:
    log.error(f"Import failed: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


@router.post("/encryption-key/test")
async def test_encryption_key(
  encrypted_data: str,
  encryption_key: str,
  current_user: dict = Depends(require_instructor)
):
  """Test if an encryption key can decrypt sample QR code data (instructor only)"""
  from ..services.qr_scanner import MinimalQuestionQRCode
  import logging
  log = logging.getLogger(__name__)

  try:
    # Try to decrypt with the provided key
    metadata = MinimalQuestionQRCode.decrypt_question_data(
      encrypted_data, encryption_key.encode())

    return {
      "status": "success",
      "message": "Encryption key is valid",
      "metadata": metadata
    }
  except Exception as e:
    log.warning(f"Failed to decrypt with provided key: {e}")
    return {
      "status": "failed",
      "message": f"Encryption key failed to decrypt: {str(e)}"
    }


@router.post("/encryption-key/set")
async def set_encryption_key(
  encryption_key: str,
  current_user: dict = Depends(require_instructor)
):
  """
    Set the encryption key for the current session (instructor only, runtime only, not persisted).
    This is a workaround for when the QUIZ_ENCRYPTION_KEY env var isn't available.
    """
  import logging
  log = logging.getLogger(__name__)

  # Set the environment variable for this process
  os.environ['QUIZ_ENCRYPTION_KEY'] = encryption_key

  log.info("Encryption key updated for current session (runtime only)")

  return {
    "status":
    "success",
    "message":
    "Encryption key set for current session. This will be lost when the server restarts."
  }


@router.post("/{session_id}/rescan-qr")
async def rescan_qr_codes(
  session_id: int,
  current_user: dict = Depends(require_session_access())
):
  """
    Re-scan QR codes for all problems in a session using progressive DPI (requires session access).
    This is useful when the initial scan fails to detect QR codes.

    Uses progressive DPI escalation (150, 300, 600, 900) - tries low DPI first
    for speed, then increases only if needed for complex QR codes.
    This matches the logic used during initial exam upload.

    Args:
        session_id: The session ID to re-scan

    Returns:
        Statistics about QR codes found and updated
    """
  log = logging.getLogger(__name__)
  log.info(f"Re-scanning QR codes for session {session_id} with progressive DPI")

  # Initialize QR scanner
  qr_scanner = QRScanner()
  if not qr_scanner.available:
    raise HTTPException(
      status_code=400,
      detail="QR scanner not available (opencv-python or pyzbar not installed)"
    )

  session_repo = SessionRepository()
  submission_repo = SubmissionRepository()
  problem_repo = ProblemRepository()
  metadata_repo = ProblemMetadataRepository()

  # Verify session exists
  if not session_repo.exists(session_id):
    raise HTTPException(status_code=404, detail="Session not found")

  # Get all submissions with their PDF data
  submissions = submission_repo.get_by_session(session_id)
  submissions_with_pdf = [s for s in submissions if s.exam_pdf_data]

  if not submissions_with_pdf:
    raise HTTPException(
      status_code=400,
      detail="No submissions with PDF data found in this session")

  total_submissions = len(submissions_with_pdf)
  total_problems_scanned = 0
  total_qr_codes_found = 0
  problems_updated = 0
  dpi_stats = {150: 0, 300: 0, 600: 0, 900: 0}  # Track which DPI found QR codes

  for submission in submissions_with_pdf:
    pdf_base64 = submission.exam_pdf_data

    # Decode PDF
    pdf_bytes = base64.b64decode(pdf_base64)
    pdf_document = fitz.open("pdf", pdf_bytes)

    # Get all problems for this submission
    problems = problem_repo.get_by_submission(submission.id)

    for problem in problems:
      if not problem.region_coords:
        log.warning(
          f"Problem {problem.id} (number {problem.problem_number}) has no region coordinates, skipping"
        )
        continue

      # Parse region coordinates
      region_coords = problem.region_coords
      start_page = region_coords["page_number"]
      start_y = region_coords["region_y_start"]
      end_page = region_coords.get("end_page_number", start_page)
      end_y = region_coords["region_y_end"]

      total_problems_scanned += 1

      # Use progressive DPI: start low (fast), increase only if needed
      # This matches the logic in exam_processor.py
      from ..services.problem_service import ProblemService
      problem_service = ProblemService()

      qr_data = None
      for dpi in [150, 300, 600, 900]:
        problem_image_base64, _ = problem_service.extract_image_from_document(
          pdf_document,
          start_page,
          start_y,
          end_page,
          end_y,
          page_transforms=region_coords.get("page_transforms"),
          dpi=dpi)

        # Try scanning at this resolution
        qr_data = qr_scanner.scan_qr_from_image(problem_image_base64)
        if qr_data:
          if dpi > 150:
            log.info(
              f"Problem {problem.problem_number} (ID {problem.id}): Found QR code at {dpi} DPI (after trying lower resolutions)"
            )
          else:
            log.info(
              f"Problem {problem.problem_number} (ID {problem.id}): Found QR code at {dpi} DPI"
            )
          dpi_stats[dpi] += 1
          break  # Found it, no need to try higher DPI

      if qr_data:
        total_qr_codes_found += 1

        # Update problem with QR data
        problem_repo.update_qr_data(problem.id, qr_data["max_points"],
                                     qr_data.get("encrypted_data"))

        # Also update problem_metadata for this session
        metadata_repo.upsert_max_points(session_id, problem.problem_number,
                                        qr_data["max_points"])

        problems_updated += 1
      else:
        log.debug(
          f"Problem {problem.problem_number} (ID {problem.id}): No QR code found at any DPI")

    pdf_document.close()

  log.info(
    f"QR re-scan complete: {total_qr_codes_found} codes found in {total_problems_scanned} problems across {total_submissions} submissions"
  )
  log.info(f"DPI breakdown: 150={dpi_stats[150]}, 300={dpi_stats[300]}, 600={dpi_stats[600]}, 900={dpi_stats[900]}")

  return {
      "status":
      "success",
      "total_submissions":
      total_submissions,
      "total_problems_scanned":
      total_problems_scanned,
      "qr_codes_found":
      total_qr_codes_found,
      "problems_updated":
      problems_updated,
      "dpi_stats":
      dpi_stats,
      "message":
      f"Re-scanned {total_problems_scanned} problems with progressive DPI. Found {total_qr_codes_found} QR codes and updated {problems_updated} problems."
    }
