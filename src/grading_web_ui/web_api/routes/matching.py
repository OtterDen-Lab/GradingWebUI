"""
Name matching endpoints for unmatched submissions.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List
import json
from pathlib import Path
import base64
import fitz  # PyMuPDF

from ..models import NameMatchRequest
from ..database import get_db_connection
from ..repositories import SessionRepository, SubmissionRepository
from lms_interface.canvas_interface import CanvasInterface
from ..auth import require_session_access

router = APIRouter()


def _resolve_submission_pdf_path(session_data: dict,
                                 file_hash: str | None,
                                 original_filename: str | None,
                                 document_id: int | None = None) -> Path | None:
  """Resolve submission PDF path from session metadata."""
  file_metadata = session_data.get("file_metadata") if session_data else None

  if isinstance(file_metadata, dict):
    if file_hash:
      for path_str, metadata in file_metadata.items():
        if isinstance(metadata, dict) and metadata.get("hash") == file_hash:
          return Path(path_str)

    if original_filename:
      for path_str, metadata in file_metadata.items():
        if isinstance(metadata, dict) and metadata.get("original_filename") == original_filename:
          return Path(path_str)

  file_paths = session_data.get("file_paths") if session_data else None
  if isinstance(file_paths, list) and document_id is not None:
    if 0 <= document_id < len(file_paths):
      return Path(file_paths[document_id])

  return None


@router.get("/{session_id}/submissions")
async def get_all_submissions(
  session_id: int,
  current_user: dict = Depends(require_session_access())
):
  """Get all submissions for a session (requires session access)"""
  submission_repo = SubmissionRepository()

  # Get all submissions, ordered by match status
  all_submissions = submission_repo.get_by_session(session_id)

  # Sort: unmatched first, then by document_id
  all_submissions.sort(key=lambda s: (s.is_matched(), s.document_id))

  submissions = []
  for sub in all_submissions:
    submissions.append({
      "id": sub.id,
      "document_id": sub.document_id,
      "approximate_name": sub.approximate_name or "(no name detected)",
      "name_image_data": sub.name_image_data,
      "student_name": sub.student_name,
      "canvas_user_id": sub.canvas_user_id,
      "is_matched": sub.is_matched()
    })

  return {"submissions": submissions}


@router.get("/{session_id}/submissions/{submission_id}/page-preview")
async def get_submission_page_preview(
  session_id: int,
  submission_id: int,
  current_user: dict = Depends(require_session_access())
):
  """Get full first page preview for a submission."""
  submission_repo = SubmissionRepository()
  session_repo = SessionRepository()

  submission = submission_repo.get_by_id(submission_id)
  if not submission or submission.session_id != session_id:
    raise HTTPException(status_code=404, detail="Submission not found")

  page_image_base64 = None
  page_number = 0

  if submission.exam_pdf_data:
    try:
      pdf_bytes = base64.b64decode(submission.exam_pdf_data)
      with fitz.open("pdf", pdf_bytes) as pdf_document:
        if pdf_document.page_count == 0:
          raise HTTPException(status_code=400, detail="Submission PDF has no pages")
        pix = pdf_document[page_number].get_pixmap(dpi=150)
        page_image_base64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
    except HTTPException:
      raise
    except Exception as exc:
      raise HTTPException(
        status_code=500,
        detail=f"Failed to render stored submission PDF: {str(exc)}"
      ) from exc

  if not page_image_base64:
    session_data = session_repo.get_metadata(session_id) or {}
    pdf_path = _resolve_submission_pdf_path(
      session_data=session_data,
      file_hash=submission.file_hash,
      original_filename=submission.original_filename,
      document_id=submission.document_id
    )

    if not pdf_path:
      raise HTTPException(
        status_code=404,
        detail="Original PDF path not found for submission"
      )
    if not pdf_path.exists():
      raise HTTPException(
        status_code=404,
        detail=f"Original PDF file not found: {pdf_path}"
      )

    try:
      with fitz.open(str(pdf_path)) as pdf_document:
        if pdf_document.page_count == 0:
          raise HTTPException(status_code=400, detail="Submission PDF has no pages")
        pix = pdf_document[page_number].get_pixmap(dpi=150)
        page_image_base64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
    except HTTPException:
      raise
    except Exception as exc:
      raise HTTPException(
        status_code=500,
        detail=f"Failed to render submission PDF: {str(exc)}"
      ) from exc

  return {
    "submission_id": submission_id,
    "page_number": page_number,
    "page_image": page_image_base64
  }


@router.get("/{session_id}/students")
async def get_all_students(
  session_id: int,
  reveal_names: bool = True,
  current_user: dict = Depends(require_session_access())
):
  """Get all Canvas students with match status (requires session access)."""
  session_repo = SessionRepository()
  submission_repo = SubmissionRepository()

  # Get session info
  session = session_repo.get_by_id(session_id)
  if not session:
    raise HTTPException(status_code=404, detail="Session not found")

  # Get Canvas students (optionally reveal real names for explicit matching flow).
  privacy_mode = "none" if reveal_names else "id_only"
  canvas_interface = CanvasInterface(
    prod=session.use_prod_canvas,
    privacy_mode=privacy_mode
  )
  course = canvas_interface.get_course(session.course_id)
  assignment = course.get_assignment(session.assignment_id)
  all_students = assignment.get_students(include_names=True)

  # Get already matched user IDs
  matched_ids = submission_repo.get_existing_canvas_users(session_id)

  # Create list with all students, marked as matched or not
  students = [{
    "user_id": s.user_id,
    "name": s.name,
    "is_matched": s.user_id in matched_ids
  } for s in all_students]

  # Sort: unmatched first, then alphabetically within each group
  students.sort(key=lambda s: (s["is_matched"], s["name"]))

  return {"students": students}


@router.post("/{session_id}/match")
async def match_submission(
  session_id: int,
  match: NameMatchRequest,
  reveal_names: bool = True,
  current_user: dict = Depends(require_session_access())
):
  """Manually match a submission to a Canvas student (requires session access)"""
  session_repo = SessionRepository()
  submission_repo = SubmissionRepository()

  # Verify the submission exists and belongs to this session
  submission = submission_repo.get_by_id(match.submission_id)
  if not submission or submission.session_id != session_id:
    raise HTTPException(status_code=404, detail="Submission not found")

  # Get session info for Canvas access
  session = session_repo.get_by_id(session_id)
  if not session:
    raise HTTPException(status_code=404, detail="Session not found")

  # Get student name from Canvas. When reveal_names=true, store the real name.
  privacy_mode = "none" if reveal_names else "id_only"
  canvas_interface = CanvasInterface(
    prod=session.use_prod_canvas,
    privacy_mode=privacy_mode
  )
  course = canvas_interface.get_course(session.course_id)
  assignment = course.get_assignment(session.assignment_id)
  students = assignment.get_students(include_names=True)

  student = next((s for s in students if s.user_id == match.canvas_user_id), None)
  if not student:
    raise HTTPException(status_code=404, detail="Student not found in Canvas")

  # Check if this student is already matched to another submission
  previous_submission = submission_repo.get_by_canvas_user(session_id, match.canvas_user_id)
  previous_submission_id = None

  if previous_submission and previous_submission.id != match.submission_id:
    # If student was previously matched to a different submission, unassign them
    previous_submission_id = previous_submission.id
    submission_repo.clear_match(previous_submission_id)

  # Update submission with new match
  submission_repo.update_match(match.submission_id, match.canvas_user_id, student.name)

  # Check if all submissions are now matched
  unmatched_count = submission_repo.count_unmatched(session_id)

  # DON'T auto-update status to 'ready' - wait for user to click "Confirm All Matches"
  # This allows the user to review all matches before proceeding

  return {
    "status": "matched",
    "student_name": student.name,
    "remaining_unmatched": unmatched_count,
    "reassigned_from": previous_submission_id
  }
