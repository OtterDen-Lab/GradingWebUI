"""
Finalization endpoints for completing grading and uploading to Canvas.
"""
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Body
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from pathlib import Path
import tempfile
import shutil
import logging
import asyncio
from pydantic import BaseModel

from ..database import get_db_connection
from ..repositories import SessionRepository, SubmissionRepository, ProblemRepository
from ..domain.common import SessionStatus
from ..services.finalizer import FinalizationService
from .. import sse
from ..auth import require_instructor, require_session_access
from .. import workflow_locks

router = APIRouter()
log = logging.getLogger(__name__)


class FinalizeOptions(BaseModel):
  keep_previous_best: bool = True
  clobber_feedback: bool = False
  submission_ids: list[int] | None = None


@router.get("/{session_id}/finalize-stream")
async def finalize_progress_stream(
  session_id: int,
  current_user: dict = Depends(require_instructor)
):
  """SSE stream for finalization progress (instructor only)"""
  stream_id = sse.make_stream_id("finalize", session_id)

  # Create stream if it doesn't exist
  if not sse.get_stream(stream_id):
    sse.create_stream(stream_id)

  return StreamingResponse(sse.event_generator(stream_id),
                           media_type="text/event-stream",
                           headers={
                             "Cache-Control": "no-cache",
                             "Connection": "keep-alive",
                           })


@router.post("/{session_id}/finalize")
async def finalize_session(
  session_id: int,
  background_tasks: BackgroundTasks,
  options: Optional[FinalizeOptions] = Body(default=None),
  current_user: dict = Depends(require_instructor)
):
  """Start finalization process for a session (instructor only)"""
  session_repo = SessionRepository()
  problem_repo = ProblemRepository()

  # Verify session exists
  session = session_repo.get_by_id(session_id)
  if not session:
    raise HTTPException(status_code=404, detail="Session not found")
  if session.metadata and session.metadata.get("mock_roster"):
    raise HTTPException(
      status_code=400,
      detail="Mock roster sessions cannot be finalized to Canvas."
    )

  # Check if all problems are graded
  ungraded_count = problem_repo.count_ungraded(session_id)
  if ungraded_count > 0:
    raise HTTPException(
      status_code=400,
      detail=f"Cannot finalize: {ungraded_count} problems still ungraded")

  if session.status == SessionStatus.FINALIZING:
    raise HTTPException(status_code=409,
                        detail="Finalization is already running for this session")

  acquired, conflict = workflow_locks.acquire_with_conflicts(
    "finalize", session_id, ("autograde", ))
  if not acquired:
    if conflict == "autograde":
      raise HTTPException(
        status_code=409,
        detail="Cannot finalize while autograding is in progress for this session")
    raise HTTPException(status_code=409,
                        detail="Finalization is already running for this session")

  finalization_options = options or FinalizeOptions()
  if finalization_options.submission_ids is not None:
    if len(finalization_options.submission_ids) == 0:
      workflow_locks.release("finalize", session_id)
      raise HTTPException(status_code=400,
                          detail="Select at least one student to upload.")

    submission_repo = SubmissionRepository()
    submissions = {
      submission.id: submission
      for submission in submission_repo.get_by_session_lightweight(session_id)
    }
    invalid_ids = sorted(
      submission_id for submission_id in finalization_options.submission_ids
      if submission_id not in submissions)
    if invalid_ids:
      workflow_locks.release("finalize", session_id)
      raise HTTPException(status_code=400,
                          detail="One or more selected submissions are invalid.")

    unmatched_ids = sorted(
      submission_id for submission_id in finalization_options.submission_ids
      if submissions[submission_id].canvas_user_id is None)
    if unmatched_ids:
      workflow_locks.release("finalize", session_id)
      raise HTTPException(
        status_code=400,
        detail="Selected submissions must be matched to Canvas students.")

  # Update session status with initial progress message
  session_repo.update_status(session_id, SessionStatus.FINALIZING, "Starting finalization...")

  # Create SSE stream for progress updates
  stream_id = sse.make_stream_id("finalize", session_id)
  sse.create_stream(stream_id)

  # Start background finalization
  try:
    background_tasks.add_task(run_finalization, session_id, stream_id,
                              finalization_options)
  except Exception:
    workflow_locks.release("finalize", session_id)
    raise

  return {
    "status": "started",
    "session_id": session_id,
    "message": "Finalization started in background"
  }


@router.get("/{session_id}/finalization-status")
async def get_finalization_status(
  session_id: int,
  current_user: dict = Depends(require_instructor)
):
  """Get status of finalization process (instructor only)"""
  session_repo = SessionRepository()

  session = session_repo.get_by_id(session_id)
  if not session:
    raise HTTPException(status_code=404, detail="Session not found")

  return {
    "status": session.status.value,
    "message": session.processing_message
  }


@router.get("/{session_id}/submissions/{submission_id}/feedback-preview",
            response_class=HTMLResponse)
async def get_finalize_feedback_preview(
  session_id: int,
  submission_id: int,
  current_user: dict = Depends(require_session_access())
):
  """Render the exact feedback HTML that finalization would upload."""
  submission_repo = SubmissionRepository()
  submission = submission_repo.get_by_id(submission_id)
  if not submission or submission.session_id != session_id:
    raise HTTPException(status_code=404,
                        detail="Submission not found in this session")

  finalizer = FinalizationService(session_id,
                                  Path("."),
                                  "preview",
                                  None,
                                  selected_submission_ids=[submission_id])
  session_info = finalizer._get_session_info()
  submissions = finalizer._get_submissions()
  if not submissions:
    raise HTTPException(status_code=404,
                        detail="Submission not found in this session")

  preview_submission = submissions[0]
  preview_submission["assignment_name"] = session_info.get("assignment_name")
  preview_submission["session_name"] = session_info.get("session_name")

  return HTMLResponse(content=finalizer._generate_comments(preview_submission))


async def run_finalization(session_id: int, stream_id: str,
                           options: Optional[FinalizeOptions] = None):
  """Background task to finalize grading and upload to Canvas"""
  try:
    log.info(f"Starting finalization for session {session_id}")

    # Send start event
    await sse.send_event(stream_id, "start",
                         {"message": "Starting finalization..."})

    # Create temp directory for PDF processing
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir)

      # Get event loop reference to pass to finalizer (for thread communication)
      loop = asyncio.get_event_loop()

      # Initialize finalizer with event loop reference
      finalization_options = options or FinalizeOptions()
      finalizer = FinalizationService(
        session_id,
        temp_path,
        stream_id,
        loop,
        keep_previous_best=finalization_options.keep_previous_best,
        clobber_feedback=finalization_options.clobber_feedback,
        selected_submission_ids=finalization_options.submission_ids,
      )

      # Run finalization in thread executor so event loop can send SSE events
      await loop.run_in_executor(None, finalizer.finalize)

    # Update session to finalized
    session_repo = SessionRepository()
    session_repo.update_status(session_id, SessionStatus.FINALIZED, "Finalized and uploaded to Canvas")

    log.info(f"Finalization complete for session {session_id}")

    # Send completion event
    await sse.send_event(
      stream_id, "complete",
      {"message": "Finalization complete - all grades uploaded to Canvas"})

  except Exception as e:
    log.error(f"Finalization failed for session {session_id}: {e}",
              exc_info=True)

    # Send error event
    await sse.send_event(stream_id, "error", {
      "error": str(e),
      "message": f"Finalization failed: {str(e)}"
    })

    # Update session to error state
    session_repo = SessionRepository()
    session_repo.update_status(session_id, SessionStatus.ERROR, f"Finalization failed: {str(e)}")
  finally:
    workflow_locks.release("finalize", session_id)
