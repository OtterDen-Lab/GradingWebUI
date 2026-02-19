"""
AI grading endpoints for AI-assisted grading.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import logging
import asyncio

from ..repositories import SessionRepository, ProblemRepository, SubmissionRepository, ProblemMetadataRepository
from ..database import update_problem_stats
from ..services.ai_grader import AIGraderService
from .. import sse
from .. import workflow_locks
from ..auth import require_instructor

router = APIRouter()
log = logging.getLogger(__name__)


class ExtractQuestionRequest(BaseModel):
  """Request to extract question text from a problem"""
  problem_number: int


class ExtractQuestionResponse(BaseModel):
  """Response with extracted question text"""
  problem_number: int
  question_text: str
  message: str


class ImageAutogradeSettings(BaseModel):
  """Settings for image-only autograding."""
  batch_size: str
  image_quality: str
  include_answer: bool = False
  include_default_feedback: bool = False
  auto_accept: bool = False
  dry_run: bool = False


class AutogradeRequest(BaseModel):
  """Request to autograde a problem"""
  mode: str = "text-rubric"
  problem_number: int
  question_text: Optional[str] = None  # User-verified question text
  max_points: Optional[float] = None  # Maximum points for this problem
  settings: Optional[ImageAutogradeSettings] = None
  auto_accept: bool = False


class AutogradeAllRequest(BaseModel):
  """Request to autograde all problems in a session"""
  mode: str = "image-only"
  settings: Optional[ImageAutogradeSettings] = None


class AutogradeResponse(BaseModel):
  """Response when autograding starts"""
  status: str
  problem_number: int
  message: str


class GenerateRubricRequest(BaseModel):
  """Request to generate a grading rubric"""
  problem_number: int
  question_text: str
  max_points: float
  num_examples: int = 3  # Number of manually graded examples to include


class GenerateRubricResponse(BaseModel):
  """Response with generated rubric"""
  problem_number: int
  rubric: str
  message: str


class SaveRubricRequest(BaseModel):
  """Request to save/update a rubric"""
  problem_number: int
  rubric: str


def acquire_autograde_lock(session_id: int) -> None:
  """Prevent concurrent autograde/finalize runs for the same session."""
  acquired, conflict = workflow_locks.acquire_with_conflicts(
    "autograde", session_id, ("finalize", ))
  if acquired:
    return
  if conflict == "finalize":
    raise HTTPException(
      status_code=409,
      detail="Cannot start autograding while finalization is in progress")
  raise HTTPException(status_code=409,
                      detail="Autograding is already in progress for this session")


@router.get("/{session_id}/autograde-stream")
async def autograde_progress_stream(
  session_id: int,
  current_user: dict = Depends(require_instructor)
):
  """SSE stream for autograding progress (instructor only)"""
  stream_id = sse.make_stream_id("autograde", session_id)

  # Create stream if it doesn't exist
  if not sse.get_stream(stream_id):
    sse.create_stream(stream_id)

  return StreamingResponse(sse.event_generator(stream_id),
                           media_type="text/event-stream",
                           headers={
                             "Cache-Control": "no-cache",
                             "Connection": "keep-alive",
                           })


@router.post("/{session_id}/extract-question",
             response_model=ExtractQuestionResponse)
async def extract_question(
  session_id: int,
  request: ExtractQuestionRequest,
  current_user: dict = Depends(require_instructor)
):
  """Extract question text from a problem image (instructor only)"""

  session_repo = SessionRepository()
  problem_repo = ProblemRepository()
  submission_repo = SubmissionRepository()

  # Verify session exists
  if not session_repo.exists(session_id):
    raise HTTPException(status_code=404, detail="Session not found")

  # Get a sample problem for this problem number
  problem = problem_repo.get_sample_for_problem_number(
    session_id, request.problem_number)

  if not problem:
    raise HTTPException(
      status_code=404,
      detail=f"No problems found for problem number {request.problem_number}"
    )

  # Get image data - either directly or extract from PDF
  image_data = None
  if getattr(problem, "image_data", None):
    # Legacy: image_data is stored
    image_data = problem.image_data
  elif problem.region_coords:
    # New: extract from PDF using region_coords
    import json

    region_data = problem.region_coords
    if isinstance(region_data, str):
      region_data = json.loads(region_data)

    submission = submission_repo.get_by_id(problem.submission_id)

    if submission and submission.exam_pdf_data:
      image_data = AIGraderService().problem_service.extract_image_from_pdf_data(
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
    raise HTTPException(status_code=500,
                        detail="Problem image data not available")

  try:
    # Extract question text
    ai_grader = AIGraderService()
    question_text = ai_grader.get_or_extract_question(session_id,
                                                      request.problem_number,
                                                      image_data)

    return ExtractQuestionResponse(
      problem_number=request.problem_number,
      question_text=question_text,
      message="Question text extracted successfully")

  except Exception as e:
    log.error(f"Failed to extract question: {e}", exc_info=True)
    raise HTTPException(status_code=500,
                        detail=f"Failed to extract question: {str(e)}")


@router.post("/{session_id}/autograde", response_model=AutogradeResponse)
async def start_autograde(
  session_id: int,
  request: AutogradeRequest,
  background_tasks: BackgroundTasks,
  current_user: dict = Depends(require_instructor)
):
  """Start autograding process for a problem (instructor only)"""

  mode_handlers = {
    "text-rubric": _start_autograde_text,
    "image-only": _start_autograde_image,
  }
  handler = mode_handlers.get(request.mode)
  if not handler:
    raise HTTPException(status_code=400,
                        detail=f"Unsupported autograding mode: {request.mode}")
  return await handler(session_id, request, background_tasks)


@router.post("/{session_id}/autograde-all", response_model=AutogradeResponse)
async def start_autograde_all(
  session_id: int,
  request: AutogradeAllRequest,
  background_tasks: BackgroundTasks,
  current_user: dict = Depends(require_instructor)
):
  """Start autograding for all problems in a session (instructor only)."""
  session_repo = SessionRepository()
  problem_repo = ProblemRepository()

  if not session_repo.exists(session_id):
    raise HTTPException(status_code=404, detail="Session not found")

  if request.mode != "image-only":
    raise HTTPException(status_code=400,
                        detail="Only image-only mode is supported for all-problem runs")

  if not request.settings:
    raise HTTPException(status_code=400,
                        detail="settings are required for image-only mode")

  problem_numbers = problem_repo.get_distinct_problem_numbers(session_id)
  if not problem_numbers:
    raise HTTPException(status_code=400,
                        detail="No problems found for this session")

  totals_by_problem = {
    number: problem_repo.count_ungraded_for_problem_number(session_id, number)
    for number in problem_numbers
  }
  total_ungraded = sum(totals_by_problem.values())
  if total_ungraded == 0:
    raise HTTPException(status_code=400,
                        detail="No ungraded problems found for this session")

  acquire_autograde_lock(session_id)

  stream_id = sse.make_stream_id("autograde", session_id)
  try:
    sse.create_stream(stream_id)
    settings = request.settings.model_dump()
    background_tasks.add_task(run_autograding_all, session_id, problem_numbers,
                              totals_by_problem, settings, stream_id)
  except Exception:
    workflow_locks.release("autograde", session_id)
    raise

  start_message = "Image-only autograding started"
  if settings.get("dry_run"):
    start_message = "Image-only autograding dry run started"

  return AutogradeResponse(
    status="started",
    problem_number=0,
    message=f"{start_message} for {total_ungraded} problems across {len(problem_numbers)} problem numbers")


async def _start_autograde_text(session_id: int, request: AutogradeRequest,
                                background_tasks: BackgroundTasks):
  session_repo = SessionRepository()
  problem_repo = ProblemRepository()
  metadata_repo = ProblemMetadataRepository()

  # Verify session exists
  if not session_repo.exists(session_id):
    raise HTTPException(status_code=404, detail="Session not found")

  if not request.question_text or request.max_points is None:
    raise HTTPException(status_code=400,
                        detail="question_text and max_points are required")

  # Count ungraded problems (include blank submissions for feedback)
  ungraded_count = problem_repo.count_ungraded_for_problem_number(
    session_id, request.problem_number)

  if ungraded_count == 0:
    raise HTTPException(
      status_code=400,
      detail=
      f"No ungraded problems found for problem number {request.problem_number}"
    )

  # Update question_text and max_points in metadata
  metadata_repo.upsert_question_text(session_id, request.problem_number,
                                     request.question_text)
  metadata_repo.upsert_max_points(session_id, request.problem_number,
                                  request.max_points)

  acquire_autograde_lock(session_id)

  # Create SSE stream for progress updates
  stream_id = sse.make_stream_id("autograde", session_id)
  try:
    sse.create_stream(stream_id)

    # Start background autograding
    background_tasks.add_task(run_autograding, session_id,
                              request.problem_number, request.max_points,
                              stream_id, request.auto_accept)
  except Exception:
    workflow_locks.release("autograde", session_id)
    raise

  return AutogradeResponse(
    status="started",
    problem_number=request.problem_number,
    message=f"Autograding started for {ungraded_count} problems")


async def _start_autograde_image(session_id: int, request: AutogradeRequest,
                                 background_tasks: BackgroundTasks):
  session_repo = SessionRepository()
  problem_repo = ProblemRepository()

  # Verify session exists
  if not session_repo.exists(session_id):
    raise HTTPException(status_code=404, detail="Session not found")

  if not request.settings:
    raise HTTPException(status_code=400,
                        detail="settings are required for image-only mode")

  ungraded_count = problem_repo.count_ungraded_for_problem_number(
    session_id, request.problem_number)

  if ungraded_count == 0:
    raise HTTPException(
      status_code=400,
      detail=
      f"No ungraded problems found for problem number {request.problem_number}"
    )

  acquire_autograde_lock(session_id)

  stream_id = sse.make_stream_id("autograde", session_id)
  try:
    sse.create_stream(stream_id)
    settings = request.settings.model_dump()
    background_tasks.add_task(run_autograding_image, session_id,
                              request.problem_number, settings, stream_id)
  except Exception:
    workflow_locks.release("autograde", session_id)
    raise

  start_message = "Image-only autograding started"
  if settings.get("dry_run"):
    start_message = "Image-only autograding dry run started"

  return AutogradeResponse(
    status="started",
    problem_number=request.problem_number,
    message=f"{start_message} for {ungraded_count} problems")


async def run_autograding_image(session_id: int, problem_number: int,
                                settings: dict, stream_id: str):
  """Background task to autograde problems from images with SSE updates."""
  try:
    log.info(
      f"Starting image-only autograding for session {session_id}, problem {problem_number}"
    )

    await sse.send_event(
      stream_id, "start",
      {"message": f"Starting image-only autograding for problem {problem_number}..."})

    loop = asyncio.get_event_loop()
    ai_grader = AIGraderService()

    def update_progress(current, total, message):
      progress_percent = min(100, int(
        (current / total) * 100)) if total > 0 else 0
      try:
        asyncio.run_coroutine_threadsafe(
          sse.send_event(
            stream_id, "progress", {
              "current": current,
              "total": total,
              "progress": progress_percent,
              "message": message
            }), loop)
      except Exception as e:
        log.error(f"Failed to send SSE event: {e}")

    result = await loop.run_in_executor(
      None,
      lambda: ai_grader.autograde_problem_image_only(session_id,
                                                     problem_number,
                                                     settings=settings,
                                                     progress_callback=update_progress))

    log.info(
      f"Image-only autograding complete for session {session_id}, problem {problem_number}: {result}"
    )

    if settings.get("auto_accept") and not settings.get("dry_run"):
      update_problem_stats(session_id)

    await sse.send_event(
      stream_id, "complete", {
        "graded": result["graded"],
        "total": result["total"],
        "message": result["message"]
      })

  except Exception as e:
    log.error(
      f"Image-only autograding failed for session {session_id}, problem {problem_number}: {e}",
      exc_info=True)

    await sse.send_event(stream_id, "error", {
      "error": str(e),
      "message": f"Autograding failed: {str(e)}"
    })
  finally:
    workflow_locks.release("autograde", session_id)


async def run_autograding_all(session_id: int, problem_numbers: list,
                              totals_by_problem: dict, settings: dict,
                              stream_id: str):
  """Background task to autograde all problems with SSE updates."""
  try:
    log.info(
      "Starting image-only autograding for all problems in session %s",
      session_id)

    total_ungraded = sum(totals_by_problem.values())
    await sse.send_event(
      stream_id, "start",
      {"message": f"Starting image-only autograding for {len(problem_numbers)} problems..."})

    loop = asyncio.get_event_loop()
    ai_grader = AIGraderService()

    def send_progress(current, total, message):
      progress_percent = min(100, int(
        (current / total) * 100)) if total > 0 else 0
      try:
        asyncio.run_coroutine_threadsafe(
          sse.send_event(
            stream_id, "progress", {
              "current": current,
              "total": total,
              "progress": progress_percent,
              "message": message
            }), loop)
      except Exception as e:
        log.error(f"Failed to send SSE event: {e}")

    graded_total = 0
    processed_offset = 0
    for problem_number in problem_numbers:
      problem_total = totals_by_problem.get(problem_number, 0)
      if problem_total == 0:
        continue

      def update_progress(current, total, message, problem_number=problem_number, offset=processed_offset):
        global_current = min(total_ungraded, offset + current)
        send_progress(global_current, total_ungraded,
                      f"Problem {problem_number}: {message}")

      result = await loop.run_in_executor(
        None,
        lambda: ai_grader.autograde_problem_image_only(session_id,
                                                       problem_number,
                                                       settings=settings,
                                                       progress_callback=update_progress))
      graded_total += result.get("graded", 0)
      processed_offset += problem_total

    if settings.get("auto_accept") and not settings.get("dry_run"):
      update_problem_stats(session_id)

    await sse.send_event(
      stream_id, "complete", {
        "graded": graded_total,
        "total": total_ungraded,
        "message": f"AI graded {graded_total}/{total_ungraded} problems (all problems)"
      })

  except Exception as e:
    log.error(
      f"Image-only autograding all problems failed for session {session_id}: {e}",
      exc_info=True)
    await sse.send_event(stream_id, "error", {
      "error": str(e),
      "message": f"Autograding failed: {str(e)}"
    })
  finally:
    workflow_locks.release("autograde", session_id)


async def run_autograding(session_id: int, problem_number: int,
                          max_points: float, stream_id: str,
                          auto_accept: bool):
  """Background task to autograde problems with SSE progress updates"""
  try:
    log.info(
      f"Starting autograding for session {session_id}, problem {problem_number}"
    )

    # Send start event
    await sse.send_event(
      stream_id, "start",
      {"message": f"Starting autograding for problem {problem_number}..."})

    # Get event loop reference
    loop = asyncio.get_event_loop()

    # Create AI grader service
    ai_grader = AIGraderService()

    # Progress callback for SSE updates
    def update_progress(current, total, message):
      progress_percent = min(100, int(
        (current / total) * 100)) if total > 0 else 0

      try:
        asyncio.run_coroutine_threadsafe(
          sse.send_event(
            stream_id, "progress", {
              "current": current,
              "total": total,
              "progress": progress_percent,
              "message": message
            }), loop)
      except Exception as e:
        log.error(f"Failed to send SSE event: {e}")

    # Run AI grading in thread executor
    result = await loop.run_in_executor(
      None,
      lambda: ai_grader.autograde_problem(session_id,
                                          problem_number,
                                          max_points=max_points,
                                          progress_callback=update_progress,
                                          auto_accept=auto_accept))

    log.info(
      f"Autograding complete for session {session_id}, problem {problem_number}: {result}"
    )

    if auto_accept:
      update_problem_stats(session_id)

    # Send completion event
    await sse.send_event(
      stream_id, "complete", {
        "graded": result["graded"],
        "total": result["total"],
        "message": result["message"]
      })

  except Exception as e:
    log.error(
      f"Autograding failed for session {session_id}, problem {problem_number}: {e}",
      exc_info=True)

    # Send error event
    await sse.send_event(stream_id, "error", {
      "error": str(e),
      "message": f"Autograding failed: {str(e)}"
    })
  finally:
    workflow_locks.release("autograde", session_id)


@router.post("/{session_id}/generate-rubric",
             response_model=GenerateRubricResponse)
async def generate_rubric(
  session_id: int,
  request: GenerateRubricRequest,
  current_user: dict = Depends(require_instructor)
):
  """Generate a grading rubric using AI and representative examples (instructor only)"""

  session_repo = SessionRepository()

  # Verify session exists
  if not session_repo.exists(session_id):
    raise HTTPException(status_code=404, detail="Session not found")

  try:
    ai_grader = AIGraderService()

    # Get grading examples (manually graded submissions)
    example_answers = ai_grader.get_grading_examples(
      session_id, request.problem_number, limit=request.num_examples)

    if not example_answers:
      raise HTTPException(
        status_code=400,
        detail=
        f"No manually graded examples found for problem {request.problem_number}. "
        "Please manually grade at least a few submissions first.")

    # Generate rubric
    rubric = ai_grader.generate_rubric(request.question_text,
                                       request.max_points,
                                       example_answers=example_answers)

    return GenerateRubricResponse(
      problem_number=request.problem_number,
      rubric=rubric,
      message=f"Generated rubric based on {len(example_answers)} example(s)")

  except HTTPException:
    raise
  except Exception as e:
    log.error(f"Failed to generate rubric: {e}", exc_info=True)
    raise HTTPException(status_code=500,
                        detail=f"Failed to generate rubric: {str(e)}")


@router.post("/{session_id}/save-rubric")
async def save_rubric(
  session_id: int,
  request: SaveRubricRequest,
  current_user: dict = Depends(require_instructor)
):
  """Save or update a grading rubric (instructor only)"""

  session_repo = SessionRepository()
  metadata_repo = ProblemMetadataRepository()

  # Verify session exists
  if not session_repo.exists(session_id):
    raise HTTPException(status_code=404, detail="Session not found")

  # Save rubric to metadata
  metadata_repo.upsert_grading_rubric(session_id, request.problem_number,
                                      request.rubric)

  return {"status": "success", "message": "Rubric saved successfully"}


@router.get("/{session_id}/rubric/{problem_number}")
async def get_rubric(
  session_id: int,
  problem_number: int,
  current_user: dict = Depends(require_instructor)
):
  """Get the current rubric for a problem (instructor only)"""

  session_repo = SessionRepository()
  metadata_repo = ProblemMetadataRepository()

  # Verify session exists
  if not session_repo.exists(session_id):
    raise HTTPException(status_code=404, detail="Session not found")

  # Get rubric from metadata
  rubric = metadata_repo.get_grading_rubric(session_id, problem_number)

  return {"problem_number": problem_number, "rubric": rubric}
