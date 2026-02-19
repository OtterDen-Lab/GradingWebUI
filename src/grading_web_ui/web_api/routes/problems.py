"""
Problem grading endpoints.
"""
import textwrap
import os
import asyncio
import threading
import hashlib

from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
from typing import Optional
import base64
import fitz  # PyMuPDF

from ..models import (ProblemResponse, GradeSubmission, ManualQRCodeSubmission,
                      SubjectiveTriageSubmission)
from ..database import get_db_connection, update_problem_stats
from ..repositories import (ProblemRepository, SubmissionRepository,
                            SessionRepository, ProblemMetadataRepository,
                            SubjectiveTriageRepository)
from ..services.problem_service import ProblemService
from ..services.quiz_regeneration import regenerate_from_encrypted_compat
from ..auth import require_session_access, get_current_user

from grading_web_ui import ai_helper

from PIL import Image
import io

import json
import logging

log = logging.getLogger(__name__)

router = APIRouter()

# Create singleton problem service
_problem_service = ProblemService()
_regeneration_cache = {}
_regeneration_cache_lock = threading.Lock()
_regeneration_cache_max_entries = 2000
_session_prefetch_tasks = {}
_session_prefetch_tasks_lock = threading.Lock()

_DEFAULT_SUBJECTIVE_BUCKETS = [
  {"id": "perfect", "label": "Perfect", "color": "#16a34a"},
  {"id": "excellent", "label": "Excellent", "color": "#22c55e"},
  {"id": "good", "label": "Good", "color": "#3b82f6"},
  {"id": "passable", "label": "Passable", "color": "#f59e0b"},
  {"id": "poor_blank", "label": "Poor/Blank", "color": "#ef4444"},
]


def _cache_key_for_regeneration(problem, quiz_yaml_text: Optional[str]) -> tuple:
  yaml_fingerprint = ""
  if quiz_yaml_text:
    yaml_fingerprint = hashlib.sha1(
      quiz_yaml_text.encode("utf-8")
    ).hexdigest()
  return (
    problem.qr_encrypted_data,
    float(problem.max_points or 0.0),
    yaml_fingerprint
  )


def _get_cached_regeneration(problem_id: int, cache_key: tuple) -> Optional[dict]:
  with _regeneration_cache_lock:
    entry = _regeneration_cache.get(problem_id)
    if not entry:
      return None
    if entry.get("cache_key") != cache_key:
      return None
    return entry.get("response")


def _set_cached_regeneration(problem_id: int, cache_key: tuple,
                             response: dict) -> None:
  with _regeneration_cache_lock:
    _regeneration_cache[problem_id] = {
      "cache_key": cache_key,
      "response": response
    }
    if len(_regeneration_cache) > _regeneration_cache_max_entries:
      oldest_key = next(iter(_regeneration_cache))
      _regeneration_cache.pop(oldest_key, None)


def _clear_cached_regeneration(problem_id: int) -> None:
  with _regeneration_cache_lock:
    _regeneration_cache.pop(problem_id, None)


def _parse_manual_qr_payload(payload_text: str) -> dict:
  if not payload_text or not payload_text.strip():
    raise ValueError("QR payload is empty")

  try:
    payload = json.loads(payload_text.strip())
  except Exception as exc:
    raise ValueError(f"Invalid JSON payload: {exc}") from exc

  if isinstance(payload, str):
    try:
      payload = json.loads(payload)
    except Exception as exc:
      raise ValueError(f"Invalid nested JSON payload: {exc}") from exc

  if not isinstance(payload, dict):
    raise ValueError("QR payload must decode to a JSON object")

  question_number = (
    payload.get("q") or payload.get("question_number") or
    payload.get("questionNumber")
  )
  max_points = (
    payload.get("pts") or payload.get("p") or payload.get("points") or
    payload.get("max_points") or payload.get("maxPoints")
  )
  encrypted_data = (
    payload.get("s") or payload.get("encrypted_data") or
    payload.get("encryptedData")
  )

  if question_number is None:
    raise ValueError("QR payload is missing question number ('q')")
  if max_points is None:
    raise ValueError("QR payload is missing max points ('pts')")

  try:
    parsed_question_number = int(question_number)
  except (ValueError, TypeError) as exc:
    raise ValueError(f"Invalid question number '{question_number}'") from exc

  try:
    parsed_max_points = float(max_points)
  except (ValueError, TypeError) as exc:
    raise ValueError(f"Invalid max points '{max_points}'") from exc

  if parsed_max_points <= 0:
    raise ValueError("max_points must be greater than 0")

  if encrypted_data is not None and not isinstance(encrypted_data, str):
    encrypted_data = str(encrypted_data)

  return {
    "question_number": parsed_question_number,
    "max_points": parsed_max_points,
    "encrypted_data": encrypted_data
  }


def _get_subjective_settings(session_id: int, problem_number: int) -> tuple[str, list[dict]]:
  metadata_repo = ProblemMetadataRepository()
  grading_mode = metadata_repo.get_grading_mode(session_id, problem_number)
  buckets = metadata_repo.get_subjective_buckets(session_id, problem_number)
  if not buckets:
    buckets = [dict(bucket) for bucket in _DEFAULT_SUBJECTIVE_BUCKETS]
  return grading_mode, buckets


def _build_regeneration_response(problem_id: int, problem,
                                 result: dict) -> dict:
  question_type = result.get('question_type')
  seed = result.get('seed')
  version = result.get('version')
  config = result.get('config') or result.get('kwargs')

  # Format answers for display.
  # QuizGenerator may return answer_objects as dict, list, or custom objects.
  answers = []
  answer_objects = result.get('answer_objects')

  iterable_answers = []
  if isinstance(answer_objects, dict):
    iterable_answers = list(answer_objects.items())
  elif isinstance(answer_objects, list):
    iterable_answers = [(f"answer_{idx + 1}", obj)
                        for idx, obj in enumerate(answer_objects)]
  elif answer_objects is not None:
    iterable_answers = [("answer", answer_objects)]

  for key, answer_obj in iterable_answers:
    if isinstance(answer_obj, dict):
      value = answer_obj.get('value')
      if value is None:
        value = answer_obj.get('answer_text')
      if value is None:
        value = answer_obj
      tolerance = answer_obj.get('tolerance')
      html = answer_obj.get('html')
    else:
      value = getattr(answer_obj, 'value', answer_obj)
      tolerance = getattr(answer_obj, 'tolerance', None)
      html = getattr(answer_obj, 'html', None)

    answer_dict = {"key": str(key), "value": str(value)}
    if tolerance is not None:
      answer_dict['tolerance'] = tolerance
    if html is not None:
      answer_dict['html'] = str(html)

    answers.append(answer_dict)

  # Fallback to canvas-formatted answers if answer_objects was absent/empty.
  if not answers:
    answers_payload = result.get('answers')
    if isinstance(answers_payload, dict):
      raw_answers = answers_payload.get('data', [])
    elif isinstance(answers_payload, list):
      raw_answers = answers_payload
    else:
      raw_answers = []

    for idx, raw_answer in enumerate(raw_answers):
      if isinstance(raw_answer, dict):
        key = (raw_answer.get('blank_id') or raw_answer.get('id') or
               raw_answer.get('name') or f"answer_{idx + 1}")
        value = raw_answer.get('answer_text')
        if value is None:
          value = raw_answer.get('value')
        if value is None:
          value = raw_answer
        answer_dict = {"key": str(key), "value": str(value)}
        if raw_answer.get('tolerance') is not None:
          answer_dict['tolerance'] = raw_answer.get('tolerance')
      else:
        answer_dict = {"key": f"answer_{idx + 1}", "value": str(raw_answer)}
      answers.append(answer_dict)

  response = {
    "problem_id": problem_id,
    "problem_number": problem.problem_number,
    "question_type": question_type,
    "seed": seed,
    "version": version,
    "max_points": problem.max_points if problem.max_points is not None else result.get('points'),
    "answers": answers
  }

  if config:
    response['config'] = config
  if 'answer_key_html' in result:
    response['answer_key_html'] = result['answer_key_html']
  if 'explanation_html' in result:
    response['explanation_html'] = result['explanation_html']
  if 'explanation_markdown' in result:
    response['explanation_markdown'] = result['explanation_markdown']

  return response


async def _regenerate_answer_payload(problem) -> dict:
  if not problem.qr_encrypted_data:
    raise ValueError("QR code data not available for this problem")

  session_metadata = SessionRepository().get_metadata(problem.session_id) or {}
  quiz_yaml_text = session_metadata.get("quiz_yaml_text")
  if not isinstance(quiz_yaml_text, str) or not quiz_yaml_text.strip():
    quiz_yaml_text = None

  cache_key = _cache_key_for_regeneration(problem, quiz_yaml_text)
  cached = _get_cached_regeneration(problem.id, cache_key)
  if cached:
    return cached

  result = await asyncio.to_thread(
    regenerate_from_encrypted_compat,
    encrypted_data=problem.qr_encrypted_data,
    points=problem.max_points or 0.0,
    yaml_text=quiz_yaml_text
  )
  response = _build_regeneration_response(problem.id, problem, result)
  _set_cached_regeneration(problem.id, cache_key, response)
  return response


async def _prefetch_session_regeneration(session_id: int) -> None:
  problem_repo = ProblemRepository()
  problems = [
    problem for problem in problem_repo.get_by_session_batch(session_id)
    if problem.qr_encrypted_data
  ]

  if not problems:
    log.info("Regeneration prefetch skipped for session %s: no QR problems", session_id)
    return

  max_workers = min(4, len(problems), os.cpu_count() or 1)
  semaphore = asyncio.Semaphore(max(1, max_workers))
  warmed_count = 0
  failed_count = 0

  async def warm_problem(problem) -> None:
    nonlocal warmed_count, failed_count
    async with semaphore:
      try:
        await _regenerate_answer_payload(problem)
        warmed_count += 1
      except Exception as exc:
        failed_count += 1
        log.debug(
          "Regeneration prefetch failed for problem %s in session %s: %s",
          problem.id,
          session_id,
          exc
        )

  await asyncio.gather(*(warm_problem(problem) for problem in problems))
  log.info(
    "Regeneration prefetch complete for session %s: warmed=%s failed=%s",
    session_id,
    warmed_count,
    failed_count
  )


def extract_problem_image(pdf_data: str,
                          page_number: int,
                          region_y_start: int,
                          region_y_end: int,
                          end_page_number: int = None,
                          end_region_y: int = None,
                          region_y_start_pct: float = None,
                          region_y_end_pct: float = None,
                          end_region_y_pct: float = None,
                          page_transforms: dict = None) -> str:
  """
    Extract a problem image from stored PDF data using region coordinates.
    Supports cross-page regions.

    DEPRECATED: Use ProblemService.extract_image_from_pdf_data() directly.
    This function is kept for backwards compatibility.

    Args:
        pdf_data: Base64 encoded PDF
        page_number: 0-indexed start page number
        region_y_start: Y coordinate of region start on start page
        region_y_end: Y coordinate of region end on start page (or end page if cross-page)
        end_page_number: Optional end page number for cross-page regions
        end_region_y: Optional end y-coordinate for cross-page regions

    Returns:
        Base64 encoded PNG image of the problem region
    """
  return _problem_service.extract_image_from_pdf_data(
    pdf_base64=pdf_data,
    page_number=page_number,
    region_y_start=region_y_start,
    region_y_end=region_y_end,
    end_page_number=end_page_number,
    end_region_y=end_region_y,
    region_y_start_pct=region_y_start_pct,
    region_y_end_pct=region_y_end_pct,
    end_region_y_pct=end_region_y_pct,
    page_transforms=page_transforms)


def get_problem_image_data(problem, submission_repo: SubmissionRepository = None) -> str:
  """
    Get image data for a problem, extracting from PDF if needed.

    Args:
        problem: Problem domain object or dict-like with region_coords, submission_id, id
        submission_repo: Optional SubmissionRepository (creates new if None)

    Returns:
        Base64 encoded PNG image
    """
  # Handle both Problem objects and dict-like rows
  problem_id = problem.id if hasattr(problem, 'id') else problem["id"]
  submission_id = problem.submission_id if hasattr(problem, 'submission_id') else problem["submission_id"]
  region_coords = problem.region_coords if hasattr(problem, 'region_coords') else (
    json.loads(problem["region_coords"]) if problem.get("region_coords") else None
  )

  # Extract from PDF using region metadata from region_coords
  # Note: image_data column removed in v21, always use PDF-based extraction
  if region_coords:
    try:
      # Get PDF data from submission
      if submission_repo is None:
        submission_repo = SubmissionRepository()

      pdf_data = submission_repo.get_pdf_data(submission_id)

      if pdf_data:
        return extract_problem_image(
          pdf_data,
          region_coords["page_number"],
          region_coords["region_y_start"],
          region_coords["region_y_end"],
          region_coords.get("end_page_number"),  # Optional: for cross-page regions
          region_coords.get("end_region_y"),  # Optional: for cross-page regions
          region_coords.get("region_y_start_pct"),
          region_coords.get("region_y_end_pct"),
          region_coords.get("end_region_y_pct"),
          region_coords.get("page_transforms")
        )
      else:
        log.error(
          f"Problem {problem_id}: No PDF data found for submission {submission_id}"
        )
    except (json.JSONDecodeError, KeyError) as e:
      log.error(
        f"Problem {problem_id}: Invalid region_coords data: {str(e)}")
      raise HTTPException(status_code=500,
                          detail=f"Invalid region_coords data: {str(e)}")

  # Fallback: no image data available
  log.error(
    f"Problem {problem_id}: No image data available. has_region_coords={bool(region_coords)}"
  )
  raise HTTPException(
    status_code=500,
    detail="Problem image data not available (no region_coords or PDF data)")


@router.get("/{session_id}/{problem_number}/next",
            response_model=ProblemResponse)
async def get_next_problem(
  session_id: int,
  problem_number: int,
  current_user: dict = Depends(require_session_access())
):
  """Get next ungraded problem for a specific problem number (requires session access)"""
  problem_repo = ProblemRepository()
  submission_repo = SubmissionRepository()

  triage_repo = SubjectiveTriageRepository()
  grading_mode, _ = _get_subjective_settings(session_id, problem_number)

  # In subjective mode, "next" means next untriaged response.
  if grading_mode == "subjective":
    problem = problem_repo.get_next_ungraded_untriaged(session_id, problem_number)
  else:
    problem = problem_repo.get_next_ungraded(session_id, problem_number)
  if not problem:
    raise HTTPException(
      status_code=404,
      detail=(
        f"No untriaged problems found for problem {problem_number}"
        if grading_mode == "subjective"
        else f"No ungraded problems found for problem {problem_number}"
      )
    )

  # Get counts for context (including blank counts)
  counts = problem_repo.get_counts_for_problem_number(session_id, problem_number)
  total_count = counts["total"]
  graded_count = counts["graded"]
  ungraded_blank = counts["ungraded_blank"]
  ungraded_nonblank = counts["ungraded_nonblank"]
  triaged_count = triage_repo.count_ungraded_for_problem_number(
    session_id, problem_number
  )
  untriaged_count = max((total_count - graded_count) - triaged_count, 0)
  current_index = (triaged_count + 1) if grading_mode == "subjective" else (graded_count + 1)

  triage_entry = triage_repo.get_for_problem(problem.id)

  # Get image data (extract from PDF if needed)
  image_data = get_problem_image_data(problem, submission_repo)

  return ProblemResponse(
    id=problem.id,
    problem_number=problem.problem_number,
    submission_id=problem.submission_id,
    image_data=image_data,
    score=problem.score,
    feedback=problem.feedback,
    graded=problem.graded,
    max_points=problem.max_points,
    current_index=current_index,
    total_count=total_count,
    ungraded_blank=ungraded_blank,
    ungraded_nonblank=ungraded_nonblank,
    is_blank=problem.is_blank,
    blank_confidence=problem.blank_confidence,
    blank_method=problem.blank_method,
    blank_reasoning=problem.blank_reasoning,
    ai_reasoning=problem.ai_reasoning,
    has_qr_data=bool(problem.qr_encrypted_data),
    grading_mode=grading_mode,
    subjective_triaged=bool(triage_entry),
    subjective_bucket_id=triage_entry["bucket_id"] if triage_entry else None,
    subjective_notes=triage_entry["notes"] if triage_entry else None,
    subjective_triaged_count=triaged_count,
    subjective_untriaged_count=untriaged_count
  )


@router.get("/{session_id}/{problem_number}/previous",
            response_model=ProblemResponse)
async def get_previous_problem(
  session_id: int,
  problem_number: int,
  current_user: dict = Depends(require_session_access())
):
  """Get most recently graded problem for a specific problem number (requires session access)"""
  problem_repo = ProblemRepository()
  submission_repo = SubmissionRepository()

  triage_repo = SubjectiveTriageRepository()
  grading_mode, _ = _get_subjective_settings(session_id, problem_number)

  # In subjective mode, "previous" means most recently triaged response.
  if grading_mode == "subjective":
    problem = problem_repo.get_previous_triaged(session_id, problem_number)
  else:
    problem = problem_repo.get_previous_graded(session_id, problem_number)
  if not problem:
    raise HTTPException(
      status_code=404,
      detail=(
        f"No triaged problems found for problem {problem_number}"
        if grading_mode == "subjective"
        else f"No graded problems found for problem {problem_number}"
      )
    )

  # Get counts for context (including blank counts)
  counts = problem_repo.get_counts_for_problem_number(session_id, problem_number)
  total_count = counts["total"]
  graded_count = counts["graded"]
  ungraded_blank = counts["ungraded_blank"]
  ungraded_nonblank = counts["ungraded_nonblank"]
  triaged_count = triage_repo.count_ungraded_for_problem_number(
    session_id, problem_number
  )
  untriaged_count = max((total_count - graded_count) - triaged_count, 0)
  current_index = triaged_count if grading_mode == "subjective" else graded_count

  triage_entry = triage_repo.get_for_problem(problem.id)

  # Get image data (extract from PDF if needed)
  image_data = get_problem_image_data(problem, submission_repo)

  return ProblemResponse(
    id=problem.id,
    problem_number=problem.problem_number,
    submission_id=problem.submission_id,
    image_data=image_data,
    score=problem.score,
    feedback=problem.feedback,
    graded=problem.graded,
    max_points=problem.max_points,
    current_index=current_index,
    total_count=total_count,
    ungraded_blank=ungraded_blank,
    ungraded_nonblank=ungraded_nonblank,
    is_blank=problem.is_blank,
    blank_confidence=problem.blank_confidence,
    blank_method=problem.blank_method,
    blank_reasoning=problem.blank_reasoning,
    ai_reasoning=problem.ai_reasoning,
    has_qr_data=bool(problem.qr_encrypted_data),
    grading_mode=grading_mode,
    subjective_triaged=bool(triage_entry),
    subjective_bucket_id=triage_entry["bucket_id"] if triage_entry else None,
    subjective_notes=triage_entry["notes"] if triage_entry else None,
    subjective_triaged_count=triaged_count,
    subjective_untriaged_count=untriaged_count
  )


@router.get("/{session_id}/{problem_number}/bucket/{bucket_id}/next",
            response_model=ProblemResponse)
async def get_next_problem_in_bucket(
  session_id: int,
  problem_number: int,
  bucket_id: str,
  current_problem_id: Optional[int] = None,
  current_user: dict = Depends(require_session_access())
):
  """Get next triaged/ungraded response within one subjective bucket."""
  problem_repo = ProblemRepository()
  submission_repo = SubmissionRepository()
  triage_repo = SubjectiveTriageRepository()
  session_repo = SessionRepository()

  if not session_repo.exists(session_id):
    raise HTTPException(status_code=404, detail="Session not found")

  grading_mode, buckets = _get_subjective_settings(session_id, problem_number)
  if grading_mode != "subjective":
    raise HTTPException(
      status_code=400,
      detail="Bucket navigation is only available in subjective mode"
    )
  valid_bucket_ids = {bucket.get("id") for bucket in buckets}
  if bucket_id not in valid_bucket_ids:
    raise HTTPException(status_code=404, detail="Bucket not found for this problem")

  problem = problem_repo.get_next_triaged_in_bucket(
    session_id, problem_number, bucket_id, current_problem_id
  )
  if not problem:
    raise HTTPException(
      status_code=404,
      detail=f"No triaged problems found in bucket '{bucket_id}'"
    )

  counts = problem_repo.get_counts_for_problem_number(session_id, problem_number)
  total_count = counts["total"]
  graded_count = counts["graded"]
  ungraded_blank = counts["ungraded_blank"]
  ungraded_nonblank = counts["ungraded_nonblank"]
  triaged_count = triage_repo.count_ungraded_for_problem_number(
    session_id, problem_number
  )
  untriaged_count = max((total_count - graded_count) - triaged_count, 0)
  current_index = triaged_count if triaged_count > 0 else 1

  triage_entry = triage_repo.get_for_problem(problem.id)
  image_data = get_problem_image_data(problem, submission_repo)

  return ProblemResponse(
    id=problem.id,
    problem_number=problem.problem_number,
    submission_id=problem.submission_id,
    image_data=image_data,
    score=problem.score,
    feedback=problem.feedback,
    graded=problem.graded,
    max_points=problem.max_points,
    current_index=current_index,
    total_count=total_count,
    ungraded_blank=ungraded_blank,
    ungraded_nonblank=ungraded_nonblank,
    is_blank=problem.is_blank,
    blank_confidence=problem.blank_confidence,
    blank_method=problem.blank_method,
    blank_reasoning=problem.blank_reasoning,
    ai_reasoning=problem.ai_reasoning,
    has_qr_data=bool(problem.qr_encrypted_data),
    grading_mode=grading_mode,
    subjective_triaged=bool(triage_entry),
    subjective_bucket_id=triage_entry["bucket_id"] if triage_entry else None,
    subjective_notes=triage_entry["notes"] if triage_entry else None,
    subjective_triaged_count=triaged_count,
    subjective_untriaged_count=untriaged_count
  )


@router.get("/{session_id}/{problem_number}/bucket/{bucket_id}/previous",
            response_model=ProblemResponse)
async def get_previous_problem_in_bucket(
  session_id: int,
  problem_number: int,
  bucket_id: str,
  current_problem_id: Optional[int] = None,
  current_user: dict = Depends(require_session_access())
):
  """Get previous triaged/ungraded response within one subjective bucket."""
  problem_repo = ProblemRepository()
  submission_repo = SubmissionRepository()
  triage_repo = SubjectiveTriageRepository()
  session_repo = SessionRepository()

  if not session_repo.exists(session_id):
    raise HTTPException(status_code=404, detail="Session not found")

  grading_mode, buckets = _get_subjective_settings(session_id, problem_number)
  if grading_mode != "subjective":
    raise HTTPException(
      status_code=400,
      detail="Bucket navigation is only available in subjective mode"
    )
  valid_bucket_ids = {bucket.get("id") for bucket in buckets}
  if bucket_id not in valid_bucket_ids:
    raise HTTPException(status_code=404, detail="Bucket not found for this problem")

  problem = problem_repo.get_previous_triaged_in_bucket(
    session_id, problem_number, bucket_id, current_problem_id
  )
  if not problem:
    raise HTTPException(
      status_code=404,
      detail=f"No triaged problems found in bucket '{bucket_id}'"
    )

  counts = problem_repo.get_counts_for_problem_number(session_id, problem_number)
  total_count = counts["total"]
  graded_count = counts["graded"]
  ungraded_blank = counts["ungraded_blank"]
  ungraded_nonblank = counts["ungraded_nonblank"]
  triaged_count = triage_repo.count_ungraded_for_problem_number(
    session_id, problem_number
  )
  untriaged_count = max((total_count - graded_count) - triaged_count, 0)
  current_index = triaged_count if triaged_count > 0 else 1

  triage_entry = triage_repo.get_for_problem(problem.id)
  image_data = get_problem_image_data(problem, submission_repo)

  return ProblemResponse(
    id=problem.id,
    problem_number=problem.problem_number,
    submission_id=problem.submission_id,
    image_data=image_data,
    score=problem.score,
    feedback=problem.feedback,
    graded=problem.graded,
    max_points=problem.max_points,
    current_index=current_index,
    total_count=total_count,
    ungraded_blank=ungraded_blank,
    ungraded_nonblank=ungraded_nonblank,
    is_blank=problem.is_blank,
    blank_confidence=problem.blank_confidence,
    blank_method=problem.blank_method,
    blank_reasoning=problem.blank_reasoning,
    ai_reasoning=problem.ai_reasoning,
    has_qr_data=bool(problem.qr_encrypted_data),
    grading_mode=grading_mode,
    subjective_triaged=bool(triage_entry),
    subjective_bucket_id=triage_entry["bucket_id"] if triage_entry else None,
    subjective_notes=triage_entry["notes"] if triage_entry else None,
    subjective_triaged_count=triaged_count,
    subjective_untriaged_count=untriaged_count
  )


@router.get("/{session_id}/{problem_number}/bucket/{bucket_id}/sample",
            response_model=ProblemResponse)
async def get_sample_problem_in_bucket(
  session_id: int,
  problem_number: int,
  bucket_id: str,
  current_user: dict = Depends(require_session_access())
):
  """Get a random triaged/ungraded response from one subjective bucket."""
  problem_repo = ProblemRepository()
  submission_repo = SubmissionRepository()
  triage_repo = SubjectiveTriageRepository()
  session_repo = SessionRepository()

  if not session_repo.exists(session_id):
    raise HTTPException(status_code=404, detail="Session not found")

  grading_mode, buckets = _get_subjective_settings(session_id, problem_number)
  if grading_mode != "subjective":
    raise HTTPException(
      status_code=400,
      detail="Bucket sampling is only available in subjective mode"
    )
  valid_bucket_ids = {bucket.get("id") for bucket in buckets}
  if bucket_id not in valid_bucket_ids:
    raise HTTPException(status_code=404, detail="Bucket not found for this problem")

  problem = problem_repo.get_random_triaged_in_bucket(
    session_id, problem_number, bucket_id
  )
  if not problem:
    raise HTTPException(
      status_code=404,
      detail=f"No triaged problems found in bucket '{bucket_id}'"
    )

  counts = problem_repo.get_counts_for_problem_number(session_id, problem_number)
  total_count = counts["total"]
  graded_count = counts["graded"]
  ungraded_blank = counts["ungraded_blank"]
  ungraded_nonblank = counts["ungraded_nonblank"]
  triaged_count = triage_repo.count_ungraded_for_problem_number(
    session_id, problem_number
  )
  untriaged_count = max((total_count - graded_count) - triaged_count, 0)
  current_index = triaged_count if triaged_count > 0 else 1

  triage_entry = triage_repo.get_for_problem(problem.id)
  image_data = get_problem_image_data(problem, submission_repo)

  return ProblemResponse(
    id=problem.id,
    problem_number=problem.problem_number,
    submission_id=problem.submission_id,
    image_data=image_data,
    score=problem.score,
    feedback=problem.feedback,
    graded=problem.graded,
    max_points=problem.max_points,
    current_index=current_index,
    total_count=total_count,
    ungraded_blank=ungraded_blank,
    ungraded_nonblank=ungraded_nonblank,
    is_blank=problem.is_blank,
    blank_confidence=problem.blank_confidence,
    blank_method=problem.blank_method,
    blank_reasoning=problem.blank_reasoning,
    ai_reasoning=problem.ai_reasoning,
    has_qr_data=bool(problem.qr_encrypted_data),
    grading_mode=grading_mode,
    subjective_triaged=bool(triage_entry),
    subjective_bucket_id=triage_entry["bucket_id"] if triage_entry else None,
    subjective_notes=triage_entry["notes"] if triage_entry else None,
    subjective_triaged_count=triaged_count,
    subjective_untriaged_count=untriaged_count
  )


@router.post("/{problem_id}/grade")
async def grade_problem(
  problem_id: int,
  grade: GradeSubmission,
  current_user: dict = Depends(get_current_user)
):
  """Submit a grade for a problem (requires authentication and session access)

    Special handling: If score is exactly "-" (dash), mark the problem as blank
    and set score to 0. This allows manual blank detection alongside AI heuristics.
    Feedback can still be provided normally for context.
    """
  problem_repo = ProblemRepository()

  # Get problem to check session access
  problem = problem_repo.get_by_id(problem_id)
  if not problem:
    raise HTTPException(status_code=404, detail="Problem not found")

  # Check if user has access to this session
  if current_user["role"] != "instructor":
    from ..repositories.session_assignment_repository import SessionAssignmentRepository
    assignment_repo = SessionAssignmentRepository()
    if not assignment_repo.is_user_assigned(problem.session_id, current_user["user_id"]):
      raise HTTPException(status_code=403, detail="You do not have access to this grading session")

  # Check if score indicates manual blank marking (dash)
  is_manual_blank = isinstance(grade.score, str) and grade.score.strip() == "-"

  if is_manual_blank:
    # Mark as blank with score 0
    problem_repo.mark_as_blank(problem_id, grade.feedback)
  else:
    # Normal grading - convert score to float and save
    try:
      score_value = float(grade.score)
    except (ValueError, TypeError):
      raise HTTPException(
        status_code=400,
        detail=f"Invalid score value: {grade.score}. Must be a number or '-' for blank."
      )

    problem_repo.update_grade(problem_id, score_value, grade.feedback)

  # If this response had a subjective triage assignment, clear it now that
  # the response is explicitly graded.
  SubjectiveTriageRepository().clear(problem_id)

  # Update statistics after grading
  update_problem_stats(problem.session_id)

  return {
    "status": "graded",
    "problem_id": problem_id,
    "is_blank": is_manual_blank
  }


@router.get("/{problem_id}", response_model=ProblemResponse)
async def get_problem(
  problem_id: int,
  current_user: dict = Depends(get_current_user)
):
  """Get a specific problem by ID (requires authentication and session access)"""
  problem_repo = ProblemRepository()
  submission_repo = SubmissionRepository()

  problem = problem_repo.get_by_id(problem_id)
  if not problem:
    raise HTTPException(status_code=404, detail="Problem not found")

  # Check if user has access to this session
  if current_user["role"] != "instructor":
    from ..repositories.session_assignment_repository import SessionAssignmentRepository
    assignment_repo = SessionAssignmentRepository()
    if not assignment_repo.is_user_assigned(problem.session_id, current_user["user_id"]):
      raise HTTPException(status_code=403, detail="You do not have access to this grading session")

  # Get context counts
  counts = problem_repo.get_counts_for_problem_number(problem.session_id, problem.problem_number)
  triage_repo = SubjectiveTriageRepository()
  grading_mode, _ = _get_subjective_settings(problem.session_id, problem.problem_number)
  triaged_count = triage_repo.count_ungraded_for_problem_number(
    problem.session_id, problem.problem_number
  )
  untriaged_count = max((counts["total"] - counts["graded"]) - triaged_count, 0)
  triage_entry = triage_repo.get_for_problem(problem.id)

  # Get image data (extract from PDF if needed)
  image_data = get_problem_image_data(problem, submission_repo)

  return ProblemResponse(
    id=problem.id,
    problem_number=problem.problem_number,
    submission_id=problem.submission_id,
    image_data=image_data,
    score=problem.score,
    feedback=problem.feedback,
    graded=problem.graded,
    current_index=counts["graded"] + 1,
    total_count=counts["total"],
    is_blank=problem.is_blank,
    blank_confidence=problem.blank_confidence,
    blank_method=problem.blank_method,
    blank_reasoning=problem.blank_reasoning,
    ai_reasoning=problem.ai_reasoning,
    has_qr_data=bool(problem.qr_encrypted_data),
    grading_mode=grading_mode,
    subjective_triaged=bool(triage_entry),
    subjective_bucket_id=triage_entry["bucket_id"] if triage_entry else None,
    subjective_notes=triage_entry["notes"] if triage_entry else None,
    subjective_triaged_count=triaged_count,
    subjective_untriaged_count=untriaged_count
  )


@router.get("/{problem_id}/context")
async def get_problem_in_context(
  problem_id: int,
  current_user: dict = Depends(get_current_user)
):
  """
    Get the full page containing this problem, with the problem region highlighted (requires auth and session access).

    Returns:
        JSON with:
        - page_image: Base64 PNG of full page
        - problem_region: Coordinates {y_start, y_end, height} for highlighting
    """
  problem_repo = ProblemRepository()
  submission_repo = SubmissionRepository()

  # Get problem with region metadata
  problem = problem_repo.get_by_id(problem_id)
  if not problem:
    raise HTTPException(status_code=404, detail="Problem not found")

  # Check if user has access to this session
  if current_user["role"] != "instructor":
    from ..repositories.session_assignment_repository import SessionAssignmentRepository
    assignment_repo = SessionAssignmentRepository()
    if not assignment_repo.is_user_assigned(problem.session_id, current_user["user_id"]):
      raise HTTPException(status_code=403, detail="You do not have access to this grading session")

  # Check if PDF-based storage is available
  if not problem.region_coords:
    raise HTTPException(
      status_code=400,
      detail="Context view not available (problem uses legacy image storage)"
    )

  # Get PDF data from submission
  pdf_data = submission_repo.get_pdf_data(problem.submission_id)
  if not pdf_data:
    raise HTTPException(status_code=500,
                        detail="PDF data not found for submission")

  # Extract full page as image
  pdf_bytes = base64.b64decode(pdf_data)
  pdf_document = fitz.open("pdf", pdf_bytes)
  page = pdf_document[problem.region_coords["page_number"]]

  # Convert full page to PNG
  pix = page.get_pixmap(dpi=150)
  img_bytes = pix.tobytes("png")
  page_image_base64 = base64.b64encode(img_bytes).decode("utf-8")

  pdf_document.close()

  return {
    "problem_id": problem_id,
    "page_image": page_image_base64,
    "problem_region": {
      "y_start": problem.region_coords["region_y_start"],
      "y_end": problem.region_coords["region_y_end"],
      "height": problem.region_coords.get("region_height")
    },
    "page_number": problem.region_coords["page_number"]
  }


@router.post("/{problem_id}/decipher")
async def decipher_handwriting(
  problem_id: int,
  model: str = "default",
  current_user: dict = Depends(get_current_user)
):
  """Use AI to transcribe handwritten text from a problem image (requires auth and session access)

    Args:
        problem_id: ID of the problem to transcribe
        model: AI model to use ("default", "ollama", "sonnet", "opus")
               "default" uses Ollama (cheapest, may have quality issues)
    """
  problem_repo = ProblemRepository()
  submission_repo = SubmissionRepository()

  problem = problem_repo.get_by_id(problem_id)
  if not problem:
    raise HTTPException(status_code=404, detail="Problem not found")

  # Check if user has access to this session
  if current_user["role"] != "instructor":
    from ..repositories.session_assignment_repository import SessionAssignmentRepository
    assignment_repo = SessionAssignmentRepository()
    if not assignment_repo.is_user_assigned(problem.session_id, current_user["user_id"]):
      raise HTTPException(status_code=403, detail="You do not have access to this grading session")

  # Get image data (extract from PDF if needed)
  image_base64 = get_problem_image_data(problem, submission_repo)

  # Simple, direct prompt to avoid editorializing or commentary
  query = "Transcribe all handwritten text from this image. Output only the transcribed text."

  try:
    # Select AI provider based on model parameter
    if model == "opus":
      # Use Opus directly via Anthropic client
      ai = ai_helper.AI_Helper__Anthropic()
      response = ai._client.messages.create(
        model="claude-opus-4-20250514",  # Most capable model
        max_tokens=2000,
        messages=[{
          "role":
          "user",
          "content": [{
            "type": "text",
            "text": query
          }, {
            "type": "image",
            "source": {
              "type": "base64",
              "media_type": "image/png",
              "data": image_base64
            }
          }]
        }])
      transcription = response.content[0].text
      model_name = "Opus (Premium)"
    elif model == "sonnet":
      # Use Sonnet via standard query_ai
      ai = ai_helper.AI_Helper__Anthropic()
      response, _ = ai.query_ai(query, attachments=[("png", image_base64)])
      transcription = response
      model_name = "Sonnet"
    elif model == "ollama":
      # Use Ollama explicitly
      ai = ai_helper.AI_Helper__Ollama()
      response, _ = ai.query_ai(query, attachments=[("png", image_base64)])
      transcription = response
      model_name = f"Ollama ({os.getenv('OLLAMA_MODEL', 'qwen3-vl:2b')})"
    else:  # "default" or any other value
      # Default to Ollama (cheapest, may have quality issues)
      ai = ai_helper.AI_Helper__Ollama()
      response, _ = ai.query_ai(query, attachments=[("png", image_base64)])
      transcription = response
      model_name = f"Ollama ({os.getenv('OLLAMA_MODEL', 'qwen3-vl:2b')})"

    # Validate transcription is not empty
    if not transcription or not transcription.strip():
      error_msg = f"Model returned empty transcription. Try a different model (Sonnet or Opus)."
      log.warning(
        f"Empty transcription from {model_name} for problem {problem_id}")
      raise HTTPException(status_code=500, detail=error_msg)

    # Cache Ollama results for future use (to avoid repeated slow requests)
    if model == "ollama":
      problem_repo.update_transcription(problem_id, transcription.strip(), model_name)
      log.info(f"Cached Ollama transcription for problem {problem_id}")

    return {
      "problem_id": problem_id,
      "transcription": transcription.strip(),
      "model": model_name
    }
  except Exception as e:
    import traceback
    log.error(f"Transcription failed: {traceback.format_exc()}")
    raise HTTPException(status_code=500,
                        detail=f"Transcription failed: {str(e)}")


@router.get("/{session_id}/{problem_number}/graded")
async def get_graded_problems(
  session_id: int,
  problem_number: int,
  offset: int = 0,
  limit: int = 20,
  current_user: dict = Depends(require_session_access())
):
  """
    Get graded problems for a specific problem number for review (requires session access).

    Args:
        session_id: Grading session ID
        problem_number: Problem number to fetch
        offset: Pagination offset (default 0)
        limit: Max number of problems to return (default 20)

    Returns:
        List of graded problems with metadata
    """
  problem_repo = ProblemRepository()

  problems_data, total_count = problem_repo.get_graded_with_student_names(
    session_id, problem_number, limit, offset
  )

  if total_count == 0:
    return {"problems": [], "total": 0, "offset": offset, "limit": limit}

  # Format for response
  problems = []
  for row in problems_data:
    problems.append({
      "id": row["id"],
      "problem_number": row["problem_number"],
      "submission_id": row["submission_id"],
      "student_name": row.get("student_name"),
      "score": row["score"],
      "feedback": row["feedback"],
      "max_points": row["max_points"],
      "graded_at": row["graded_at"],
      "is_blank": bool(row["is_blank"])
    })

  return {
    "problems": problems,
    "total": total_count,
    "offset": offset,
    "limit": limit
  }


@router.post("/session/{session_id}/prefetch-regeneration")
async def prefetch_session_regeneration(
  session_id: int,
  current_user: dict = Depends(require_session_access())
):
  """
    Start background regeneration prefetch for all QR-backed problems in a session.

    This warms server-side cache so Show Answer/Explanation opens faster later.
  """
  problem_repo = ProblemRepository()
  total_qr_problems = sum(
    1 for problem in problem_repo.get_by_session_batch(session_id)
    if problem.qr_encrypted_data
  )

  if total_qr_problems == 0:
    return {
      "status": "no_qr_data",
      "session_id": session_id,
      "total_qr_problems": 0
    }

  with _session_prefetch_tasks_lock:
    existing_task = _session_prefetch_tasks.get(session_id)
    if existing_task and not existing_task.done():
      return {
        "status": "already_running",
        "session_id": session_id,
        "total_qr_problems": total_qr_problems
      }

    task = asyncio.create_task(_prefetch_session_regeneration(session_id))
    _session_prefetch_tasks[session_id] = task

  def _cleanup_prefetch_task(done_task: asyncio.Task) -> None:
    with _session_prefetch_tasks_lock:
      current = _session_prefetch_tasks.get(session_id)
      if current is done_task:
        _session_prefetch_tasks.pop(session_id, None)

  task.add_done_callback(_cleanup_prefetch_task)

  return {
    "status": "started",
    "session_id": session_id,
    "total_qr_problems": total_qr_problems
  }


@router.get("/{problem_id}/regenerate-answer")
async def regenerate_answer(
  problem_id: int,
  current_user: dict = Depends(get_current_user)
):
  """
    Regenerate the correct answer from QR code metadata (requires auth and session access).

    This endpoint uses the question_type, seed, and version stored from
    the QR code to regenerate the original correct answer.

    Args:
        problem_id: ID of the problem

    Returns:
        JSON with regenerated answers or error if QR metadata not available
    """
  problem_repo = ProblemRepository()

  problem = problem_repo.get_by_id(problem_id)
  if not problem:
    raise HTTPException(status_code=404, detail="Problem not found")

  # Check if user has access to this session
  if current_user["role"] != "instructor":
    from ..repositories.session_assignment_repository import SessionAssignmentRepository
    assignment_repo = SessionAssignmentRepository()
    if not assignment_repo.is_user_assigned(problem.session_id, current_user["user_id"]):
      raise HTTPException(status_code=403, detail="You do not have access to this grading session")

  # Check if QR encrypted data is available
  if not problem.qr_encrypted_data:
    raise HTTPException(status_code=400,
                        detail="QR code data not available for this problem")

  try:
    return await _regenerate_answer_payload(problem)

  except ImportError:
    raise HTTPException(
      status_code=500,
      detail=
      "QuizGenerator module not available. Please install it (pip install QuizGenerator>=0.4.0) to use answer regeneration."
    )
  except ValueError as e:
    error_msg = str(e)
    if "Must provide yaml_path, yaml_text, or yaml_docs." in error_msg:
      raise HTTPException(
        status_code=400,
        detail=
        "This problem uses YAML-based regeneration. Upload the quiz YAML file for this session before regenerating answers."
      )
    raise HTTPException(status_code=500,
                        detail=f"Failed to regenerate answer: {error_msg}")
  except Exception as e:
    raise HTTPException(
      status_code=500,
      detail=f"Unexpected error during answer regeneration: {str(e)}")


@router.post("/{problem_id}/subjective-triage")
async def assign_subjective_triage(
  problem_id: int,
  request: SubjectiveTriageSubmission,
  current_user: dict = Depends(get_current_user)
):
  """Assign current response to a subjective grading bucket."""
  problem_repo = ProblemRepository()
  triage_repo = SubjectiveTriageRepository()

  problem = problem_repo.get_by_id(problem_id)
  if not problem:
    raise HTTPException(status_code=404, detail="Problem not found")

  if current_user["role"] != "instructor":
    from ..repositories.session_assignment_repository import SessionAssignmentRepository
    assignment_repo = SessionAssignmentRepository()
    if not assignment_repo.is_user_assigned(problem.session_id, current_user["user_id"]):
      raise HTTPException(status_code=403, detail="You do not have access to this grading session")

  grading_mode, buckets = _get_subjective_settings(
    problem.session_id, problem.problem_number
  )
  if grading_mode != "subjective":
    raise HTTPException(
      status_code=400,
      detail="Subjective triage is only available when grading mode is subjective"
    )

  valid_bucket_ids = {bucket.get("id") for bucket in buckets}
  if request.bucket_id not in valid_bucket_ids:
    raise HTTPException(
      status_code=400,
      detail=f"Unknown bucket id '{request.bucket_id}' for this problem"
    )

  triage_repo.upsert(
    problem_id=problem.id,
    session_id=problem.session_id,
    problem_number=problem.problem_number,
    bucket_id=request.bucket_id,
    notes=request.notes
  )

  counts = problem_repo.get_counts_for_problem_number(problem.session_id, problem.problem_number)
  triaged_count = triage_repo.count_ungraded_for_problem_number(
    problem.session_id, problem.problem_number
  )
  untriaged_count = max((counts["total"] - counts["graded"]) - triaged_count, 0)

  return {
    "status": "triaged",
    "problem_id": problem_id,
    "problem_number": problem.problem_number,
    "bucket_id": request.bucket_id,
    "triaged_count": triaged_count,
    "untriaged_count": untriaged_count
  }


@router.delete("/{problem_id}/subjective-triage")
async def clear_subjective_triage(
  problem_id: int,
  current_user: dict = Depends(get_current_user)
):
  """Clear subjective triage assignment for a response."""
  problem_repo = ProblemRepository()
  triage_repo = SubjectiveTriageRepository()

  problem = problem_repo.get_by_id(problem_id)
  if not problem:
    raise HTTPException(status_code=404, detail="Problem not found")

  if current_user["role"] != "instructor":
    from ..repositories.session_assignment_repository import SessionAssignmentRepository
    assignment_repo = SessionAssignmentRepository()
    if not assignment_repo.is_user_assigned(problem.session_id, current_user["user_id"]):
      raise HTTPException(status_code=403, detail="You do not have access to this grading session")

  triage_repo.clear(problem_id)
  counts = problem_repo.get_counts_for_problem_number(problem.session_id, problem.problem_number)
  triaged_count = triage_repo.count_ungraded_for_problem_number(
    problem.session_id, problem.problem_number
  )
  untriaged_count = max((counts["total"] - counts["graded"]) - triaged_count, 0)

  return {
    "status": "cleared",
    "problem_id": problem_id,
    "problem_number": problem.problem_number,
    "triaged_count": triaged_count,
    "untriaged_count": untriaged_count
  }


@router.post("/{problem_id}/manual-qr")
async def apply_manual_qr_payload(
  problem_id: int,
  request: ManualQRCodeSubmission,
  current_user: dict = Depends(get_current_user)
):
  """
    Manually apply decoded QR JSON payload for the current problem.
    Accepts payload text pasted from external QR decode tools.
  """
  from ..repositories import with_transaction

  problem_repo = ProblemRepository()

  problem = problem_repo.get_by_id(problem_id)
  if not problem:
    raise HTTPException(status_code=404, detail="Problem not found")

  if current_user["role"] != "instructor":
    from ..repositories.session_assignment_repository import SessionAssignmentRepository
    assignment_repo = SessionAssignmentRepository()
    if not assignment_repo.is_user_assigned(problem.session_id, current_user["user_id"]):
      raise HTTPException(status_code=403, detail="You do not have access to this grading session")

  try:
    qr_data = _parse_manual_qr_payload(request.payload_text)
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc))

  if qr_data["question_number"] != problem.problem_number:
    raise HTTPException(
      status_code=400,
      detail=(
        f"QR payload question number {qr_data['question_number']} does not match "
        f"current problem number {problem.problem_number}"
      )
    )

  with with_transaction() as repos:
    repos.problems.update_qr_data(
      problem_id,
      qr_data["max_points"],
      qr_data.get("encrypted_data")
    )
    repos.metadata.upsert_max_points(
      problem.session_id,
      problem.problem_number,
      qr_data["max_points"]
    )

  _clear_cached_regeneration(problem_id)

  has_qr_data = bool(qr_data.get("encrypted_data"))
  return {
    "status": "success",
    "problem_id": problem_id,
    "problem_number": problem.problem_number,
    "max_points": qr_data["max_points"],
    "has_qr_data": has_qr_data,
    "message": (
      f"Manual QR payload applied for Problem {problem.problem_number} "
      f"(max points: {qr_data['max_points']})"
      if has_qr_data else
      f"Manual payload applied for Problem {problem.problem_number}, but no encrypted answer metadata was present"
    )
  }


@router.post("/{problem_id}/rescan-qr")
async def rescan_qr_for_single_problem(
  problem_id: int,
  dpi: int = 600,
  current_user: dict = Depends(get_current_user)
):
  """
    Re-scan QR code for a specific problem instance at a specified DPI (requires auth and session access).
    This is useful when the initial scan fails to detect the QR code.

    Args:
        problem_id: The specific problem ID to re-scan
        dpi: DPI to use for rendering (default 600, higher = better for complex QR codes)

    Returns:
        Statistics about QR code found and updated
    """
  # Import required modules
  from ..services.qr_scanner import QRScanner

  log.info(f"Re-scanning QR code for problem ID {problem_id} at {dpi} DPI")

  # Initialize QR scanner
  qr_scanner = QRScanner()
  if not qr_scanner.available:
    raise HTTPException(
      status_code=400,
      detail="QR scanner not available (opencv-python or pyzbar not installed)"
    )

  from ..repositories import with_transaction, ProblemMetadataRepository

  # Get problem and submission data
  problem_repo = ProblemRepository()
  submission_repo = SubmissionRepository()

  # Get problem to check session access
  problem = problem_repo.get_by_id(problem_id)
  if not problem:
    raise HTTPException(status_code=404, detail="Problem not found")

  # Check if user has access to this session
  if current_user["role"] != "instructor":
    from ..repositories.session_assignment_repository import SessionAssignmentRepository
    assignment_repo = SessionAssignmentRepository()
    if not assignment_repo.is_user_assigned(problem.session_id, current_user["user_id"]):
      raise HTTPException(status_code=403, detail="You do not have access to this grading session")

  if not problem.region_coords:
    raise HTTPException(status_code=400,
                        detail="Problem has no region coordinates")

  # Get PDF data
  pdf_base64 = submission_repo.get_pdf_data(problem.submission_id)
  if not pdf_base64:
    raise HTTPException(status_code=404, detail="No PDF data found for submission")

  # Decode PDF
  pdf_bytes = base64.b64decode(pdf_base64)
  pdf_document = fitz.open("pdf", pdf_bytes)

  # Parse region coordinates
  start_page = problem.region_coords["page_number"]
  start_y = problem.region_coords["region_y_start"]
  end_page = problem.region_coords.get("end_page_number", start_page)
  end_y = problem.region_coords["region_y_end"]

  # Use ProblemService to extract the region at higher DPI
  problem_image_base64, _ = _problem_service.extract_image_from_document(
    pdf_document,
    start_page,
    start_y,
    end_page,
    end_y,
    page_transforms=problem.region_coords.get("page_transforms"),
    dpi=dpi)

  # Scan for QR code
  qr_data = qr_scanner.scan_qr_from_image(problem_image_base64)

  pdf_document.close()

  if qr_data:
    log.info(
      f"Problem {problem.problem_number} (ID {problem_id}): Found QR code with max_points={qr_data['max_points']}"
    )

    # Update problem and metadata in transaction
    with with_transaction() as repos:
      # Update problem with QR data
      repos.problems.update_qr_data(problem_id, qr_data["max_points"], qr_data.get("encrypted_data"))

      # Also update problem_metadata for this session
      repos.metadata.upsert_max_points(problem.session_id, problem.problem_number, qr_data["max_points"])

    _clear_cached_regeneration(problem_id)

    log.info(
      f"QR re-scan complete for problem ID {problem_id}: QR code found and updated"
    )

    return {
      "status": "success",
      "problem_id": problem_id,
      "problem_number": problem.problem_number,
      "qr_found": True,
      "max_points": qr_data["max_points"],
      "dpi_used": dpi,
      "message": f"Successfully found and updated QR code for Problem {problem.problem_number} (max points: {qr_data['max_points']}) at {dpi} DPI."
    }
  else:
    log.warning(
      f"Problem {problem.problem_number} (ID {problem_id}): No QR code found at {dpi} DPI"
    )

    return {
      "status": "success",
      "problem_id": problem_id,
      "problem_number": problem.problem_number,
      "qr_found": False,
      "dpi_used": dpi,
      "message": f"No QR code found for Problem {problem.problem_number} at {dpi} DPI. Try increasing DPI or check if QR code is present."
    }
