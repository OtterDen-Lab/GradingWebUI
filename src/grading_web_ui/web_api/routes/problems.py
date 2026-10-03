"""
Problem grading endpoints.
"""
import textwrap
import os
import asyncio
import threading
import hashlib
from uuid import uuid4
from time import perf_counter

from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
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
from ..services.feedback_text import (
  merge_general_feedback,
  extract_response_specific_feedback,
)
from ..services.quiz_regeneration import regenerate_from_encrypted_compat
from ..services.qr_scanner import qr_matches_problem_number
from ..auth import require_session_access, get_current_user
from ..services.model_settings import (
  BUILT_IN_TRANSCRIPTION_INSTRUCTIONS,
  get_handwriting_default,
  get_transcription_additional_instructions,
  resolve_model,
)
from ..services import ollama_settings
from ..services import model_latency

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
_handwriting_jobs = {}
_handwriting_jobs_lock = threading.Lock()
_HANDWRITING_MAX_RESPONSE_TOKENS = 4096

_DEFAULT_SUBJECTIVE_BUCKETS = [
  {"id": "above_beyond", "label": "Above and beyond", "color": "#16a34a"},
  {"id": "has_everything", "label": "Has everything", "color": "#2563eb"},
  {"id": "missing_little", "label": "Missing a little", "color": "#f59e0b"},
  {"id": "missing_lot", "label": "Missing a lot", "color": "#ef4444"},
  {"id": "off_topic", "label": "Off topic", "color": "#6b7280"},
  {"id": "blank", "label": "Blank", "color": "#9ca3af"},
]
TAG_SIGNATURE_DELIMITER = "|"


def _display_feedback(problem) -> Optional[str]:
  metadata_repo = ProblemMetadataRepository()
  default_feedback_row = metadata_repo.get_default_feedback(
    problem.session_id,
    problem.problem_number,
  )
  default_feedback = default_feedback_row[0] if default_feedback_row else None
  return merge_general_feedback(default_feedback, problem.feedback)


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


def _serialize_regeneration_cache_key(cache_key: tuple) -> str:
  return json.dumps(
    list(cache_key), separators=(",", ":"), ensure_ascii=False
  )


def _get_cached_regeneration(problem_id: int, cache_key: tuple) -> Optional[dict]:
  with _regeneration_cache_lock:
    entry = _regeneration_cache.get(problem_id)
    if entry and entry.get("cache_key") == cache_key:
      return entry.get("response")

  persisted = ProblemRepository().get_regeneration_cache(problem_id)
  if not persisted:
    return None

  cache_key_str = _serialize_regeneration_cache_key(cache_key)
  if persisted.get("cache_key") != cache_key_str:
    return None

  response = persisted.get("response")
  if not isinstance(response, dict):
    return None

  with _regeneration_cache_lock:
    _regeneration_cache[problem_id] = {
      "cache_key": cache_key,
      "response": response
    }
    if len(_regeneration_cache) > _regeneration_cache_max_entries:
      oldest_key = next(iter(_regeneration_cache))
      _regeneration_cache.pop(oldest_key, None)

  return response


def _set_cached_regeneration(problem_id: int, cache_key: tuple,
                             response: dict) -> None:
  cache_key_str = _serialize_regeneration_cache_key(cache_key)

  with _regeneration_cache_lock:
    _regeneration_cache[problem_id] = {
      "cache_key": cache_key,
      "response": response
    }
    if len(_regeneration_cache) > _regeneration_cache_max_entries:
      oldest_key = next(iter(_regeneration_cache))
      _regeneration_cache.pop(oldest_key, None)

  try:
    ProblemRepository().set_regeneration_cache(problem_id, cache_key_str, response)
  except Exception as exc:
    log.warning(
      "Failed to persist regeneration cache for problem %s: %s",
      problem_id,
      exc
    )


def _clear_cached_regeneration(problem_id: int) -> None:
  with _regeneration_cache_lock:
    _regeneration_cache.pop(problem_id, None)
  try:
    ProblemRepository().clear_regeneration_cache(problem_id)
  except Exception as exc:
    log.warning(
      "Failed to clear persistent regeneration cache for problem %s: %s",
      problem_id,
      exc
    )


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

  def _first_present(*keys: str):
    for key in keys:
      if key in payload and payload[key] is not None:
        return payload[key]
    return None

  question_number = _first_present("q", "question_number", "questionNumber")
  max_points = _first_present("pts", "p", "points", "max_points", "maxPoints")
  encrypted_data = _first_present("s", "encrypted_data", "encryptedData")

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

  if parsed_max_points < 0 or parsed_max_points > 100:
    raise ValueError("max_points must be between 0 and 100")

  if encrypted_data is not None and not isinstance(encrypted_data, str):
    encrypted_data = str(encrypted_data)

  return {
    "question_number": parsed_question_number,
    "max_points": parsed_max_points,
    "encrypted_data": encrypted_data
  }


def _parse_exclude_problem_ids_param(raw_value: Optional[str]) -> list[int]:
  if not raw_value:
    return []

  exclude_ids: list[int] = []
  for raw_part in raw_value.split(","):
    value = raw_part.strip()
    if not value:
      continue
    try:
      parsed = int(value)
    except (TypeError, ValueError) as exc:
      raise ValueError(f"Invalid problem id '{value}' in exclude_problem_ids") from exc
    if parsed <= 0:
      raise ValueError(f"Problem id must be positive in exclude_problem_ids: {value}")
    exclude_ids.append(parsed)

  # Preserve order but remove duplicates.
  return list(dict.fromkeys(exclude_ids))


def _parse_model_csv(raw_value: str) -> list[str]:
  return [
    value.strip() for value in (raw_value or "").split(",") if value.strip()
  ]


def _is_truthy(raw_value: str) -> bool:
  return (raw_value or "").strip().lower() in ("1", "true", "yes", "on")


def _get_subjective_settings(session_id: int, problem_number: int) -> tuple[str, list[dict]]:
  metadata_repo = ProblemMetadataRepository()
  grading_mode = metadata_repo.get_grading_mode(session_id, problem_number)
  buckets = metadata_repo.get_subjective_buckets(session_id, problem_number)
  if not buckets:
    buckets = [dict(bucket) for bucket in _DEFAULT_SUBJECTIVE_BUCKETS]
  return grading_mode, buckets


def _is_grouping_mode(grading_mode: Optional[str]) -> bool:
  return grading_mode in ("subjective", "tag")


def _normalize_tag_ids(raw_tag_ids: list[str]) -> list[str]:
  normalized = {
    str(tag_id).strip()
    for tag_id in raw_tag_ids
    if tag_id is not None and str(tag_id).strip()
  }
  return sorted(normalized)


def _canonical_tag_signature(tag_ids: list[str]) -> str:
  normalized = _normalize_tag_ids(tag_ids)
  return TAG_SIGNATURE_DELIMITER.join(normalized)


def _tag_ids_from_signature(signature: Optional[str]) -> list[str]:
  if not signature:
    return []
  return _normalize_tag_ids(signature.split(TAG_SIGNATURE_DELIMITER))


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


def _get_regenerated_question_text(problem) -> Optional[str]:
  """Regenerate the exact seeded question for one QR-backed response.

  The frozen YAML is session-scoped, but a QR seed may make the rendered
  question different for each student. Therefore this deliberately uses the
  per-problem regeneration cache instead of problem_metadata.question_text.
  """
  if not problem.qr_encrypted_data:
    return None

  session_metadata = SessionRepository().get_metadata(problem.session_id) or {}
  quiz_yaml_text = session_metadata.get("quiz_yaml_text")
  if not isinstance(quiz_yaml_text, str) or not quiz_yaml_text.strip():
    quiz_yaml_text = None
  cache_key = _cache_key_for_regeneration(problem, quiz_yaml_text)
  response = _get_cached_regeneration(problem.id, cache_key)
  if response is None:
    result = regenerate_from_encrypted_compat(
      encrypted_data=problem.qr_encrypted_data,
      points=problem.max_points or 0.0,
      yaml_text=quiz_yaml_text,
      image_mode="none",
    )
    response = _build_regeneration_response(problem.id, problem, result)
    _set_cached_regeneration(problem.id, cache_key, response)

  question_html = response.get("answer_key_html")
  if not isinstance(question_html, str) or not question_html.strip():
    return None
  return question_html.strip()


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
  exclude_problem_ids: Optional[str] = None,
  current_user: dict = Depends(require_session_access())
):
  """Get next ungraded problem for a specific problem number (requires session access)"""
  problem_repo = ProblemRepository()
  submission_repo = SubmissionRepository()

  triage_repo = SubjectiveTriageRepository()
  grading_mode, _ = _get_subjective_settings(session_id, problem_number)
  try:
    excluded_ids = _parse_exclude_problem_ids_param(exclude_problem_ids)
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc))

  # In grouping modes, "next" means next untriaged response.
  if _is_grouping_mode(grading_mode):
    problem = problem_repo.get_next_ungraded_untriaged(
      session_id,
      problem_number,
      excluded_ids
    )
  else:
    problem = problem_repo.get_next_ungraded(
      session_id,
      problem_number,
      excluded_ids
    )
  if not problem:
    raise HTTPException(
      status_code=404,
      detail=(
        f"No untriaged problems found for problem {problem_number}"
        if _is_grouping_mode(grading_mode)
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
  current_index = (triaged_count + 1) if _is_grouping_mode(grading_mode) else (graded_count + 1)

  triage_entry = triage_repo.get_for_problem(problem.id)

  # Get image data (extract from PDF if needed)
  image_data = get_problem_image_data(problem, submission_repo)

  return ProblemResponse(
    id=problem.id,
    problem_number=problem.problem_number,
    submission_id=problem.submission_id,
    image_data=image_data,
    score=problem.score,
    feedback=_display_feedback(problem),
    response_specific_feedback=problem.feedback,
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
    transcription_is_blank=problem.transcription_is_blank,
    transcription_is_effectively_blank=problem.transcription_is_effectively_blank,
    transcription_is_relevant=problem.transcription_is_relevant,
    transcription_model=problem.transcription_model,
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

  # In grouping modes, prefer most recently triaged (ungraded) response.
  # If everything is already finalized for this problem, fall back to the
  # most recently graded response so the UI can still reopen/adjust scores.
  if _is_grouping_mode(grading_mode):
    problem = problem_repo.get_previous_triaged(session_id, problem_number)
    if not problem:
      problem = problem_repo.get_previous_graded(session_id, problem_number)
  else:
    problem = problem_repo.get_previous_graded(session_id, problem_number)
  if not problem:
    raise HTTPException(
      status_code=404,
      detail=(
        f"No triaged or graded problems found for problem {problem_number}"
        if _is_grouping_mode(grading_mode)
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
  current_index = triaged_count if _is_grouping_mode(grading_mode) else graded_count

  triage_entry = triage_repo.get_for_problem(problem.id)

  # Get image data (extract from PDF if needed)
  image_data = get_problem_image_data(problem, submission_repo)

  return ProblemResponse(
    id=problem.id,
    problem_number=problem.problem_number,
    submission_id=problem.submission_id,
    image_data=image_data,
    score=problem.score,
    feedback=_display_feedback(problem),
    response_specific_feedback=problem.feedback,
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
    transcription_is_blank=problem.transcription_is_blank,
    transcription_is_effectively_blank=problem.transcription_is_effectively_blank,
    transcription_is_relevant=problem.transcription_is_relevant,
    transcription_model=problem.transcription_model,
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
  if not _is_grouping_mode(grading_mode):
    raise HTTPException(
      status_code=400,
      detail="Bucket navigation is only available in subjective or tag mode"
    )
  if grading_mode == "subjective":
    valid_bucket_ids = {bucket.get("id") for bucket in buckets}
    if bucket_id not in valid_bucket_ids:
      raise HTTPException(status_code=404, detail="Bucket not found for this problem")
  else:
    active_bucket_ids = set(
      triage_repo.get_bucket_counts(session_id, problem_number).keys()
    )
    if bucket_id not in active_bucket_ids:
      raise HTTPException(status_code=404, detail="Tag signature not found for this problem")

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
    feedback=_display_feedback(problem),
    response_specific_feedback=problem.feedback,
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
    transcription_is_blank=problem.transcription_is_blank,
    transcription_is_effectively_blank=problem.transcription_is_effectively_blank,
    transcription_is_relevant=problem.transcription_is_relevant,
    transcription_model=problem.transcription_model,
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
  if not _is_grouping_mode(grading_mode):
    raise HTTPException(
      status_code=400,
      detail="Bucket navigation is only available in subjective or tag mode"
    )
  if grading_mode == "subjective":
    valid_bucket_ids = {bucket.get("id") for bucket in buckets}
    if bucket_id not in valid_bucket_ids:
      raise HTTPException(status_code=404, detail="Bucket not found for this problem")
  else:
    active_bucket_ids = set(
      triage_repo.get_bucket_counts(session_id, problem_number).keys()
    )
    if bucket_id not in active_bucket_ids:
      raise HTTPException(status_code=404, detail="Tag signature not found for this problem")

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
    feedback=_display_feedback(problem),
    response_specific_feedback=problem.feedback,
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
    transcription_is_blank=problem.transcription_is_blank,
    transcription_is_effectively_blank=problem.transcription_is_effectively_blank,
    transcription_is_relevant=problem.transcription_is_relevant,
    transcription_model=problem.transcription_model,
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
  if not _is_grouping_mode(grading_mode):
    raise HTTPException(
      status_code=400,
      detail="Bucket sampling is only available in subjective or tag mode"
    )
  if grading_mode == "subjective":
    valid_bucket_ids = {bucket.get("id") for bucket in buckets}
    if bucket_id not in valid_bucket_ids:
      raise HTTPException(status_code=404, detail="Bucket not found for this problem")
  else:
    active_bucket_ids = set(
      triage_repo.get_bucket_counts(session_id, problem_number).keys()
    )
    if bucket_id not in active_bucket_ids:
      raise HTTPException(status_code=404, detail="Tag signature not found for this problem")

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
    feedback=_display_feedback(problem),
    response_specific_feedback=problem.feedback,
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
    transcription_is_blank=problem.transcription_is_blank,
    transcription_is_effectively_blank=problem.transcription_is_effectively_blank,
    transcription_is_relevant=problem.transcription_is_relevant,
    transcription_model=problem.transcription_model,
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
  metadata_repo = ProblemMetadataRepository()

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
  default_feedback_row = metadata_repo.get_default_feedback(
    problem.session_id, problem.problem_number
  )
  default_feedback = default_feedback_row[0] if default_feedback_row else None
  stored_feedback = extract_response_specific_feedback(
    default_feedback,
    grade.feedback
  )

  if is_manual_blank:
    # Mark as blank with score 0
    problem_repo.mark_as_blank(problem_id, stored_feedback)
  else:
    # Normal grading - convert score to float and save
    try:
      score_value = float(grade.score)
    except (ValueError, TypeError):
      raise HTTPException(
        status_code=400,
        detail=f"Invalid score value: {grade.score}. Must be a number or '-' for blank."
      )

    problem_repo.update_grade(problem_id, score_value, stored_feedback)

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
    feedback=_display_feedback(problem),
    response_specific_feedback=problem.feedback,
    graded=problem.graded,
    current_index=counts["graded"] + 1,
    total_count=counts["total"],
    is_blank=problem.is_blank,
    blank_confidence=problem.blank_confidence,
    blank_method=problem.blank_method,
    blank_reasoning=problem.blank_reasoning,
    ai_reasoning=problem.ai_reasoning,
    transcription_is_blank=problem.transcription_is_blank,
    transcription_is_effectively_blank=problem.transcription_is_effectively_blank,
    transcription_is_relevant=problem.transcription_is_relevant,
    transcription_model=problem.transcription_model,
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


def _parse_handwriting_analysis(
    raw_response: str) -> tuple[str, bool, bool | None, bool | None]:
  """Parse structured transcription, tolerating smaller-model JSON variants."""
  payload_text = (raw_response or "").strip()
  if payload_text.startswith("```") and payload_text.endswith("```"):
    payload_text = "\n".join(payload_text.splitlines()[1:-1]).strip()
  try:
    payload = json.loads(payload_text)
  except (TypeError, json.JSONDecodeError) as error:
    # Some smaller vision models prepend a short explanation despite the
    # instruction. Accept a JSON object embedded in that response, but do not
    # fall back to treating arbitrary prose as a transcription result.
    start = payload_text.find("{")
    if start < 0:
      raise ValueError("Model did not return handwriting-analysis JSON") from error
    try:
      payload, _ = json.JSONDecoder().raw_decode(payload_text[start:])
    except json.JSONDecodeError as nested_error:
      raise ValueError("Model did not return valid handwriting-analysis JSON") from nested_error

  if not isinstance(payload, dict):
    raise ValueError("Handwriting-analysis JSON must be an object")

  missing = object()

  def read_bool(field: str, *aliases: str, required: bool = False) -> bool | None:
    value = next((payload[key] for key in (field, *aliases) if key in payload),
                 missing)
    if value is missing:
      if required:
        raise ValueError(f"Handwriting-analysis JSON is missing {field}")
      return None
    if isinstance(value, bool):
      return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
      return value.strip().lower() == "true"
    raise ValueError(f"Handwriting-analysis {field} must be true or false")

  is_blank = read_bool("is_blank", "blank", required=True)
  is_effectively_blank = read_bool(
    "is_effectively_blank", "effectively_blank")
  is_relevant = read_bool("is_relevant", "relevant")
  text = payload.get("text", payload.get("transcription"))
  # Some smaller vision models emit JSON null for a blank transcription even
  # when they correctly set is_blank=true. Treat that as the explicit blank
  # marker rather than discarding an otherwise valid analysis result.
  if text is None and is_blank:
    text = "[blank]"
  if not isinstance(text, str):
    raise ValueError("Handwriting-analysis text must be a string")

  transcription = text.strip()
  if is_blank and not transcription:
    transcription = "[blank]"
  if not transcription and not is_blank:
    raise ValueError("Model returned an empty transcription without marking it blank")
  return transcription, is_blank, is_effectively_blank, is_relevant


def _parse_relevance_classification(raw_response: str) -> bool:
  """Parse the lightweight text-only relevance JSON response."""
  payload_text = (raw_response or "").strip()
  if payload_text.startswith("```") and payload_text.endswith("```"):
    payload_text = "\n".join(payload_text.splitlines()[1:-1]).strip()
  try:
    payload = json.loads(payload_text)
  except (TypeError, json.JSONDecodeError) as error:
    raise ValueError("Model did not return relevance-classification JSON") from error
  if not isinstance(payload, dict):
    raise ValueError("Relevance-classification JSON must be an object")

  def read_bool(field: str) -> bool:
    value = payload.get(field)
    if isinstance(value, bool):
      return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
      return value.strip().lower() == "true"
    raise ValueError(f"Relevance-classification {field} must be true or false")

  return read_bool("is_relevant")


def _decipher_handwriting(problem_id: int, model: str, user_id: int,
                          skip_cached: bool = False) -> dict:
  """Transcribe and persist one response; optionally leave a cached result intact.

    This is deliberately synchronous: the batch caller runs it in FastAPI's
    background-task thread pool, one response at a time, to avoid saturating
    the configured AI provider.
    """
  problem_repo = ProblemRepository()
  submission_repo = SubmissionRepository()

  problem = problem_repo.get_by_id(problem_id)
  if not problem:
    raise HTTPException(status_code=404, detail="Problem not found")

  if skip_cached and problem.transcription and problem.transcription.strip():
    return {
      "problem_id": problem_id,
      "transcription": problem.transcription,
      "model": problem.transcription_model,
      "is_blank": problem.transcription_is_blank,
      "is_effectively_blank": problem.transcription_is_effectively_blank,
      "is_relevant": problem.transcription_is_relevant,
      "cached": True,
    }

  # Get image data (extract from PDF if needed)
  image_base64 = get_problem_image_data(problem, submission_repo)

  query = BUILT_IN_TRANSCRIPTION_INSTRUCTIONS
  additional_instructions = get_transcription_additional_instructions(
    user_id)["text"]
  if additional_instructions:
    query += f"\n\nAdditional transcription instructions:\n{additional_instructions}"

  timing_start = None
  timing_provider = None
  timing_model = None
  timing_server = None
  try:
    selected_model = (model or "default").strip().lower()
    if selected_model == "default":
      selected_model = get_handwriting_default(user_id)["target"]

    if selected_model == "ollama":
      server = ollama_settings.get_active_server()
      if not server:
        raise HTTPException(
          status_code=400,
          detail="No active Ollama model is configured in Settings")
      ai = ai_helper.AI_Helper__Ollama(server["base_url"], server["active_model"])
      timing_start = perf_counter()
      timing_provider = "ollama"
      timing_model = server["active_model"]
      timing_server = server["name"]
      transcription, usage = ai.query_ai(
        query, attachments=[("png", image_base64)],
        max_response_tokens=_HANDWRITING_MAX_RESPONSE_TOKENS,
        json_output=True)
      log.debug("Ollama handwriting transcription response for problem %s: %r",
                problem_id, transcription)
      model_name = f"Ollama ({usage.get('model', server['active_model'])} on {server['name']})"
    else:

    # Compatibility aliases preserve old links while all choices now resolve
    # dynamically from persistent settings instead of the grading session.
      tier = {"default": "medium", "sonnet": "medium", "opus": "large"}.get(
        selected_model, selected_model)
      provider = "anthropic"
      explicit_model = None
      if ":" in selected_model:
        provider, explicit_model = selected_model.split(":", 1)
        tier = "medium"
      try:
        selection = resolve_model(user_id, provider, tier,
                                  explicit_model)
      except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
      if selection.provider != "anthropic":
        raise HTTPException(status_code=400,
                            detail="Handwriting support is not yet available for this provider")
      ai = ai_helper.AI_Helper__Anthropic()
      timing_start = perf_counter()
      timing_provider = "anthropic"
      timing_model = selection.model_id
      response, usage = ai.query_ai(
        query, attachments=[("png", image_base64)],
        candidate_models=[selection.model_id],
        max_response_tokens=_HANDWRITING_MAX_RESPONSE_TOKENS)
      transcription = response
      timing_model = usage.get("model", timing_model)
      model_name = f"Anthropic ({usage.get('model', selection.model_id)})"

    try:
      transcription, is_blank, is_effectively_blank, is_relevant = \
        _parse_handwriting_analysis(transcription)
    except ValueError as error:
      log.warning("Invalid handwriting-analysis result from %s for problem %s: %s",
                  model_name, problem_id, error)
      raise HTTPException(status_code=500, detail=str(error)) from error

    # Relevance is intentionally a second, text-only pass. The vision crop
    # includes printed question material, which can distort semantic judgment.
    def query_followup(prompt: str, attachments: list[tuple[str, str]],
                       max_tokens: int, json_output: bool = False) -> str:
      if selected_model == "ollama":
        response_text, _ = ai.query_ai(
          prompt, attachments=attachments, max_response_tokens=max_tokens,
          json_output=json_output)
        log.debug("Ollama handwriting follow-up response for problem %s: %r",
                  problem_id, response_text)
      else:
        response_text, _ = ai.query_ai(
          prompt, attachments=attachments, max_response_tokens=max_tokens,
          candidate_models=[selection.model_id])
      return response_text.strip()

    is_relevant = None
    if transcription and not is_blank:
      metadata_repo = ProblemMetadataRepository()
      try:
        question_text = _get_regenerated_question_text(problem)
        if not question_text:
          question_text = metadata_repo.get_question_text(
            problem.session_id, problem.problem_number)
        if not question_text:
          # This is only a legacy fallback for exams without QR regeneration
          # or saved question text. QR-backed exams use their exact frozen-YAML
          # question and per-student seed above.
          question_text = query_followup(
            "Extract only the printed exam question from this image. Ignore all "
            "student handwriting, answer boxes, and page furniture. Return plain "
            "question text only.",
            [("png", image_base64)], 2000)
          if question_text:
            metadata_repo.upsert_question_text(
              problem.session_id, problem.problem_number, question_text)

        if question_text:
          classification_response = query_followup(
            "Determine whether the student's transcribed response attempts to "
            "answer the exam question. This is a relevance judgment, not a "
            "correctness, completeness, quality, or syntax judgment. Return only "
            "JSON: {\"is_relevant\": true or false}.\n\n"
            "Set is_relevant=true for any recognizable attempt to answer, including "
            "a short answer, a partial answer, an incorrect answer, intermediate "
            "work, equations, notation, pseudocode, identifiers, code fragments, "
            "or code in any language. For programming questions, treat code as "
            "relevant even if it is incomplete, non-compiling, poorly formatted, "
            "or lacks explanation. Do not require the response to match a reference "
            "answer.\n"
            "Set is_relevant=false only when the response is clearly unrelated to "
            "the question, is conversational text, is a name/doodle, or is random "
            "letters/symbols with no plausible connection. Do not mark a response "
            "relevant solely because it contains letters, arrows, or punctuation.\n\n"
            f"Question:\n{question_text}\n\n"
            f"Student response:\n{transcription}",
            [], 256, json_output=True)
          is_relevant = _parse_relevance_classification(classification_response)
      except Exception as error:
        # Preserve a successful transcription even if the optional semantic
        # classifier or one-time question extraction fails.
        log.warning("Text classification failed for problem %s: %s",
                    problem_id, error)

    model_latency.record(
      "handwriting", timing_provider, timing_model,
      (perf_counter() - timing_start) * 1000, "success", timing_server)
    problem_repo.update_transcription(
      problem_id, transcription, model_name, is_blank, None,
      is_relevant)

    return {
      "problem_id": problem_id,
      "transcription": transcription,
      "model": model_name,
      "is_blank": is_blank,
      "is_effectively_blank": None,
      "is_relevant": is_relevant,
    }
  except HTTPException:
    if timing_start is not None:
      model_latency.record(
        "handwriting", timing_provider, timing_model,
        (perf_counter() - timing_start) * 1000, "error", timing_server)
    raise
  except Exception as e:
    if timing_start is not None:
      model_latency.record(
        "handwriting", timing_provider, timing_model,
        (perf_counter() - timing_start) * 1000, "error", timing_server)
    import traceback
    log.error(f"Transcription failed: {traceback.format_exc()}")
    raise HTTPException(status_code=500,
                        detail=f"Transcription failed: {str(e)}")


def _batch_decipher_handwriting(job_id: str, problem_ids: list[int], model: str,
                                user_id: int, overwrite: bool) -> None:
  """Fill a batch while exposing progress and continuing after failures."""
  with _handwriting_jobs_lock:
    _handwriting_jobs[job_id]["status"] = "running"
  for problem_id in problem_ids:
    try:
      result = _decipher_handwriting(problem_id, model, user_id,
                                     skip_cached=not overwrite)
    except Exception as error:
      log.warning("Batch transcription failed for problem %s: %s", problem_id, error)
      with _handwriting_jobs_lock:
        _handwriting_jobs[job_id]["failed"] += 1
    else:
      with _handwriting_jobs_lock:
        _handwriting_jobs[job_id]["succeeded"] += 1
        if result.get("is_blank"):
          _handwriting_jobs[job_id]["reported_blank"] += 1
        if result.get("is_relevant"):
          _handwriting_jobs[job_id]["reported_relevant"] += 1
        elif result.get("is_relevant") is False and \
            not result.get("is_blank"):
          _handwriting_jobs[job_id]["reported_irrelevant"] += 1
        # Blank responses intentionally skip the text-only classifier. Count
        # only nonblank responses that should have received that second pass.
        if not result.get("is_blank") and result.get("is_relevant") is None:
          _handwriting_jobs[job_id]["classification_incomplete"] += 1
    finally:
      with _handwriting_jobs_lock:
        job = _handwriting_jobs[job_id]
        job["processed"] += 1
        elapsed_seconds = perf_counter() - job["started_at"]
        job["average_item_seconds"] = elapsed_seconds / job["processed"]
        job["estimated_remaining_seconds"] = (
          job["average_item_seconds"] * (job["total"] - job["processed"])
        )
  with _handwriting_jobs_lock:
    job = _handwriting_jobs[job_id]
    job["status"] = "completed"
    job["estimated_remaining_seconds"] = 0


@router.post("/{problem_id}/decipher")
async def decipher_handwriting(
  problem_id: int,
  model: str = "default",
  current_user: dict = Depends(get_current_user)
):
  """Use AI to transcribe handwritten text from one accessible problem image."""
  problem = ProblemRepository().get_by_id(problem_id)
  if not problem:
    raise HTTPException(status_code=404, detail="Problem not found")
  if current_user["role"] != "instructor":
    from ..repositories.session_assignment_repository import SessionAssignmentRepository
    if not SessionAssignmentRepository().is_user_assigned(
        problem.session_id, current_user["user_id"]):
      raise HTTPException(status_code=403, detail="You do not have access to this grading session")
  # The normal/default action reads a pre-populated result. Explicit model
  # choices remain an intentional re-analysis and replace the cached value.
  return _decipher_handwriting(
    problem_id, model, current_user["user_id"],
    skip_cached=(model or "default").strip().lower() == "default")


@router.post("/session/{session_id}/{problem_number}/decipher-all")
async def decipher_all_handwriting(
  session_id: int,
  problem_number: int,
  background_tasks: BackgroundTasks,
  model: str = "default",
  overwrite: bool = False,
  current_user: dict = Depends(require_session_access())
):
  """Queue handwriting analysis for all submissions of a problem."""
  problem_ids = [
    problem.id for problem in ProblemRepository().get_for_handwriting_analysis(
      session_id, problem_number, overwrite)
  ]
  job_id = str(uuid4())
  with _handwriting_jobs_lock:
    _handwriting_jobs[job_id] = {
      "session_id": session_id,
      "problem_number": problem_number,
      "status": "queued",
      "started_at": perf_counter(),
      "total": len(problem_ids),
      "processed": 0,
      "succeeded": 0,
      "failed": 0,
      "reported_blank": 0,
      "reported_irrelevant": 0,
      "reported_relevant": 0,
      "classification_incomplete": 0,
      "average_item_seconds": None,
      "estimated_remaining_seconds": None,
    }
  background_tasks.add_task(
    _batch_decipher_handwriting, job_id, problem_ids, model,
    current_user["user_id"], overwrite)
  return {"job_id": job_id, "status": "queued", "queued": len(problem_ids)}


@router.get("/session/{session_id}/{problem_number}/decipher-all/{job_id}")
async def get_decipher_all_status(
  session_id: int,
  problem_number: int,
  job_id: str,
  current_user: dict = Depends(require_session_access())
):
  """Return progress for an in-process handwriting-analysis batch."""
  with _handwriting_jobs_lock:
    job = _handwriting_jobs.get(job_id)
    if not job or job["session_id"] != session_id or \
        job["problem_number"] != problem_number:
      raise HTTPException(status_code=404, detail="Handwriting analysis job not found")
    return dict(job)


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
  if not _is_grouping_mode(grading_mode):
    raise HTTPException(
      status_code=400,
      detail="Triage is only available when grading mode is subjective or tag"
    )

  valid_bucket_ids = {bucket.get("id") for bucket in buckets}
  bucket_id: Optional[str] = None
  if grading_mode == "subjective":
    bucket_id = (request.bucket_id or "").strip()
    if not bucket_id:
      raise HTTPException(
        status_code=400,
        detail="bucket_id is required in subjective mode"
      )
    if bucket_id not in valid_bucket_ids:
      raise HTTPException(
        status_code=400,
        detail=f"Unknown bucket id '{bucket_id}' for this problem"
      )
  else:
    normalized_tag_ids = _normalize_tag_ids(request.tag_ids or [])
    if not normalized_tag_ids:
      raise HTTPException(
        status_code=400,
        detail="Select at least one tag in tag mode"
      )
    unknown_tag_ids = sorted(
      [tag_id for tag_id in normalized_tag_ids if tag_id not in valid_bucket_ids]
    )
    if unknown_tag_ids:
      raise HTTPException(
        status_code=400,
        detail=f"Unknown tag id(s) for this problem: {', '.join(unknown_tag_ids)}"
      )
    bucket_id = _canonical_tag_signature(normalized_tag_ids)
    if not bucket_id:
      raise HTTPException(
        status_code=400,
        detail="Select at least one tag in tag mode"
      )

  existing_triage = triage_repo.get_for_problem(problem.id)
  previous_bucket_id = existing_triage["bucket_id"] if existing_triage else None

  triage_repo.upsert(
    problem_id=problem.id,
    session_id=problem.session_id,
    problem_number=problem.problem_number,
    bucket_id=bucket_id,
    notes=request.notes
  )

  counts = problem_repo.get_counts_for_problem_number(problem.session_id, problem.problem_number)
  triaged_count = triage_repo.count_ungraded_for_problem_number(
    problem.session_id, problem.problem_number
  )
  finalized_count = triage_repo.count_graded_for_problem_number(
    problem.session_id, problem.problem_number
  )
  untriaged_count = max((counts["total"] - counts["graded"]) - triaged_count, 0)
  bucket_usage = triage_repo.get_bucket_counts(problem.session_id, problem.problem_number)

  response_payload = {
    "status": "triaged",
    "problem_id": problem_id,
    "problem_number": problem.problem_number,
    "previous_bucket_id": previous_bucket_id,
    "bucket_id": bucket_id,
    "bucket_usage": bucket_usage,
    "triaged_count": triaged_count,
    "finalized_count": finalized_count,
    "untriaged_count": untriaged_count
  }
  if grading_mode == "tag":
    response_payload["tag_ids"] = _tag_ids_from_signature(bucket_id)
  return response_payload


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

  existing_triage = triage_repo.get_for_problem(problem.id)
  cleared_bucket_id = existing_triage["bucket_id"] if existing_triage else None

  triage_repo.clear(problem_id)
  counts = problem_repo.get_counts_for_problem_number(problem.session_id, problem.problem_number)
  triaged_count = triage_repo.count_ungraded_for_problem_number(
    problem.session_id, problem.problem_number
  )
  finalized_count = triage_repo.count_graded_for_problem_number(
    problem.session_id, problem.problem_number
  )
  untriaged_count = max((counts["total"] - counts["graded"]) - triaged_count, 0)
  bucket_usage = triage_repo.get_bucket_counts(problem.session_id, problem.problem_number)

  return {
    "status": "cleared",
    "problem_id": problem_id,
    "problem_number": problem.problem_number,
    "cleared_bucket_id": cleared_bucket_id,
    "bucket_usage": bucket_usage,
    "triaged_count": triaged_count,
    "finalized_count": finalized_count,
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
    if not qr_matches_problem_number(qr_data, problem.problem_number):
      raise HTTPException(
        status_code=400,
        detail=(
          f"Scanned QR payload question number {qr_data.get('question_number')} "
          f"does not match problem number {problem.problem_number}"
        )
      )
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
