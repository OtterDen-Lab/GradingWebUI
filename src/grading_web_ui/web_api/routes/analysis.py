"""Persistent, non-destructive handwriting model comparison experiments."""
import json
import logging
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth import get_current_user
from ..database import get_db_connection
from ..repositories import ProblemRepository, SubmissionRepository

router = APIRouter()
log = logging.getLogger(__name__)


class AnalysisRunCreate(BaseModel):
  session_id: int
  problem_number: int
  models: list[str] = Field(min_length=1, max_length=8)
  sample_size: int | None = Field(default=None, ge=1, le=250)


def _assert_session_access(session_id: int, current_user: dict) -> None:
  if current_user["role"] == "instructor":
    return
  from ..repositories.session_assignment_repository import SessionAssignmentRepository
  if not SessionAssignmentRepository().is_user_assigned(
      session_id, current_user["user_id"]):
    raise HTTPException(status_code=403, detail="You do not have access to this grading session")


def _get_run(run_id: str) -> dict:
  with get_db_connection() as conn:
    row = conn.execute("SELECT * FROM handwriting_analysis_runs WHERE id = ?",
                       (run_id,)).fetchone()
  if not row:
    raise HTTPException(status_code=404, detail="Analysis run not found")
  return dict(row)


def _run_analysis(run_id: str, user_id: int) -> None:
  """Use the production analysis path, but never modify cached grading data."""
  run = _get_run(run_id)
  problem_ids = json.loads(run["problem_ids_json"])
  models = json.loads(run["models_json"])
  with get_db_connection() as conn:
    conn.execute("UPDATE handwriting_analysis_runs SET status = 'running', "
                 "started_at = CURRENT_TIMESTAMP WHERE id = ?", (run_id,))

  # Delayed import avoids a route-import cycle while sharing the production
  # crop, prompts, parser, and relevance classifier exactly.
  from .problems import _decipher_handwriting

  # Keep one model resident for its complete response set. Alternating models
  # for each submission defeats Ollama's model cache and makes large-model
  # comparisons disproportionately slow.
  for model_id in models:
    for problem_id in problem_ids:
      # A provider request already in flight cannot be cancelled safely, but a
      # cancellation takes effect before the next response/model pair.
      if _get_run(run_id)["status"] in ("cancelling", "cancelled"):
        with get_db_connection() as conn:
          conn.execute("UPDATE handwriting_analysis_runs SET status = 'cancelled', "
                       "completed_at = CURRENT_TIMESTAMP WHERE id = ?", (run_id,))
        return
      started = perf_counter()
      result = None
      error_message = None
      try:
        result = _decipher_handwriting(
          problem_id, f"ollama:{model_id}", user_id, persist=False)
      except Exception as error:
        error_message = str(getattr(error, "detail", error))
        log.warning("Analysis run %s: %s / %s failed: %s", run_id,
                    problem_id, model_id, error_message)

      with get_db_connection() as conn:
        conn.execute("""
          INSERT OR REPLACE INTO handwriting_analysis_results (
            run_id, problem_id, model_id, transcription, is_blank, is_relevant,
            model_label, duration_ms, raw_transcription_response,
            raw_relevance_response, error
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
          run_id, problem_id, model_id,
          result.get("transcription") if result else None,
          result.get("is_blank") if result else None,
          result.get("is_relevant") if result else None,
          result.get("model") if result else None,
          (perf_counter() - started) * 1000,
          result.get("raw_transcription_response") if result else None,
          result.get("raw_relevance_response") if result else None,
          error_message,
        ))
        conn.execute("""
          UPDATE handwriting_analysis_runs
          SET completed_items = completed_items + 1,
              failed_items = failed_items + ?
          WHERE id = ?
        """, (1 if error_message else 0, run_id))

  with get_db_connection() as conn:
    conn.execute("""
      UPDATE handwriting_analysis_runs
      SET status = CASE WHEN status IN ('cancelling', 'cancelled') THEN 'cancelled' ELSE 'completed' END,
          completed_at = CURRENT_TIMESTAMP
      WHERE id = ?
    """, (run_id,))


@router.get("/sessions/{session_id}/questions")
async def list_questions(session_id: int,
                         current_user: dict = Depends(get_current_user)):
  _assert_session_access(session_id, current_user)
  with get_db_connection() as conn:
    rows = conn.execute("""
      SELECT problem_number, COUNT(*) AS response_count
      FROM problems WHERE session_id = ?
      GROUP BY problem_number ORDER BY problem_number
    """, (session_id,)).fetchall()
  return {"questions": [dict(row) for row in rows]}


@router.post("/runs")
async def create_run(request: AnalysisRunCreate, background_tasks: BackgroundTasks,
                     current_user: dict = Depends(get_current_user)):
  _assert_session_access(request.session_id, current_user)
  models = list(dict.fromkeys(model.strip() for model in request.models if model.strip()))
  if not models:
    raise HTTPException(status_code=400, detail="Select at least one Ollama model")
  problems = ProblemRepository().get_for_handwriting_analysis(
    request.session_id, request.problem_number, overwrite=True)
  if request.sample_size:
    problems = problems[:request.sample_size]
  problem_ids = [problem.id for problem in problems]
  if not problem_ids:
    raise HTTPException(status_code=404, detail="No responses found for that question")

  run_id = str(uuid4())
  with get_db_connection() as conn:
    conn.execute("""
      INSERT INTO handwriting_analysis_runs (
        id, session_id, problem_number, created_by, models_json, problem_ids_json,
        total_items
      ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (run_id, request.session_id, request.problem_number,
          current_user["user_id"], json.dumps(models), json.dumps(problem_ids),
          len(problem_ids) * len(models)))
  background_tasks.add_task(_run_analysis, run_id, current_user["user_id"])
  return {"run_id": run_id, "status": "queued",
          "total_items": len(problem_ids) * len(models)}


@router.get("/runs")
async def list_runs(limit: int = 20,
                    current_user: dict = Depends(get_current_user)):
  """List recent runs visible to the signed-in grader for refresh recovery."""
  limit = min(max(limit, 1), 100)
  with get_db_connection() as conn:
    rows = conn.execute("""
      SELECT * FROM handwriting_analysis_runs
      ORDER BY created_at DESC LIMIT ?
    """, (limit,)).fetchall()
  visible_runs = []
  for row in rows:
    run = dict(row)
    try:
      _assert_session_access(run["session_id"], current_user)
    except HTTPException:
      continue
    run["models"] = json.loads(run.pop("models_json"))
    run.pop("problem_ids_json", None)
    visible_runs.append(run)
  return {"runs": visible_runs}


@router.get("/runs/{run_id}")
async def get_run(run_id: str, current_user: dict = Depends(get_current_user)):
  run = _get_run(run_id)
  _assert_session_access(run["session_id"], current_user)
  with get_db_connection() as conn:
    rows = conn.execute("""
      SELECT r.*, p.is_blank AS heuristic_is_blank
      FROM handwriting_analysis_results r JOIN problems p ON p.id = r.problem_id
      WHERE r.run_id = ? ORDER BY r.problem_id, r.model_id
    """, (run_id,)).fetchall()
  run["models"] = json.loads(run.pop("models_json"))
  run["problem_ids"] = json.loads(run.pop("problem_ids_json"))
  run["results"] = [dict(row) for row in rows]
  return run


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str,
                     current_user: dict = Depends(get_current_user)):
  """Request cancellation; the in-flight provider request is allowed to finish."""
  run = _get_run(run_id)
  _assert_session_access(run["session_id"], current_user)
  if run["status"] in ("completed", "cancelled"):
    return {"run_id": run_id, "status": run["status"]}
  with get_db_connection() as conn:
    conn.execute("UPDATE handwriting_analysis_runs SET status = 'cancelled', "
                 "completed_at = CURRENT_TIMESTAMP WHERE id = ?", (run_id,))
  return {"run_id": run_id, "status": "cancelled"}


@router.get("/runs/{run_id}/problems/{problem_id}/image")
async def get_run_problem_image(run_id: str, problem_id: int,
                                current_user: dict = Depends(get_current_user)):
  run = _get_run(run_id)
  _assert_session_access(run["session_id"], current_user)
  if problem_id not in json.loads(run["problem_ids_json"]):
    raise HTTPException(status_code=404, detail="Problem is not part of this run")
  problem = ProblemRepository().get_by_id(problem_id)
  if not problem:
    raise HTTPException(status_code=404, detail="Problem not found")
  from .problems import get_problem_image_data
  return {"image_data": get_problem_image_data(problem, SubmissionRepository())}
