"""Cross-session, point-bucketed grading outcome comparisons."""
import statistics

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth import get_current_user
from ..database import get_db_connection

router = APIRouter()
_BINS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]


class PointBucket(BaseModel):
  name: str = Field(min_length=1, max_length=80)
  points: list[float] = Field(min_length=1, max_length=20)


class SessionComparisonRequest(BaseModel):
  session_ids: list[int] = Field(min_length=1, max_length=20)
  buckets: list[PointBucket] = Field(min_length=1, max_length=20)


def _assert_session_access(session_id: int, current_user: dict) -> None:
  if current_user["role"] == "instructor":
    return
  from ..repositories.session_assignment_repository import SessionAssignmentRepository
  if not SessionAssignmentRepository().is_user_assigned(
      session_id, current_user["user_id"]):
    raise HTTPException(status_code=403, detail="You do not have access to this grading session")


def _session_label(row: dict) -> str:
  import json
  try:
    metadata = json.loads(row.get("metadata") or "{}")
  except (TypeError, ValueError):
    metadata = {}
  return metadata.get("session_name") or row.get("assignment_name") or f"Session {row['id']}"


@router.post("")
async def compare_sessions(request: SessionComparisonRequest,
                           current_user: dict = Depends(get_current_user)):
  # A stable numeric column order makes cross-semester comparisons readable
  # regardless of the order in which sessions were selected in the browser.
  session_ids = sorted(set(request.session_ids))
  for session_id in session_ids:
    _assert_session_access(session_id, current_user)

  placeholders = ", ".join("?" for _ in session_ids)
  with get_db_connection() as conn:
    session_rows = [dict(row) for row in conn.execute(f"""
      SELECT id, assignment_name, metadata FROM grading_sessions
      WHERE id IN ({placeholders})
    """, session_ids).fetchall()]
    if len(session_rows) != len(session_ids):
      raise HTTPException(status_code=404, detail="One or more sessions were not found")
    response_rows = [dict(row) for row in conn.execute(f"""
      SELECT session_id, problem_number, score, max_points, graded, is_blank
      FROM problems WHERE session_id IN ({placeholders})
    """, session_ids).fetchall()]

  sessions = {row["id"]: {"id": row["id"], "name": _session_label(row)}
              for row in session_rows}
  bucket_specs = [("All questions", None)]
  for bucket in request.buckets:
    points = {round(point, 6) for point in bucket.points if point > 0}
    if not points:
      raise HTTPException(status_code=400,
                          detail=f"Bucket '{bucket.name}' needs a positive point value")
    bucket_specs.append((bucket.name, points))

  output = []
  for bucket_name, point_values in bucket_specs:
    for session_id in session_ids:
      rows = [row for row in response_rows if row["session_id"] == session_id and
              row["max_points"] is not None and float(row["max_points"]) > 0 and
              (point_values is None or round(float(row["max_points"]), 6) in point_values)]
      normalized = []
      for row in rows:
        if row["score"] is not None and float(row["max_points"]) > 0:
          normalized.append(max(0.0, min(1.0, float(row["score"]) / float(row["max_points"]))))
      histogram = []
      for lower, upper in _BINS:
        count = sum(lower / 100 <= value < upper / 100 for value in normalized)
        histogram.append({"label": f"{lower}–{100 if upper == 101 else upper}%",
                          "count": count,
                          "percentage": (count / len(normalized) * 100) if normalized else 0})
      output.append({
        "session": sessions[session_id],
        "bucket": bucket_name,
        "point_values": sorted(point_values) if point_values is not None else [],
        "question_numbers": sorted({row["problem_number"] for row in rows}),
        "response_count": len(rows),
        "scored_count": len(normalized),
        "mean_normalized": statistics.mean(normalized) if normalized else None,
        "stddev_normalized": statistics.pstdev(normalized) if len(normalized) > 1 else 0 if normalized else None,
        "blank_count": sum(bool(row["is_blank"]) for row in rows),
        "blank_percentage": (sum(bool(row["is_blank"]) for row in rows) / len(rows) * 100) if rows else None,
        "distribution": histogram,
      })
  return {"rows": output}
