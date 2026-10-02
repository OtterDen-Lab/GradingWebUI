"""Persistence and percentile summaries for AI model response time."""
import math
from collections import defaultdict
from typing import Optional

from ..database import get_db_connection


def record(feature: str, provider: str, model_id: str, duration_ms: float,
           outcome: str, server_name: Optional[str] = None) -> None:
  """Best-effort telemetry must never affect a grading request."""
  try:
    with get_db_connection() as conn:
      conn.execute("""INSERT INTO ai_model_latency_events
        (feature, provider, model_id, server_name, duration_ms, outcome)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (feature, provider, model_id, server_name, round(duration_ms, 2), outcome))
  except Exception:
    # Metrics are observability only; a locked or unavailable metrics DB must
    # not make model transcription fail.
    pass


def handwriting_summary() -> list[dict]:
  """Return all-time success latency percentiles plus failure counts."""
  with get_db_connection() as conn:
    rows = conn.execute("""SELECT provider, model_id, server_name, duration_ms, outcome
      FROM ai_model_latency_events WHERE feature = 'handwriting'""").fetchall()
  groups = defaultdict(lambda: {"durations": [], "failures": 0})
  for row in rows:
    key = (row["provider"], row["model_id"], row["server_name"])
    if row["outcome"] == "success":
      groups[key]["durations"].append(row["duration_ms"])
    else:
      groups[key]["failures"] += 1
  result = []
  for (provider, model_id, server_name), values in groups.items():
    durations = sorted(values["durations"])
    count = len(durations)
    result.append({
      "provider": provider,
      "model_id": model_id,
      "server_name": server_name,
      "samples": count,
      "failures": values["failures"],
      "p50_ms": _percentile(durations, 0.50),
      "p90_ms": _percentile(durations, 0.90),
    })
  return sorted(result, key=lambda item: (item["provider"], item["server_name"] or "", item["model_id"]))


def _percentile(sorted_values: list[float], percentile: float) -> Optional[float]:
  if not sorted_values:
    return None
  return sorted_values[math.ceil(percentile * len(sorted_values)) - 1]
