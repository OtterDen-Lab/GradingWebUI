"""
In-memory runtime request metrics.
"""
from collections import defaultdict
from threading import Lock
from time import time
from typing import Dict


class RuntimeMetrics:
  """
  Lightweight in-memory metrics store for basic operational visibility.
  """

  def __init__(self):
    self._lock = Lock()
    self._started_at = time()
    self._totals: Dict[str, int] = defaultdict(int)
    self._status_buckets: Dict[str, int] = defaultdict(int)
    self._route_totals: Dict[str, int] = defaultdict(int)

  def record(self, route: str, status_code: int) -> None:
    """Record one completed request."""
    status_bucket = f"{status_code // 100}xx"
    with self._lock:
      self._totals["requests_total"] += 1
      if status_code >= 500:
        self._totals["requests_5xx_total"] += 1
      self._status_buckets[status_bucket] += 1
      self._route_totals[route] += 1

  def snapshot(self) -> dict:
    """Return current metrics snapshot."""
    with self._lock:
      return {
        "uptime_seconds": int(time() - self._started_at),
        "requests_total": self._totals.get("requests_total", 0),
        "requests_5xx_total": self._totals.get("requests_5xx_total", 0),
        "status_buckets": dict(self._status_buckets),
        "route_totals": dict(self._route_totals),
      }
