"""
In-memory workflow lock helpers for per-session background jobs.
"""
from threading import Lock
from typing import Tuple

_LOCK = Lock()
_ACTIVE_WORKFLOWS: set[Tuple[str, int]] = set()


def acquire(workflow: str, session_id: int) -> bool:
  """Try to acquire a workflow lock for a session."""
  key = (workflow, session_id)
  with _LOCK:
    if key in _ACTIVE_WORKFLOWS:
      return False
    _ACTIVE_WORKFLOWS.add(key)
    return True


def release(workflow: str, session_id: int) -> None:
  """Release a workflow lock for a session."""
  key = (workflow, session_id)
  with _LOCK:
    _ACTIVE_WORKFLOWS.discard(key)


def is_active(workflow: str, session_id: int) -> bool:
  """Check whether a workflow is active for a session."""
  key = (workflow, session_id)
  with _LOCK:
    return key in _ACTIVE_WORKFLOWS

