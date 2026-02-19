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


def acquire_with_conflicts(workflow: str, session_id: int,
                           conflicting_workflows: tuple[str, ...]) -> tuple[bool, str | None]:
  """
  Atomically acquire a workflow lock while checking conflicting workflows.

  Returns:
    (acquired, conflicting_workflow)
    - acquired=True, conflicting_workflow=None when successful
    - acquired=False and conflicting_workflow set when blocked
  """
  key = (workflow, session_id)
  with _LOCK:
    if key in _ACTIVE_WORKFLOWS:
      return False, workflow
    for other in conflicting_workflows:
      if (other, session_id) in _ACTIVE_WORKFLOWS:
        return False, other
    _ACTIVE_WORKFLOWS.add(key)
    return True, None


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
