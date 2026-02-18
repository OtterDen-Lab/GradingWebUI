"""
Unit tests for in-memory workflow lock coordination.
"""
from grading_web_ui.web_api import workflow_locks


def test_workflow_lock_acquire_release_cycle():
  """Same workflow/session cannot be acquired twice until released."""
  workflow = "autograde"
  session_id = 101

  try:
    assert workflow_locks.acquire(workflow, session_id) is True
    assert workflow_locks.is_active(workflow, session_id) is True
    assert workflow_locks.acquire(workflow, session_id) is False
  finally:
    workflow_locks.release(workflow, session_id)

  assert workflow_locks.is_active(workflow, session_id) is False


def test_workflow_lock_allows_different_workflows_per_session():
  """Different workflow names should not block each other."""
  session_id = 202

  try:
    assert workflow_locks.acquire("autograde", session_id) is True
    assert workflow_locks.acquire("finalize", session_id) is True
  finally:
    workflow_locks.release("autograde", session_id)
    workflow_locks.release("finalize", session_id)
