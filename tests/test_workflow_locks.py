"""
Unit tests for in-memory workflow lock coordination.
"""
from concurrent.futures import ThreadPoolExecutor

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


def test_acquire_with_conflicts_blocks_conflicting_workflow():
  """Atomic conflict-aware acquire should refuse conflicting active workflow."""
  session_id = 303
  try:
    acquired, conflict = workflow_locks.acquire_with_conflicts(
      "finalize", session_id, ("autograde", ))
    assert acquired is True
    assert conflict is None

    acquired, conflict = workflow_locks.acquire_with_conflicts(
      "autograde", session_id, ("finalize", ))
    assert acquired is False
    assert conflict == "finalize"
  finally:
    workflow_locks.release("finalize", session_id)
    workflow_locks.release("autograde", session_id)


def test_acquire_with_conflicts_is_atomic_under_race():
  """Concurrent conflicting acquisitions should allow only one winner."""
  session_id = 404

  try:
    def try_finalize():
      return workflow_locks.acquire_with_conflicts(
        "finalize", session_id, ("autograde", ))[0]

    def try_autograde():
      return workflow_locks.acquire_with_conflicts(
        "autograde", session_id, ("finalize", ))[0]

    with ThreadPoolExecutor(max_workers=2) as executor:
      finalize_future = executor.submit(try_finalize)
      autograde_future = executor.submit(try_autograde)
      finalize_ok = finalize_future.result()
      autograde_ok = autograde_future.result()

    assert sorted([finalize_ok, autograde_ok]) == [False, True]
  finally:
    workflow_locks.release("finalize", session_id)
    workflow_locks.release("autograde", session_id)
