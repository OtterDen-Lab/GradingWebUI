from pathlib import Path

from grading_web_ui.web_api.services.finalizer import FinalizationService


def _service() -> FinalizationService:
  return FinalizationService(
    session_id=1,
    temp_dir=Path("."),
    stream_id="test",
    event_loop=None,
  )


def test_strip_auto_generated_explanation_keeps_normal_feedback():
  service = _service()
  feedback = "Nice work on setup and arithmetic."

  assert service._strip_auto_generated_explanation(feedback) == feedback


def test_strip_auto_generated_explanation_removes_appended_block():
  service = _service()
  feedback = (
    "Please show one more intermediate step.\n\n---\n\n"
    "Note: The explanation below is automatically generated and might not be correct.\n\n"
    "Auto explanation text..."
  )

  assert (
    service._strip_auto_generated_explanation(feedback)
    == "Please show one more intermediate step."
  )


def test_strip_auto_generated_explanation_returns_empty_when_only_auto_block():
  service = _service()
  feedback = (
    "Note: The explanation below is automatically generated and might not be correct.\n\n"
    "Auto explanation text..."
  )

  assert service._strip_auto_generated_explanation(feedback) == ""
