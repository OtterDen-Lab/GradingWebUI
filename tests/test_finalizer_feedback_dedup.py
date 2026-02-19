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


def test_strip_auto_generated_explanation_preserves_trailing_manual_notes():
  service = _service()
  feedback = (
    "Please show one more intermediate step.\n\n---\n\n"
    "Note: The explanation below is automatically generated and might not be correct.\n\n"
    "Auto explanation text...\n\n---\n\n"
    "Manual postscript: include units in the final answer."
  )

  assert (
    service._strip_auto_generated_explanation(feedback)
    == "Please show one more intermediate step.\n\n"
    "Manual postscript: include units in the final answer."
  )


def test_render_text_or_html_preformats_ascii_layout_feedback():
  service = _service()
  ascii_feedback = (
    "State machine:\n"
    "+-----+    +-----+\n"
    "|  A  | -> |  B  |\n"
    "+-----+    +-----+"
  )

  rendered = service._render_text_or_html(ascii_feedback,
                                          prefer_preformatted=True)

  assert rendered.startswith("<pre ")
  assert "preformatted-text" in rendered
  assert "|  A  |" in rendered


def test_render_text_or_html_preformats_plain_feedback_when_requested():
  service = _service()
  feedback = "Good setup. Please show one more algebra step."

  rendered = service._render_text_or_html(feedback, prefer_preformatted=True)

  assert rendered.startswith("<pre ")
  assert "preformatted-text" in rendered
  assert "Good setup." in rendered
