from grading_web_ui.web_api.services.feedback_text import merge_general_feedback


def test_merge_general_feedback_structures_sections():
  merged = merge_general_feedback(
    "Show units.",
    "Arithmetic error on line 2."
  )
  assert merged == (
    "General feedback:\n"
    "Show units.\n\n"
    "Response-specific feedback:\n"
    "Arithmetic error on line 2."
  )


def test_merge_general_feedback_normalizes_legacy_suffix_copy():
  merged = merge_general_feedback(
    "Show units.",
    "Arithmetic error on line 2.\n\nShow units."
  )
  assert merged == (
    "General feedback:\n"
    "Show units.\n\n"
    "Response-specific feedback:\n"
    "Arithmetic error on line 2."
  )


def test_merge_general_feedback_general_only():
  merged = merge_general_feedback("Show units.", None)
  assert merged == "General feedback:\nShow units."

