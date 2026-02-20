"""
Helpers for composing stored feedback text.
"""
from typing import Optional

GENERAL_FEEDBACK_HEADER = "General feedback:"
SPECIFIC_FEEDBACK_HEADER = "Response-specific feedback:"


def merge_general_feedback(
  general_feedback: Optional[str],
  response_specific_feedback: Optional[str]
) -> Optional[str]:
  """
  Build persisted feedback with a dedicated general-feedback section.

  If general feedback is configured, it is prepended as:
    General feedback:
    <text>

  If response-specific feedback exists, it is appended as:
    Response-specific feedback:
    <text>
  """
  general = (general_feedback or "").strip()
  specific = (response_specific_feedback or "").strip()

  if not general:
    return specific or None

  general_section = f"{GENERAL_FEEDBACK_HEADER}\n{general}"

  if not specific:
    return general_section

  # Already structured with the same general section - avoid duplication.
  if specific.startswith(general_section):
    return specific

  # If the default feedback text is already present in freeform content, keep
  # what the grader entered instead of injecting a duplicate copy.
  if general in specific:
    if specific == general:
      return general_section
    prefix = f"{general}\n\n"
    suffix = f"\n\n{general}"
    if specific.startswith(prefix):
      specific = specific[len(prefix):].strip()
    elif specific.endswith(suffix):
      specific = specific[:-len(suffix)].strip()
    else:
      return specific

    if not specific:
      return general_section

  return (
    f"{general_section}\n\n"
    f"{SPECIFIC_FEEDBACK_HEADER}\n{specific}"
  )
