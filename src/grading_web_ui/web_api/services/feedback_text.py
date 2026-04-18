"""
Helpers for composing stored feedback text.
"""
from typing import Optional

GENERAL_FEEDBACK_HEADER = "General feedback:"
SPECIFIC_FEEDBACK_HEADER = "Response-specific feedback:"


def _split_structured_feedback(
  feedback_text: Optional[str]
) -> tuple[Optional[str], Optional[str]] | None:
  """Parse the canonical merged feedback structure, if present."""
  text = (feedback_text or "").strip()
  if not text.startswith(f"{GENERAL_FEEDBACK_HEADER}\n"):
    return None

  remainder = text[len(GENERAL_FEEDBACK_HEADER) + 1:]
  structured_marker = f"\n\n{SPECIFIC_FEEDBACK_HEADER}\n"
  if structured_marker in remainder:
    general_text, specific_text = remainder.split(structured_marker, 1)
    return (
      general_text.strip() or None,
      specific_text.strip() or None,
    )

  return (remainder.strip() or None, None)


def join_feedback_parts(*parts: Optional[str]) -> Optional[str]:
  """Join non-empty feedback fragments with paragraph spacing."""
  normalized = [(part or "").strip() for part in parts if (part or "").strip()]
  if not normalized:
    return None
  return "\n\n".join(normalized)


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

  structured = _split_structured_feedback(specific)
  if structured is not None:
    _, structured_specific = structured
    if structured_specific is None:
      return f"{GENERAL_FEEDBACK_HEADER}\n{general}"
    return (
      f"{GENERAL_FEEDBACK_HEADER}\n{general}\n\n"
      f"{SPECIFIC_FEEDBACK_HEADER}\n{structured_specific}"
    )

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


def extract_response_specific_feedback(
  general_feedback: Optional[str],
  combined_feedback: Optional[str]
) -> Optional[str]:
  """Strip a structured general-feedback section from stored feedback."""
  general = (general_feedback or "").strip()
  combined = (combined_feedback or "").strip()

  if not combined:
    return None
  if not general:
    return combined

  structured = _split_structured_feedback(combined)
  if structured is not None:
    _, structured_specific = structured
    return structured_specific

  general_section = f"{GENERAL_FEEDBACK_HEADER}\n{general}"
  if combined == general_section:
    return None

  structured_prefix = (
    f"{general_section}\n\n"
    f"{SPECIFIC_FEEDBACK_HEADER}\n"
  )
  if combined.startswith(structured_prefix):
    return combined[len(structured_prefix):].strip() or None

  # Preserve freeform text when it does not match the canonical structured
  # format. This keeps legacy manual edits intact instead of guessing.
  return combined
