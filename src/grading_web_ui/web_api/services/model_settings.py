"""Persistent Bifrost default-model selection."""
from typing import Optional
import sqlite3

from ... import ai_helper
from ..database import get_db_connection

DEFAULT_MODEL_TIER = "medium"  # Reuses the existing migration-safe column.
MAX_TRANSCRIPTION_ADDITIONAL_INSTRUCTIONS = 2000
BUILT_IN_TRANSCRIPTION_INSTRUCTIONS = (
  "Transcribe only the student's handwritten response.\n\n"
  "Do not solve, grade, interpret, correct, summarize, or editorialize.\n"
  "Do not transcribe printed text, including the question, instructions, labels, and point values.\n"
  "Preserve the student's wording, spelling, notation, and line breaks where possible.\n"
  "Use [illegible] only for text that cannot reasonably be determined from the visible handwriting.\n"
  "For is_blank, evaluate only marks made by the student in the response area. "
  "Set is_blank=true when there are no student-created marks that form an answer, "
  "even if the image contains printed question text, instructions, answer boxes, "
  "ruled lines, page labels, QR codes, scan shadows, or other pre-printed material.\n"
  "Do not set is_blank=false merely because printed text or form elements are visible.\n"
  "Before reading the printed question, first look for clearly visible student-created ink. "
  "If there is no such ink, do not infer or answer the question: return the blank marker.\n"
  "Do not decide whether the response is relevant or effectively blank; a separate "
  "text-only analysis performs those classifications.\n\n"
  "Return only a JSON object with exactly these fields: "
  "{\"is_blank\": true or false, \"text\": \"the transcription\"}. "
  "When is_blank is true, set text to exactly \"[blank]\". \"[blank]\" is the "
  "correct answer for an empty response; never replace it with an inferred answer. "
  "Do not use Markdown fences.")


def get_default_model(user_id: Optional[int]) -> dict:
  """Resolve the one Bifrost model used by normal application workflows."""
  try:
    with get_db_connection() as conn:
      if user_id is not None:
        row = conn.execute("""
          SELECT model_id FROM user_model_overrides
          WHERE user_id = ? AND provider = 'bifrost' AND tier = ?
        """, (user_id, DEFAULT_MODEL_TIER)).fetchone()
        if row:
          return {"model_id": row["model_id"], "source": "user"}
      row = conn.execute("""
        SELECT model_id FROM system_model_defaults
        WHERE provider = 'bifrost' AND tier = ?
      """, (DEFAULT_MODEL_TIER,)).fetchone()
      if row:
        return {"model_id": row["model_id"], "source": "system"}
  except sqlite3.OperationalError:
    pass
  return {"model_id": ai_helper.get_default_model(), "source": "built-in"}


def set_default_model(user_id: int, model_id: Optional[str], system: bool = False) -> None:
  """Save one model choice, retaining the existing tiered-table schema."""
  value = (model_id or "").strip()
  if system and not value:
    raise ValueError("A system default model is required")
  with get_db_connection() as conn:
    if not value:
      conn.execute("""DELETE FROM user_model_overrides
        WHERE user_id = ? AND provider = 'bifrost' AND tier = ?""",
                   (user_id, DEFAULT_MODEL_TIER))
    elif system:
      conn.execute("""INSERT INTO system_model_defaults (provider, tier, model_id, updated_by)
        VALUES ('bifrost', ?, ?, ?) ON CONFLICT(provider, tier) DO UPDATE SET
        model_id = excluded.model_id, updated_by = excluded.updated_by,
        updated_at = CURRENT_TIMESTAMP""", (DEFAULT_MODEL_TIER, value, user_id))
    else:
      conn.execute("""INSERT INTO user_model_overrides (user_id, provider, tier, model_id)
        VALUES (?, 'bifrost', ?, ?) ON CONFLICT(user_id, provider, tier) DO UPDATE SET
        model_id = excluded.model_id, updated_at = CURRENT_TIMESTAMP""",
                   (user_id, DEFAULT_MODEL_TIER, value))


def get_transcription_additional_instructions(user_id: int) -> dict:
  try:
    with get_db_connection() as conn:
      row = conn.execute("""SELECT additional_instructions
        FROM user_transcription_prompt_overrides WHERE user_id = ?""", (user_id,)).fetchone()
      if row:
        return {"text": row["additional_instructions"], "source": "user"}
      row = conn.execute("SELECT additional_instructions FROM system_transcription_prompt_settings WHERE id = 1").fetchone()
      if row:
        return {"text": row["additional_instructions"], "source": "system"}
  except sqlite3.OperationalError:
    pass
  return {"text": "", "source": "built-in"}


def set_transcription_additional_instructions(user_id: int, text: Optional[str],
                                              system: bool = False) -> None:
  text = (text or "").strip()
  if len(text) > MAX_TRANSCRIPTION_ADDITIONAL_INSTRUCTIONS:
    raise ValueError(
      f"Additional transcription instructions must be at most {MAX_TRANSCRIPTION_ADDITIONAL_INSTRUCTIONS} characters")
  with get_db_connection() as conn:
    if system:
      conn.execute("""INSERT INTO system_transcription_prompt_settings
        (id, additional_instructions, updated_by) VALUES (1, ?, ?)
        ON CONFLICT(id) DO UPDATE SET additional_instructions = excluded.additional_instructions,
        updated_by = excluded.updated_by, updated_at = CURRENT_TIMESTAMP""", (text, user_id))
    elif not text:
      conn.execute("DELETE FROM user_transcription_prompt_overrides WHERE user_id = ?", (user_id,))
    else:
      conn.execute("""INSERT INTO user_transcription_prompt_overrides
        (user_id, additional_instructions) VALUES (?, ?)
        ON CONFLICT(user_id) DO UPDATE SET additional_instructions = excluded.additional_instructions,
        updated_at = CURRENT_TIMESTAMP""", (user_id, text))


def list_models() -> list[dict]:
  """Return the operator-curated catalog exposed by Bifrost."""
  return ai_helper.AI_Helper__Bifrost().list_models()
