"""Provider-neutral resolution of AI model defaults and user overrides."""
from dataclasses import dataclass
from typing import Optional
import sqlite3

from ... import ai_helper
from ..database import get_db_connection

MODEL_TIERS = ("small", "medium", "large")
HANDWRITING_TARGETS = ("ollama", *MODEL_TIERS)
MAX_TRANSCRIPTION_ADDITIONAL_INSTRUCTIONS = 2000
BUILT_IN_TRANSCRIPTION_INSTRUCTIONS = (
  "Transcribe only the student's handwritten response.\n\n"
  "Do not solve, grade, interpret, correct, summarize, or editorialize.\n"
  "Do not transcribe printed text, including the question, instructions, labels, and point values.\n"
  "Use the printed question only to determine whether the student's response is relevant; never include it in text.\n"
  "Preserve the student's wording, spelling, notation, and line breaks where possible.\n"
  "Use [illegible] only for text that cannot reasonably be determined from the visible handwriting.\n"
  "For is_blank, evaluate only marks made by the student in the response area. "
  "Set is_blank=true when there are no student-created marks that form an answer, "
  "even if the image contains printed question text, instructions, answer boxes, "
  "ruled lines, page labels, QR codes, scan shadows, or other pre-printed material.\n"
  "Do not set is_blank=false merely because printed text or form elements are visible.\n"
  "Set is_effectively_blank=true (and is_blank=false) for only doodles, stray marks, "
  "a name, an isolated symbol, or other student-created non-answer content.\n"
  "Set is_relevant=true only when the substantive response attempts to answer the printed question.\n\n"
  "Return only a JSON object with exactly these fields: "
  "{\"is_blank\": true or false, \"is_effectively_blank\": true or false, "
  "\"is_relevant\": true or false, \"text\": \"the transcription\"}. "
  "Set text to an empty string when is_blank is true. Do not use Markdown fences.")


@dataclass(frozen=True)
class ModelSelection:
  provider: str
  model_id: str
  tier: str
  source: str


def _normalise_provider(provider: str) -> str:
  value = (provider or "anthropic").strip().lower()
  if value not in ai_helper.MODEL_CONFIG:
    raise ValueError(f"Unsupported AI provider: {provider}")
  return value


def _normalise_tier(tier: str) -> str:
  value = (tier or "medium").strip().lower()
  if value not in MODEL_TIERS:
    raise ValueError(f"Unknown model tier: {tier}")
  return value


def resolve_model(user_id: Optional[int], provider: str = "anthropic",
                  tier: str = "medium", explicit_model: Optional[str] = None) -> ModelSelection:
  """Resolve a model at request time; no grading-session state is consulted."""
  provider = _normalise_provider(provider)
  tier = _normalise_tier(tier)
  if explicit_model:
    return ModelSelection(provider, explicit_model.strip(), tier, "request")

  # The reusable exam processor is also used outside a running web app. In
  # that context a database may not exist, so retain the safe built-in default.
  try:
    with get_db_connection() as conn:
      if user_id is not None:
        row = conn.execute("""
          SELECT model_id FROM user_model_overrides
          WHERE user_id = ? AND provider = ? AND tier = ?
        """, (user_id, provider, tier)).fetchone()
        if row:
          return ModelSelection(provider, row["model_id"], tier, "user")
      row = conn.execute("""
        SELECT model_id FROM system_model_defaults WHERE provider = ? AND tier = ?
      """, (provider, tier)).fetchone()
      if row:
        return ModelSelection(provider, row["model_id"], tier, "system")
  except sqlite3.OperationalError:
    pass
  return ModelSelection(provider, ai_helper.get_model_for_tier(provider, tier), tier, "built-in")


def get_effective_settings(user_id: int, provider: str = "anthropic") -> dict:
  provider = _normalise_provider(provider)
  return {
    tier: vars(resolve_model(user_id, provider, tier))
    for tier in MODEL_TIERS
  }


def get_handwriting_default(user_id: int) -> dict:
  """Resolve the normal Decipher action independently from model-tier defaults."""
  try:
    with get_db_connection() as conn:
      row = conn.execute("SELECT target FROM user_handwriting_overrides WHERE user_id = ?",
                         (user_id,)).fetchone()
      if row:
        return {"target": row["target"], "source": "user"}
      row = conn.execute("SELECT target FROM system_handwriting_settings WHERE id = 1").fetchone()
      if row:
        return {"target": row["target"], "source": "system"}
  except sqlite3.OperationalError:
    pass
  return {"target": "medium", "source": "built-in"}


def set_handwriting_default(user_id: int, target: Optional[str], system: bool = False) -> None:
  if target is not None and target not in HANDWRITING_TARGETS:
    raise ValueError("Handwriting default must be one of: ollama, small, medium, large")
  with get_db_connection() as conn:
    if system:
      if target is None:
        raise ValueError("A system handwriting default is required")
      conn.execute("""INSERT INTO system_handwriting_settings (id, target, updated_by)
        VALUES (1, ?, ?) ON CONFLICT(id) DO UPDATE SET target = excluded.target,
        updated_by = excluded.updated_by, updated_at = CURRENT_TIMESTAMP""", (target, user_id))
    elif target is None:
      conn.execute("DELETE FROM user_handwriting_overrides WHERE user_id = ?", (user_id,))
    else:
      conn.execute("""INSERT INTO user_handwriting_overrides (user_id, target)
        VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET target = excluded.target,
        updated_at = CURRENT_TIMESTAMP""", (user_id, target))


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


def set_system_defaults(provider: str, models: dict, updated_by: int) -> None:
  provider = _normalise_provider(provider)
  _validate_models(models)
  with get_db_connection() as conn:
    for tier, model_id in models.items():
      conn.execute("""
        INSERT INTO system_model_defaults (provider, tier, model_id, updated_by)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(provider, tier) DO UPDATE SET
          model_id = excluded.model_id, updated_by = excluded.updated_by,
          updated_at = CURRENT_TIMESTAMP
      """, (provider, tier, model_id.strip(), updated_by))


def set_user_overrides(user_id: int, provider: str, models: dict) -> None:
  provider = _normalise_provider(provider)
  _validate_models(models)
  with get_db_connection() as conn:
    for tier, model_id in models.items():
      if model_id is None:
        conn.execute("""DELETE FROM user_model_overrides
          WHERE user_id = ? AND provider = ? AND tier = ?""", (user_id, provider, tier))
      else:
        conn.execute("""
          INSERT INTO user_model_overrides (user_id, provider, tier, model_id)
          VALUES (?, ?, ?, ?)
          ON CONFLICT(user_id, provider, tier) DO UPDATE SET
            model_id = excluded.model_id, updated_at = CURRENT_TIMESTAMP
        """, (user_id, provider, tier, model_id.strip()))


def _validate_models(models: dict) -> None:
  if not models or not isinstance(models, dict):
    raise ValueError("models must contain one or more tier/model values")
  for tier, model_id in models.items():
    _normalise_tier(tier)
    if model_id is not None and (not isinstance(model_id, str) or not model_id.strip()):
      raise ValueError(f"A non-empty model ID is required for {tier}")


def list_provider_models(provider: str) -> list[dict]:
  """Return model IDs from a provider. More providers can implement this registry hook."""
  provider = _normalise_provider(provider)
  if provider != "anthropic":
    return []
  client = ai_helper.AI_Helper__Anthropic()._client
  page = client.models.list(limit=100)
  models = getattr(page, "data", page)
  return [{"id": item.id, "display_name": getattr(item, "display_name", item.id)}
          for item in models]
