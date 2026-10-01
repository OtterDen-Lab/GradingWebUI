"""Provider-neutral resolution of AI model defaults and user overrides."""
from dataclasses import dataclass
from typing import Optional
import sqlite3

from ... import ai_helper
from ..database import get_db_connection

MODEL_TIERS = ("small", "medium", "large")


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
