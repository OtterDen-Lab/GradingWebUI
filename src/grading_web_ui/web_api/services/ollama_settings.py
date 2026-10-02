"""Optional Ollama server administration; Ollama itself owns model installation."""
from typing import Optional
from urllib.parse import urlparse
import logging

import httpx

from ..database import get_db_connection

log = logging.getLogger(__name__)
_LIST_TIMEOUT_SECONDS = 15.0
_PULL_TIMEOUT_SECONDS = 60.0 * 60.0


def _normalise_url(base_url: str) -> str:
  value = (base_url or "").strip().rstrip("/")
  parsed = urlparse(value)
  if parsed.scheme not in ("http", "https") or not parsed.netloc:
    raise ValueError("Ollama server URL must be an absolute http(s) URL")
  if parsed.username or parsed.password or parsed.query or parsed.fragment:
    raise ValueError("Ollama server URL must not contain credentials, query, or fragment")
  return value


def list_servers() -> list[dict]:
  with get_db_connection() as conn:
    return [dict(row) for row in conn.execute(
      "SELECT id, name, base_url, active_model, is_enabled, updated_at FROM ollama_servers ORDER BY name")]


def get_server(server_id: int) -> Optional[dict]:
  with get_db_connection() as conn:
    row = conn.execute("SELECT * FROM ollama_servers WHERE id = ?", (server_id,)).fetchone()
    return dict(row) if row else None


def get_active_server() -> Optional[dict]:
  """Return the first configured server with an administrator-selected model."""
  with get_db_connection() as conn:
    row = conn.execute("""SELECT * FROM ollama_servers
      WHERE is_enabled = 1 AND active_model IS NOT NULL
      ORDER BY updated_at DESC, id DESC LIMIT 1""").fetchone()
    return dict(row) if row else None


def save_server(name: str, base_url: str, updated_by: int,
                server_id: Optional[int] = None) -> dict:
  name = (name or "").strip()
  if not name:
    raise ValueError("Ollama server name is required")
  base_url = _normalise_url(base_url)
  with get_db_connection() as conn:
    if server_id is None:
      cursor = conn.execute("""INSERT INTO ollama_servers (name, base_url, updated_by)
        VALUES (?, ?, ?)""", (name, base_url, updated_by))
      server_id = cursor.lastrowid
    else:
      conn.execute("""UPDATE ollama_servers SET name = ?, base_url = ?,
        updated_by = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?""",
                   (name, base_url, updated_by, server_id))
      if conn.total_changes == 0:
        raise ValueError("Ollama server not found")
  return get_server(server_id)


def set_active_model(server_id: int, model: Optional[str], updated_by: int) -> dict:
  model = (model or "").strip() or None
  with get_db_connection() as conn:
    conn.execute("""UPDATE ollama_servers SET active_model = ?, updated_by = ?,
      updated_at = CURRENT_TIMESTAMP WHERE id = ?""", (model, updated_by, server_id))
    if conn.total_changes == 0:
      raise ValueError("Ollama server not found")
  return get_server(server_id)


def list_models(server_id: int) -> list[dict]:
  server = get_server(server_id)
  if not server:
    raise ValueError("Ollama server not found")
  try:
    response = httpx.get(f"{server['base_url']}/api/tags", timeout=_LIST_TIMEOUT_SECONDS)
    response.raise_for_status()
    models = response.json().get("models", [])
  except (httpx.HTTPError, ValueError) as error:
    raise RuntimeError(f"Could not reach Ollama server '{server['name']}': {error}") from error
  return [{"id": item.get("name"), "display_name": item.get("name"),
           "size": item.get("size"), "modified_at": item.get("modified_at")}
          for item in models if item.get("name")]


def pull_model(server_id: int, model: str) -> None:
  """Request installation. This intentionally does not bundle or manage models."""
  server = get_server(server_id)
  model = (model or "").strip()
  if not server:
    raise ValueError("Ollama server not found")
  if not model:
    raise ValueError("An Ollama model name/tag is required")
  try:
    response = httpx.post(f"{server['base_url']}/api/pull", json={"name": model, "stream": False},
                          timeout=_PULL_TIMEOUT_SECONDS)
    response.raise_for_status()
  except httpx.HTTPError as error:
    raise RuntimeError(f"Ollama could not pull '{model}': {error}") from error
