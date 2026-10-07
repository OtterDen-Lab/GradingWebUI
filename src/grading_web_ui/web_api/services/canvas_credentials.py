"""Encrypted per-user Canvas credential storage and client creation."""
import base64
import hashlib
import os
from dataclasses import dataclass

from cryptography.fernet import Fernet, InvalidToken
from lms_interface.canvas_interface import CanvasInterface

from ..database import get_db_connection


class CanvasCredentialsError(ValueError):
  """Raised when a user's Canvas credentials cannot be used."""


def environment_name(use_prod: bool) -> str:
  return "production" if use_prod else "development"


def _fernet() -> Fernet:
  secret = os.getenv("CANVAS_CREDENTIAL_ENCRYPTION_KEY", "").strip()
  if not secret:
    raise CanvasCredentialsError(
      "Canvas credential storage is unavailable: set CANVAS_CREDENTIAL_ENCRYPTION_KEY on the server.")
  key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
  return Fernet(key)


def get_credential_status(user_id: int) -> dict:
  with get_db_connection() as conn:
    rows = conn.execute("""
      SELECT environment, updated_at FROM user_canvas_credentials
      WHERE user_id = ?
    """, (user_id,)).fetchall()
  configured = {row["environment"]: row["updated_at"] for row in rows}
  return {
    environment: {
      "configured": environment in configured,
      "updated_at": configured.get(environment),
    }
    for environment in ("development", "production")
  }


def save_credential(user_id: int, environment: str, api_key: str) -> None:
  if environment not in ("development", "production"):
    raise CanvasCredentialsError("Canvas environment must be development or production.")
  api_key = api_key.strip()
  if not api_key:
    raise CanvasCredentialsError("Canvas API key is required.")
  encrypted_key = _fernet().encrypt(api_key.encode()).decode()
  with get_db_connection() as conn:
    conn.execute("""
      INSERT INTO user_canvas_credentials (user_id, environment, encrypted_api_key)
      VALUES (?, ?, ?)
      ON CONFLICT(user_id, environment) DO UPDATE SET
        encrypted_api_key = excluded.encrypted_api_key,
        updated_at = CURRENT_TIMESTAMP
    """, (user_id, environment, encrypted_key))


def delete_credential(user_id: int, environment: str) -> bool:
  if environment not in ("development", "production"):
    raise CanvasCredentialsError("Canvas environment must be development or production.")
  with get_db_connection() as conn:
    result = conn.execute("""
      DELETE FROM user_canvas_credentials WHERE user_id = ? AND environment = ?
    """, (user_id, environment))
  return result.rowcount > 0


def _get_api_key(user_id: int, environment: str) -> str:
  with get_db_connection() as conn:
    row = conn.execute("""
      SELECT encrypted_api_key FROM user_canvas_credentials
      WHERE user_id = ? AND environment = ?
    """, (user_id, environment)).fetchone()
  if not row:
    raise CanvasCredentialsError(
      f"No {environment} Canvas API key is configured for this user.")
  try:
    return _fernet().decrypt(row["encrypted_api_key"].encode()).decode()
  except (InvalidToken, UnicodeDecodeError) as exc:
    raise CanvasCredentialsError(
      "The saved Canvas API key cannot be decrypted. Ask an administrator to verify CANVAS_CREDENTIAL_ENCRYPTION_KEY.") from exc


def create_canvas_interface(user_id: int, *, use_prod: bool = False,
                            privacy_mode: str | None = None,
                            reveal_identity: bool = False) -> CanvasInterface:
  """Create a Canvas client using only the requesting user's saved key."""
  environment = environment_name(use_prod)
  url_var = "CANVAS_API_URL_PROD" if use_prod else "CANVAS_API_URL"
  canvas_url = os.getenv(url_var, "").strip()
  if not canvas_url and use_prod:
    canvas_url = os.getenv("CANVAS_API_URL_prod", "").strip()
  if not canvas_url:
    raise CanvasCredentialsError(
      f"Canvas {environment} URL is not configured on the server ({url_var}).")
  return CanvasInterface(
    prod=use_prod,
    canvas_url=canvas_url,
    canvas_key=_get_api_key(user_id, environment),
    privacy_mode=privacy_mode,
    reveal_identity=reveal_identity,
  )
