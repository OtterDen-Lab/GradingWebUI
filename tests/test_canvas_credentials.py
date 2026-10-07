"""Tests for encrypted per-user Canvas API credentials."""
from grading_web_ui.web_api.database import get_db_connection, init_database
from grading_web_ui.web_api.services import canvas_credentials


def test_canvas_key_is_encrypted_and_resolved_per_user(tmp_path, monkeypatch):
  monkeypatch.setenv("GRADING_DB_PATH", str(tmp_path / "grading.db"))
  monkeypatch.setenv("CANVAS_CREDENTIAL_ENCRYPTION_KEY", "test-only-secret")
  monkeypatch.setenv("CANVAS_API_URL", "https://canvas.example.test")
  init_database()
  with get_db_connection() as conn:
    conn.execute("""
      INSERT INTO users (username, password_hash, role)
      VALUES ('canvas-user', 'not-used', 'instructor')
    """)
    user_id = conn.execute(
      "SELECT id FROM users WHERE username = 'canvas-user'").fetchone()["id"]

  canvas_credentials.save_credential(user_id, "development", "secret-canvas-key")

  with get_db_connection() as conn:
    stored = conn.execute("""
      SELECT encrypted_api_key FROM user_canvas_credentials
      WHERE user_id = ? AND environment = 'development'
    """, (user_id,)).fetchone()["encrypted_api_key"]
  assert stored != "secret-canvas-key"
  assert canvas_credentials.get_credential_status(user_id)["development"]["configured"]

  captured = {}

  class FakeCanvasInterface:
    def __init__(self, **kwargs):
      captured.update(kwargs)

  monkeypatch.setattr(canvas_credentials, "CanvasInterface", FakeCanvasInterface)
  canvas_credentials.create_canvas_interface(user_id)

  assert captured["canvas_url"] == "https://canvas.example.test"
  assert captured["canvas_key"] == "secret-canvas-key"
  assert canvas_credentials.delete_credential(user_id, "development")
  assert not canvas_credentials.get_credential_status(user_id)["development"]["configured"]
