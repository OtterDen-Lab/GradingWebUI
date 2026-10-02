from grading_web_ui.web_api import database
from grading_web_ui.web_api.services import ollama_settings
from grading_web_ui.ai_helper import AI_Helper__Ollama
import pytest


def test_ollama_server_lists_installed_models(tmp_path, monkeypatch):
  monkeypatch.setenv("GRADING_DB_PATH", str(tmp_path / "grading.db"))
  monkeypatch.setenv("GRADING_DB_CREATE_MIGRATION_BACKUP", "false")
  database.init_database()
  with database.get_db_connection() as conn:
    conn.execute("INSERT INTO users (username, password_hash, role) VALUES ('admin', 'hash', 'instructor')")
    user_id = conn.execute("SELECT id FROM users WHERE username = 'admin'").fetchone()[0]

  server = ollama_settings.save_server("Local GPU", "https://ollama.example.test/", user_id)

  class Response:
    def raise_for_status(self):
      pass

    def json(self):
      return {"models": [{"name": "qwen3-vl:30b", "size": 123}]}

  monkeypatch.setattr(ollama_settings.httpx, "get", lambda *args, **kwargs: Response())
  assert ollama_settings.list_models(server["id"]) == [{
    "id": "qwen3-vl:30b", "display_name": "qwen3-vl:30b", "size": 123, "modified_at": None
  }]

  saved = ollama_settings.set_active_model(server["id"], "qwen3-vl:30b", user_id)
  assert saved["active_model"] == "qwen3-vl:30b"


def test_ollama_helper_disables_thinking_and_reports_missing_final_text(monkeypatch):
  captured = {}

  class Response:
    def raise_for_status(self):
      pass

    def json(self):
      return {"message": {"content": "", "thinking": "reasoning"},
              "eval_count": 1000, "done_reason": "length"}

  def post(*args, **kwargs):
    captured.update(kwargs["json"])
    return Response()

  monkeypatch.setattr("grading_web_ui.ai_helper.httpx.post", post)
  with pytest.raises(RuntimeError, match="no final text.*1000.*length"):
    AI_Helper__Ollama("https://ollama.example.test", "qwen3-vl:30b").query_ai("read", [])
  assert captured["think"] is False
