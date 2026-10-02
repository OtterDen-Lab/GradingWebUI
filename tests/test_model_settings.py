from grading_web_ui.web_api import database
from grading_web_ui.web_api.services import model_settings


def test_model_settings_precedence_is_user_then_system_then_builtin(tmp_path, monkeypatch):
  monkeypatch.setenv("GRADING_DB_PATH", str(tmp_path / "grading.db"))
  monkeypatch.setenv("GRADING_DB_CREATE_MIGRATION_BACKUP", "false")
  database.init_database()
  with database.get_db_connection() as conn:
    conn.execute("""INSERT INTO users
      (username, password_hash, role) VALUES ('teacher', 'hash', 'instructor')""")
    user_id = conn.execute("SELECT id FROM users WHERE username = 'teacher'").fetchone()[0]

  builtin = model_settings.resolve_model(user_id, "anthropic", "medium")
  assert builtin.source == "built-in"

  model_settings.set_system_defaults("anthropic", {"medium": "system-model"}, user_id)
  system = model_settings.resolve_model(user_id, "anthropic", "medium")
  assert (system.model_id, system.source) == ("system-model", "system")

  model_settings.set_user_overrides(user_id, "anthropic", {"medium": "my-model"})
  user = model_settings.resolve_model(user_id, "anthropic", "medium")
  assert (user.model_id, user.source) == ("my-model", "user")

  model_settings.set_user_overrides(user_id, "anthropic", {"medium": None})
  assert model_settings.resolve_model(user_id, "anthropic", "medium").model_id == "system-model"


def test_handwriting_default_is_independent_of_model_tiers(tmp_path, monkeypatch):
  monkeypatch.setenv("GRADING_DB_PATH", str(tmp_path / "grading.db"))
  monkeypatch.setenv("GRADING_DB_CREATE_MIGRATION_BACKUP", "false")
  database.init_database()
  with database.get_db_connection() as conn:
    conn.execute("INSERT INTO users (username, password_hash, role) VALUES ('teacher', 'hash', 'instructor')")
    user_id = conn.execute("SELECT id FROM users WHERE username = 'teacher'").fetchone()[0]
  assert model_settings.get_handwriting_default(user_id) == {"target": "medium", "source": "built-in"}
  model_settings.set_handwriting_default(user_id, "ollama", system=True)
  assert model_settings.get_handwriting_default(user_id) == {"target": "ollama", "source": "system"}
  model_settings.set_handwriting_default(user_id, "large")
  assert model_settings.get_handwriting_default(user_id) == {"target": "large", "source": "user"}


def test_transcription_additional_instructions_follow_user_then_system(tmp_path, monkeypatch):
  monkeypatch.setenv("GRADING_DB_PATH", str(tmp_path / "grading.db"))
  monkeypatch.setenv("GRADING_DB_CREATE_MIGRATION_BACKUP", "false")
  database.init_database()
  with database.get_db_connection() as conn:
    conn.execute("INSERT INTO users (username, password_hash, role) VALUES ('teacher', 'hash', 'instructor')")
    user_id = conn.execute("SELECT id FROM users WHERE username = 'teacher'").fetchone()[0]
  model_settings.set_transcription_additional_instructions(user_id, "Keep units.", system=True)
  assert model_settings.get_transcription_additional_instructions(user_id) == {
    "text": "Keep units.", "source": "system"}
  model_settings.set_transcription_additional_instructions(user_id, "Keep line breaks.")
  assert model_settings.get_transcription_additional_instructions(user_id) == {
    "text": "Keep line breaks.", "source": "user"}
  model_settings.set_transcription_additional_instructions(user_id, None)
  assert model_settings.get_transcription_additional_instructions(user_id)["source"] == "system"
