from grading_web_ui.web_api import database
from grading_web_ui.web_api.services import model_settings


def test_default_model_prefers_user_then_system_then_environment(tmp_path, monkeypatch):
  monkeypatch.setenv("GRADING_DB_PATH", str(tmp_path / "test.db"))
  monkeypatch.setenv("GRADING_DB_CREATE_MIGRATION_BACKUP", "false")
  monkeypatch.setenv("BIFROST_DEFAULT_MODEL", "environment-model")
  database.init_database()
  with database.get_db_connection() as conn:
    conn.execute("""INSERT INTO users (username, password_hash, role)
      VALUES ('teacher', 'hash', 'instructor')""")
    user_id = conn.execute(
      "SELECT id FROM users WHERE username = 'teacher'").fetchone()[0]

  assert model_settings.get_default_model(user_id) == {
    "model_id": "environment-model", "source": "built-in"}
  model_settings.set_default_model(user_id, "system-model", system=True)
  assert model_settings.get_default_model(user_id) == {
    "model_id": "system-model", "source": "system"}
  model_settings.set_default_model(user_id, "my-model")
  assert model_settings.get_default_model(user_id) == {
    "model_id": "my-model", "source": "user"}
  model_settings.set_default_model(user_id, None)
  assert model_settings.get_default_model(user_id) == {
    "model_id": "system-model", "source": "system"}
