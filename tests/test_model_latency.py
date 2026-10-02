from grading_web_ui.web_api import database
from grading_web_ui.web_api.services import model_latency


def test_handwriting_latency_summary_groups_model_and_server(tmp_path, monkeypatch):
  monkeypatch.setenv("GRADING_DB_PATH", str(tmp_path / "grading.db"))
  monkeypatch.setenv("GRADING_DB_CREATE_MIGRATION_BACKUP", "false")
  database.init_database()
  for duration in (100, 200, 300, 400, 500):
    model_latency.record("handwriting", "ollama", "qwen:8b", duration,
                         "success", "GPU server")
  model_latency.record("handwriting", "ollama", "qwen:8b", 50,
                       "error", "GPU server")

  assert model_latency.handwriting_summary() == [{
    "provider": "ollama", "model_id": "qwen:8b", "server_name": "GPU server",
    "samples": 5, "failures": 1, "p50_ms": 300.0, "p90_ms": 500.0,
  }]
