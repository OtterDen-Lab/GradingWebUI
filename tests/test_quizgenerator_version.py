"""Tests for the QuizGenerator release-status check."""
from grading_web_ui.web_api.services import quizgenerator_version


def _clear_cache(monkeypatch):
  monkeypatch.setattr(quizgenerator_version, "_cached_status", None)
  monkeypatch.setattr(quizgenerator_version, "_cached_at", 0.0)


def test_reports_outdated_installed_quizgenerator(monkeypatch):
  _clear_cache(monkeypatch)
  monkeypatch.setattr(
    quizgenerator_version, "installed_quizgenerator_version", lambda: "0.30.0"
  )

  class Response:
    def raise_for_status(self):
      return None

    def json(self):
      return {"info": {"version": "0.31.0"}}

  monkeypatch.setattr(quizgenerator_version.requests, "get", lambda *args, **kwargs: Response())

  status = quizgenerator_version.get_quizgenerator_version_status()

  assert status["installed_version"] == "0.30.0"
  assert status["latest_version"] == "0.31.0"
  assert status["is_latest"] is False
  assert status["check_error"] is None


def test_network_failure_does_not_block_regeneration(monkeypatch):
  _clear_cache(monkeypatch)
  monkeypatch.setattr(
    quizgenerator_version, "installed_quizgenerator_version", lambda: "0.31.0"
  )

  def fail(*args, **kwargs):
    raise quizgenerator_version.requests.ConnectionError("offline")

  monkeypatch.setattr(quizgenerator_version.requests, "get", fail)

  status = quizgenerator_version.get_quizgenerator_version_status()

  assert status["installed_version"] == "0.31.0"
  assert status["is_latest"] is None
  assert status["latest_version"] is None
  assert "offline" in status["check_error"]
