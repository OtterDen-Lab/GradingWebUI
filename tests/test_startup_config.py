"""
Tests for startup configuration validation.
"""
import pytest

from grading_web_ui.web_api.startup_config import validate_startup_configuration


def test_startup_config_strict_raises_without_canvas_creds(monkeypatch):
  monkeypatch.setenv("GRADING_STRICT_STARTUP_CONFIG", "true")
  monkeypatch.delenv("CANVAS_API_URL", raising=False)
  monkeypatch.delenv("CANVAS_API_KEY", raising=False)
  monkeypatch.delenv("CANVAS_API_URL_PROD", raising=False)
  monkeypatch.delenv("CANVAS_API_KEY_PROD", raising=False)
  monkeypatch.delenv("CANVAS_API_URL_prod", raising=False)
  monkeypatch.delenv("CANVAS_API_KEY_prod", raising=False)

  with pytest.raises(RuntimeError):
    validate_startup_configuration()


def test_startup_config_non_strict_returns_warnings(monkeypatch):
  monkeypatch.setenv("GRADING_STRICT_STARTUP_CONFIG", "false")
  monkeypatch.delenv("CANVAS_API_URL", raising=False)
  monkeypatch.delenv("CANVAS_API_KEY", raising=False)
  monkeypatch.delenv("CANVAS_API_URL_PROD", raising=False)
  monkeypatch.delenv("CANVAS_API_KEY_PROD", raising=False)
  monkeypatch.delenv("CANVAS_API_URL_prod", raising=False)
  monkeypatch.delenv("CANVAS_API_KEY_prod", raising=False)
  monkeypatch.setenv("AUTH_COOKIE_SECURE", "false")

  warnings = validate_startup_configuration()
  assert any("Canvas credentials are missing" in msg for msg in warnings)
  assert any("AUTH_COOKIE_SECURE is false" in msg for msg in warnings)
