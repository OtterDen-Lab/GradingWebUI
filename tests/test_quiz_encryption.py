"""
Tests for QuizGenerator encryption key runtime handling.
"""
import os

from grading_web_ui.web_api.services.quiz_encryption import (
  install_quizgenerator_key_provider,
  clear_runtime_encryption_key,
)


def test_quizgenerator_key_provider_does_not_write_env():
  """Patched key provider should avoid setting QUIZ_ENCRYPTION_KEY in environment."""
  install_quizgenerator_key_provider()

  original_env = os.environ.get("QUIZ_ENCRYPTION_KEY")
  os.environ.pop("QUIZ_ENCRYPTION_KEY", None)
  clear_runtime_encryption_key()
  try:
    from QuizGenerator.qrcode_generator import QuestionQRCode
    key = QuestionQRCode.get_encryption_key()
    assert isinstance(key, bytes)
    assert key
    assert os.environ.get("QUIZ_ENCRYPTION_KEY") is None
  finally:
    clear_runtime_encryption_key()
    if original_env is not None:
      os.environ["QUIZ_ENCRYPTION_KEY"] = original_env
