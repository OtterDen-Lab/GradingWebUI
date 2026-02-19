"""
Compatibility tests for QuizGenerator regeneration wrapper.
"""
import pytest

from grading_web_ui.web_api.services.quiz_regeneration import regenerate_from_encrypted_compat


def test_regenerate_compat_does_not_raise_unexpected_keyword_argument():
  """
  Wrapper should filter unsupported kwargs like yaml_text/image_mode.
  """
  with pytest.raises(Exception) as exc_info:
    regenerate_from_encrypted_compat(
      encrypted_data="invalid-encrypted-payload",
      points=1.0,
      yaml_text="yaml_id: test",
      image_mode="inline")

  assert "unexpected keyword argument" not in str(exc_info.value)

