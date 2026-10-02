"""Compatibility tests for structured handwriting-analysis results."""

import pytest

from grading_web_ui.web_api.routes.problems import _parse_handwriting_analysis


def test_accepts_full_handwriting_analysis_json():
  assert _parse_handwriting_analysis(
    '{"is_blank": false, "is_effectively_blank": false, '
    '"is_relevant": true, "text": "x = 4"}'
  ) == ("x = 4", False, False, True)


def test_accepts_legacy_two_field_json_from_smaller_models():
  assert _parse_handwriting_analysis(
    '{"is_blank": true, "text": ""}'
  ) == ("", True, False, False)


def test_accepts_embedded_json_and_boolean_strings():
  assert _parse_handwriting_analysis(
    'Result: {"blank": "false", "effectively_blank": "true", '
    '"relevant": "false", "transcription": "scribble"}'
  ) == ("scribble", False, True, False)


def test_rejects_non_json_handwriting_analysis():
  with pytest.raises(ValueError, match="JSON"):
    _parse_handwriting_analysis("This is not JSON")
