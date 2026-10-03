"""Compatibility tests for structured handwriting-analysis results."""

import pytest

from grading_web_ui.web_api.routes.problems import (
  _html_to_plain_text,
  _parse_handwriting_analysis,
  _parse_text_classification,
)


def test_accepts_full_handwriting_analysis_json():
  assert _parse_handwriting_analysis(
    '{"is_blank": false, "is_effectively_blank": false, '
    '"is_relevant": true, "text": "x = 4"}'
  ) == ("x = 4", False, False, True)


def test_accepts_legacy_two_field_json_from_smaller_models():
  assert _parse_handwriting_analysis(
    '{"is_blank": true, "text": "[blank]"}'
  ) == ("[blank]", True, None, None)


def test_accepts_null_text_for_a_blank_response():
  assert _parse_handwriting_analysis(
    '{"is_blank": true, "text": null}'
  ) == ("[blank]", True, None, None)


def test_marks_legacy_nonblank_json_as_classification_incomplete():
  assert _parse_handwriting_analysis(
    '{"is_blank": false, "text": "work shown"}'
  ) == ("work shown", False, None, None)


def test_accepts_embedded_json_and_boolean_strings():
  assert _parse_handwriting_analysis(
    'Result: {"blank": "false", "effectively_blank": "true", '
    '"relevant": "false", "transcription": "scribble"}'
  ) == ("scribble", False, True, False)


def test_rejects_non_json_handwriting_analysis():
  with pytest.raises(ValueError, match="JSON"):
    _parse_handwriting_analysis("This is not JSON")


def test_parses_text_only_classification_json():
  assert _parse_text_classification(
    '{"is_effectively_blank": "true", "is_relevant": false}'
  ) == (True, False)


def test_regenerated_question_html_is_compacted_for_text_classifier():
  assert _html_to_plain_text("<p>Solve <strong>x + 2 = 5</strong>.</p>") == (
    "Solve x + 2 = 5."
  )
