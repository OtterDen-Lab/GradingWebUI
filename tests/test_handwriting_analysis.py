"""Compatibility tests for structured handwriting-analysis results."""

import pytest

from grading_web_ui.web_api.routes.problems import (
  _batch_decipher_handwriting,
  _create_handwriting_job,
  _handwriting_jobs,
  _handwriting_jobs_lock,
  _parse_handwriting_analysis,
  _parse_relevance_classification,
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


def test_parses_text_only_relevance_json():
  assert _parse_relevance_classification('{"is_relevant": "false"}') is False


def test_batch_retries_a_failed_transcription_once(monkeypatch):
  calls = []

  def retry_once(problem_id, *_args, **_kwargs):
    calls.append(problem_id)
    if len(calls) == 1:
      raise RuntimeError("transient provider failure")
    return {"is_blank": False, "is_relevant": True}

  monkeypatch.setattr(
    "grading_web_ui.web_api.routes.problems._decipher_handwriting", retry_once
  )
  job_id = _create_handwriting_job(1, 2, [99], "ollama")
  _batch_decipher_handwriting(job_id, [99], "ollama", 1, False)

  with _handwriting_jobs_lock:
    job = _handwriting_jobs.pop(job_id)
  assert calls == [99, 99]
  assert job["status"] == "completed"
  assert job["succeeded"] == 1
  assert job["failed"] == 0
  assert job["failed_problem_ids"] == []
