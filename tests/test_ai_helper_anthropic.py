import pytest

from grading_web_ui.ai_helper import AI_Helper__Anthropic


class _Usage:

  def __init__(self, input_tokens: int = 11, output_tokens: int = 7):
    self.input_tokens = input_tokens
    self.output_tokens = output_tokens


class _Content:

  def __init__(self, text: str):
    self.text = text


class _Response:

  def __init__(self, text: str):
    self.content = [_Content(text)]
    self.usage = _Usage()


class _FakeMessages:

  def __init__(self, handler):
    self._handler = handler

  def create(self, **kwargs):
    return self._handler(kwargs["model"])


class _FakeClient:

  def __init__(self, handler):
    self.messages = _FakeMessages(handler)


def test_anthropic_candidate_models_from_env(monkeypatch):
  monkeypatch.setenv("ANTHROPIC_MODEL", "model-primary")
  monkeypatch.setenv("ANTHROPIC_FALLBACK_MODELS",
                     "model-a, model-b, model-primary, model-c")

  candidates = AI_Helper__Anthropic._candidate_models()
  assert candidates[:4] == ["model-primary", "model-a", "model-b", "model-c"]
  assert len(candidates) == len(set(candidates))


def test_anthropic_query_falls_back_when_model_not_found(monkeypatch):
  monkeypatch.setenv("ANTHROPIC_MODEL", "bad-model")
  monkeypatch.setenv("ANTHROPIC_FALLBACK_MODELS", "good-model")

  calls = []

  def handler(model):
    calls.append(model)
    if model == "bad-model":
      raise RuntimeError(
        "Error code: 404 - {'type':'error','error':{'type':'not_found_error','message':'model: bad-model'}}"
      )
    return _Response("ok-from-fallback")

  AI_Helper__Anthropic._client = _FakeClient(handler)
  text, usage = AI_Helper__Anthropic.query_ai("hello", attachments=[])

  assert text == "ok-from-fallback"
  assert usage["provider"] == "anthropic"
  assert usage["model"] == "good-model"
  assert calls == ["bad-model", "good-model"]


def test_anthropic_query_does_not_swallow_non_model_errors(monkeypatch):
  monkeypatch.setenv("ANTHROPIC_MODEL", "primary-model")
  monkeypatch.setenv("ANTHROPIC_FALLBACK_MODELS", "fallback-model")

  def handler(_model):
    raise RuntimeError("429 rate limit")

  AI_Helper__Anthropic._client = _FakeClient(handler)

  with pytest.raises(RuntimeError, match="429 rate limit"):
    AI_Helper__Anthropic.query_ai("hello", attachments=[])


def test_anthropic_query_reports_clear_error_when_all_candidates_missing(monkeypatch):
  monkeypatch.setenv("ANTHROPIC_MODEL", "missing-primary")
  monkeypatch.setenv("ANTHROPIC_FALLBACK_MODELS", "missing-fallback")

  def handler(_model):
    raise RuntimeError(
      "Error code: 404 - {'type':'error','error':{'type':'not_found_error','message':'model unavailable'}}"
    )

  AI_Helper__Anthropic._client = _FakeClient(handler)

  with pytest.raises(RuntimeError, match="No available Anthropic model"):
    AI_Helper__Anthropic.query_ai("hello", attachments=[])
