from grading_web_ui.ai_helper import AI_Helper__Bifrost


class _Response:

  def __init__(self, body):
    self.body = body

  def raise_for_status(self):
    return None

  def json(self):
    return self.body


def test_bifrost_lists_proxy_catalog(monkeypatch):
  monkeypatch.setenv("BIFROST_BASE_URL", "http://bifrost.test/v1")
  monkeypatch.setenv("BIFROST_API_KEY", "token")
  calls = []

  def get(url, **kwargs):
    calls.append((url, kwargs))
    return _Response({"data": [{"id": "vision"}, {
      "id": "fast", "display_name": "Fast route"
    }]})

  monkeypatch.setattr("grading_web_ui.ai_helper.requests.get", get)
  assert AI_Helper__Bifrost().list_models() == [
    {"id": "vision", "display_name": "vision"},
    {"id": "fast", "display_name": "Fast route"},
  ]
  assert calls[0][0] == "http://bifrost.test/v1/models"
  assert calls[0][1]["headers"] == {"Authorization": "Bearer token"}


def test_bifrost_adds_v1_to_a_bare_proxy_url(monkeypatch):
  monkeypatch.setenv("BIFROST_BASE_URL", "https://bifrost.example.test")
  monkeypatch.setenv("BIFROST_API_KEY", "token")
  calls = []

  def get(url, **kwargs):
    calls.append((url, kwargs))
    return _Response({"data": []})

  monkeypatch.setattr("grading_web_ui.ai_helper.requests.get", get)
  AI_Helper__Bifrost().list_models()
  assert calls[0][0] == "https://bifrost.example.test/v1/models"


def test_bifrost_chat_uses_openai_wire_format(monkeypatch):
  monkeypatch.setenv("BIFROST_BASE_URL", "http://bifrost.test/v1")
  monkeypatch.setenv("BIFROST_API_KEY", "token")
  calls = []

  def post(url, **kwargs):
    calls.append((url, kwargs))
    return _Response({
      "model": "vision", "choices": [{"message": {"content": '{"text":"ok"}'}}],
      "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    })

  monkeypatch.setattr("grading_web_ui.ai_helper.requests.post", post)
  response, usage = AI_Helper__Bifrost().query_ai(
    "Read this", [("png", "encoded")], candidate_models=["vision"], json_output=True)

  assert response == '{"text":"ok"}'
  assert usage == {
    "prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5,
    "provider": "bifrost", "model": "vision",
  }
  payload = calls[0][1]["json"]
  assert calls[0][0] == "http://bifrost.test/v1/chat/completions"
  assert payload["model"] == "vision"
  assert payload["response_format"] == {"type": "json_object"}
  assert payload["messages"][0]["content"][1]["image_url"]["url"] == \
    "data:image/png;base64,encoded"


def test_bifrost_requires_a_selected_model(monkeypatch):
  monkeypatch.setenv("BIFROST_BASE_URL", "http://bifrost.test/v1")
  monkeypatch.setenv("BIFROST_API_KEY", "token")
  monkeypatch.delenv("BIFROST_DEFAULT_MODEL", raising=False)

  try:
    AI_Helper__Bifrost().query_ai("hello", [])
  except RuntimeError as error:
    assert "No Bifrost model is selected" in str(error)
  else:
    raise AssertionError("expected missing model error")
