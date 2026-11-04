"""Bifrost client for every application AI request."""
import os
from typing import Dict, List, Optional, Tuple

import dotenv
import requests


DEFAULT_MAX_TOKENS = 1000


def get_default_model() -> str:
  """Return the optional deployment-level default before a DB choice exists."""
  return os.getenv("BIFROST_DEFAULT_MODEL", "").strip()


class AI_Helper__Bifrost:
  """Small OpenAI-compatible Bifrost client without an SDK dependency."""

  def __init__(self) -> None:
    dotenv.load_dotenv(os.path.expanduser("~/.env"))
    self.base_url = os.getenv("BIFROST_BASE_URL", "").strip().rstrip("/")
    if self.base_url and not self.base_url.endswith("/v1"):
      self.base_url = f"{self.base_url}/v1"
    self.api_key = os.getenv("BIFROST_API_KEY", "").strip()

  def _url(self, path: str) -> str:
    if not self.base_url:
      raise RuntimeError("BIFROST_BASE_URL is not configured")
    return f"{self.base_url}/{path.lstrip('/')}"

  def _headers(self) -> dict[str, str]:
    if not self.api_key:
      raise RuntimeError("BIFROST_API_KEY is not configured")
    return {"Authorization": f"Bearer {self.api_key}"}

  def list_models(self) -> list[dict]:
    response = requests.get(self._url("models"), headers=self._headers(), timeout=30)
    response.raise_for_status()
    body = response.json()
    models = body.get("data", body) if isinstance(body, dict) else body
    if not isinstance(models, list):
      raise RuntimeError("Bifrost /models returned an unexpected response")
    return [{"id": item["id"],
             "display_name": item.get("display_name", item["id"])}
            for item in models if isinstance(item, dict) and item.get("id")]

  def query_ai(self,
               message: str,
               attachments: List[Tuple[str, str]],
               max_response_tokens: int = DEFAULT_MAX_TOKENS,
               candidate_models: Optional[List[str]] = None,
               json_output: bool = False,
               **_kwargs) -> Tuple[str, Dict]:
    model = next((value.strip() for value in (candidate_models or []) if value.strip()),
                 get_default_model())
    if not model:
      raise RuntimeError(
        "No Bifrost model is selected. Choose a model in AI Settings or set "
        "BIFROST_DEFAULT_MODEL.")
    content = [{"type": "text", "text": message}]
    content.extend({
      "type": "image_url",
      "image_url": {"url": f"data:image/png;base64,{data}"},
    } for file_type, data in attachments if file_type == "png")
    payload = {
      "model": model,
      "messages": [{"role": "user", "content": content}],
      "max_tokens": max_response_tokens,
    }
    if json_output:
      payload["response_format"] = {"type": "json_object"}
    response = requests.post(self._url("chat/completions"), headers=self._headers(),
                             json=payload, timeout=300)
    response.raise_for_status()
    body = response.json()
    try:
      raw_content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
      raise RuntimeError("Bifrost returned no chat-completion content") from error
    if isinstance(raw_content, list):
      raw_content = "".join(part.get("text", "") for part in raw_content
                            if isinstance(part, dict))
    if not isinstance(raw_content, str) or not raw_content.strip():
      raise RuntimeError("Bifrost returned empty chat-completion content")
    usage = body.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    return raw_content, {
      "prompt_tokens": prompt_tokens,
      "completion_tokens": completion_tokens,
      "total_tokens": usage.get("total_tokens", prompt_tokens + completion_tokens),
      "provider": "bifrost",
      "model": body.get("model", model),
    }
