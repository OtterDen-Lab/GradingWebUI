import abc
import json
import os
import random
import re
from typing import Tuple, Dict, List, Optional

import dotenv
import openai.types.chat.completion_create_params
from openai import OpenAI
from anthropic import Anthropic
import httpx

import logging

log = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_TOKENS = 1000  # Default token limit for AI responses
DEFAULT_MAX_RETRIES = 3  # Default number of retries for failed requests
DEFAULT_ANTHROPIC_ENV_FALLBACKS = (
  "claude-haiku-4-5,claude-sonnet-4-5,claude-opus-4-5"
)

# Provider model defaults (can be overridden via environment variables):
# ANTHROPIC_MODEL_SMALL/MEDIUM/LARGE
# OPENAI_MODEL_SMALL/MEDIUM/LARGE
MODEL_CONFIG = {
  "anthropic": {
    "small": "claude-haiku-4-5",
    "medium": "claude-sonnet-4-5",
    "large": "claude-opus-4-5",
  },
  "openai": {
    "small": "gpt-4.1-nano",
    "medium": "gpt-4.1-mini",
    "large": "gpt-4.1",
  },
}

DEFAULT_MODEL_TIER = "medium"


def _parse_model_csv(raw_value: str) -> List[str]:
  return [model.strip() for model in (raw_value or "").split(",") if model.strip()]


def get_model_for_tier(provider: str, tier: str = DEFAULT_MODEL_TIER) -> str:
  provider_key = (provider or "").strip().lower()
  tier_key = (tier or DEFAULT_MODEL_TIER).strip().lower()

  provider_models = MODEL_CONFIG.get(provider_key, {})
  if tier_key not in provider_models:
    tier_key = "small"

  env_key = f"{provider_key.upper()}_MODEL_{tier_key.upper()}"
  env_value = os.getenv(env_key, "").strip()
  if env_value:
    return env_value

  return provider_models.get(tier_key, "unknown")


class AI_Helper(abc.ABC):
  _client = None

  def __init__(self) -> None:
    if self._client is None:
      log.debug("Loading dotenv")  # Load the .env file
      dotenv.load_dotenv(os.path.expanduser('~/.env'))

  @classmethod
  @abc.abstractmethod
  def query_ai(cls, message: str, attachments: List[Tuple[str, str]], *args,
               **kwargs) -> str:
    pass


class AI_Helper__Anthropic(AI_Helper):

  def __init__(self) -> None:
    super().__init__()
    self.__class__._client = Anthropic()

  @classmethod
  def _candidate_models(cls,
                        explicit_models: Optional[List[str]] = None) -> List[str]:
    def _dedupe(sequence: List[str]) -> List[str]:
      seen = set()
      deduped = []
      for model in sequence:
        if model and model not in seen:
          seen.add(model)
          deduped.append(model)
      return deduped

    primary = os.getenv("ANTHROPIC_MODEL", "").strip()
    if not primary:
      primary = get_model_for_tier("anthropic", "medium")

    if explicit_models:
      # Explicit candidate lists should be deterministic and must not pull in
      # stale environment fallbacks (for example deprecated model aliases).
      seed_models = explicit_models
      env_fallbacks = []
    else:
      fallback_csv = os.getenv("ANTHROPIC_FALLBACK_MODELS",
                               DEFAULT_ANTHROPIC_ENV_FALLBACKS)
      env_fallbacks = _parse_model_csv(fallback_csv)
      seed_models = [primary]

    # Always append resilient built-in candidates so stale env overrides do not
    # hard-fail model selection.
    resilient_fallbacks = _dedupe([
      get_model_for_tier("anthropic", "medium"),
      get_model_for_tier("anthropic", "small"),
      get_model_for_tier("anthropic", "large"),
      *_parse_model_csv(DEFAULT_ANTHROPIC_ENV_FALLBACKS),
    ])

    return _dedupe([
      *seed_models,
      *env_fallbacks,
      *resilient_fallbacks,
    ])

  @staticmethod
  def _is_model_not_found_error(error: Exception) -> bool:
    msg = str(error).lower()
    if "not_found_error" in msg and "model" in msg:
      return True
    if "model" in msg and re.search(r"\bnot found\b", msg):
      return True
    return False

  @classmethod
  def query_ai(cls,
               message: str,
               attachments: List[Tuple[str, str]],
               max_response_tokens: int = DEFAULT_MAX_TOKENS,
               max_retries: int = DEFAULT_MAX_RETRIES,
               candidate_models: Optional[List[str]] = None) -> Tuple[str, Dict]:
    messages = []

    attachment_messages = []
    for file_type, b64_file_contents in attachments:
      if file_type == "png":
        attachment_messages.append({
          "type": "image",
          "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": b64_file_contents
          }
        })

    messages.append({
      "role":
      "user",
      "content": [{
        "type": "text",
        "text": f"{message}"
      }, *attachment_messages]
    })

    last_error = None
    model_not_found_count = 0
    candidate_models = cls._candidate_models(candidate_models)

    for index, model_name in enumerate(candidate_models):
      try:
        response = cls._client.messages.create(model=model_name,
                                               max_tokens=max_response_tokens,
                                               messages=messages)
        log.debug(response.content)

        # Extract usage information
        usage_info = {
          "prompt_tokens":
          response.usage.input_tokens if response.usage else 0,
          "completion_tokens":
          response.usage.output_tokens if response.usage else 0,
          "total_tokens":
          (response.usage.input_tokens +
           response.usage.output_tokens) if response.usage else 0,
          "provider":
          "anthropic",
          "model":
          model_name
        }

        # Newer Claude models may place a ThinkingBlock before one or more
        # TextBlocks. Do not assume content[0] is directly displayable.
        text_parts = [block.text for block in response.content
                      if getattr(block, "type", None) == "text" or hasattr(block, "text")]
        if not text_parts:
          raise RuntimeError("Anthropic response contained no text block")
        return "".join(text_parts), usage_info
      except Exception as e:
        last_error = e
        is_model_error = cls._is_model_not_found_error(e)
        if is_model_error:
          model_not_found_count += 1
        has_next_model = index < (len(candidate_models) - 1)
        if is_model_error and has_next_model:
          log.warning(
            "Anthropic model '%s' unavailable, trying fallback model '%s'",
            model_name,
            candidate_models[index + 1]
          )
          continue
        if is_model_error:
          break
        raise

    if last_error:
      if model_not_found_count == len(candidate_models):
        tried = ", ".join(candidate_models)
        raise RuntimeError(
          "No available Anthropic model from configured candidates. "
          f"Tried: {tried}. "
          "Update ANTHROPIC_MODEL / ANTHROPIC_FALLBACK_MODELS to valid models."
        ) from last_error
      raise last_error
    raise RuntimeError("Anthropic query failed with no candidate model attempts")


class AI_Helper__Ollama(AI_Helper):
  """Thin client for an operator-provided Ollama server; no models are bundled."""

  def __init__(self, base_url: str, model: str) -> None:
    self.base_url = base_url.rstrip("/")
    self.model = model

  def query_ai(self,
               message: str,
               attachments: List[Tuple[str, str]],
               max_response_tokens: int = DEFAULT_MAX_TOKENS,
               json_output: bool = False,
               **_kwargs) -> Tuple[str, Dict]:
    images = [contents for file_type, contents in attachments if file_type == "png"]
    payload = {
      "model": self.model,
      "stream": False,
      # Vision transcription needs a concise final answer. Reasoning-capable
      # models (including Qwen 3) can otherwise consume num_predict before
      # emitting it.
      "think": False,
      "messages": [{"role": "user", "content": message, "images": images}],
      "options": {"num_predict": max_response_tokens},
    }
    if json_output:
      payload["format"] = "json"
    response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=300.0)
    response.raise_for_status()
    body = response.json()
    content = body.get("message", {}).get("content", "")
    if not content or not content.strip():
      generated = body.get("eval_count", 0)
      reason = body.get("done_reason", "unknown")
      thinking = body.get("message", {}).get("thinking", "")
      detail = (
        f"Ollama completed the request but returned no final text "
        f"(generated {generated} tokens; stop reason: {reason}).")
      if thinking:
        detail += " The model returned reasoning but no final answer."
      raise RuntimeError(detail)
    return content, {
      "prompt_tokens": body.get("prompt_eval_count", 0),
      "completion_tokens": body.get("eval_count", 0),
      "total_tokens": body.get("prompt_eval_count", 0) + body.get("eval_count", 0),
      "provider": "ollama",
      "model": self.model,
    }


class AI_Helper__OpenAI(AI_Helper):

  def __init__(self) -> None:
    super().__init__()
    self.__class__._client = OpenAI()

  @classmethod
  def query_ai(cls,
               message: str,
               attachments: List[Tuple[str, str]],
               max_response_tokens: int = DEFAULT_MAX_TOKENS,
               max_retries: int = DEFAULT_MAX_RETRIES,
               candidate_models: Optional[List[str]] = None,
               return_raw: bool = False) -> Tuple[Dict | str, Dict]:
    messages = []

    attachment_messages = []
    for file_type, b64_file_contents in attachments:
      if file_type == "png":
        attachment_messages.append({
          "type": "image_url",
          "image_url": {
            "url": f"data:image/png;base64,{b64_file_contents}"
          }
        })

    messages.append({
      "role":
      "user",
      "content": [{
        "type": "text",
        "text": f"{message}"
      }, *attachment_messages]
    })

    response = cls._client.chat.completions.create(
      model=(candidate_models or ["gpt-4.1-nano"])[0],
      response_format={"type": "json_object"},
      messages=messages,
      temperature=1,
      max_tokens=max_response_tokens,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0)
    log.debug(response.choices[0])

    # Extract usage information
    usage_info = {
      "prompt_tokens":
      response.usage.prompt_tokens if response.usage else 0,
      "completion_tokens":
      response.usage.completion_tokens if response.usage else 0,
      "total_tokens":
      response.usage.total_tokens if response.usage else 0,
      "provider":
      "openai",
      "model": (candidate_models or ["gpt-4.1-nano"])[0],
    }

    raw_content = response.choices[0].message.content
    if return_raw:
      return raw_content or "", usage_info
    try:
      content = json.loads(raw_content)
      return content, usage_info
    except TypeError:
      if max_retries > 0:
        return cls.query_ai(message, attachments, max_response_tokens,
                            max_retries - 1, candidate_models, return_raw)
      else:
        return {}, usage_info
