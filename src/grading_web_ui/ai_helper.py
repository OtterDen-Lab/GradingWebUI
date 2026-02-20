import abc
import json
import os
import random
import re
import ollama
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
  "claude-3-7-sonnet-latest,claude-3-5-sonnet-latest,claude-haiku-4-5,claude-3-5-haiku-latest"
)

# Provider model defaults (can be overridden via environment variables):
# ANTHROPIC_MODEL_SMALL/MEDIUM/LARGE
# OPENAI_MODEL_SMALL/MEDIUM/LARGE
# OLLAMA_MODEL_SMALL/MEDIUM/LARGE
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
  "ollama": {
    "small": "qwen3:4b",
    "medium": "qwen3:14b",
    "large": "qwen3:32b",
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

    fallback_csv = os.getenv("ANTHROPIC_FALLBACK_MODELS",
                             DEFAULT_ANTHROPIC_ENV_FALLBACKS)
    env_fallbacks = _parse_model_csv(fallback_csv)

    seed_models = explicit_models if explicit_models else [primary]

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

        return response.content[0].text, usage_info
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


class AI_Helper__OpenAI(AI_Helper):

  def __init__(self) -> None:
    super().__init__()
    self.__class__._client = OpenAI()

  @classmethod
  def query_ai(cls,
               message: str,
               attachments: List[Tuple[str, str]],
               max_response_tokens: int = DEFAULT_MAX_TOKENS,
               max_retries: int = DEFAULT_MAX_RETRIES) -> Tuple[Dict, Dict]:
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
      model="gpt-4.1-nano",
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
      "openai"
    }

    try:
      content = json.loads(response.choices[0].message.content)
      return content, usage_info
    except TypeError:
      if max_retries > 0:
        return cls.query_ai(message, attachments, max_response_tokens,
                            max_retries - 1)
      else:
        return {}, usage_info


class AI_Helper__Ollama(AI_Helper):

  def __init__(self):
    super().__init__()
    # Initialize client if not already done
    if self.__class__._client is None:
      ollama_host = os.getenv('OLLAMA_HOST', 'http://workhorse:11434')
      ollama_timeout = int(os.getenv('OLLAMA_TIMEOUT', '30'))
      log.info(
        f"Initializing Ollama client with host: {ollama_host}, timeout: {ollama_timeout}s"
      )
      self.__class__._client = ollama.Client(host=ollama_host,
                                             timeout=ollama_timeout)

  @classmethod
  def query_ai(cls,
               message: str,
               attachments: List[Tuple[str, str]],
               max_response_tokens: int = DEFAULT_MAX_TOKENS,
               max_retries: int = DEFAULT_MAX_RETRIES) -> Tuple[str, Dict]:

    # Ensure client is initialized
    if cls._client is None:
      ollama_host = os.getenv('OLLAMA_HOST', 'http://workhorse:11434')
      ollama_timeout = int(os.getenv('OLLAMA_TIMEOUT', '30'))
      log.info(
        f"Lazily initializing Ollama client with host: {ollama_host}, timeout: {ollama_timeout}s"
      )
      cls._client = ollama.Client(host=ollama_host, timeout=ollama_timeout)

    # Extract base64 images from attachments (format: [("png", base64_str), ...])
    images = [
      att[1] for att in attachments if att[0] in ("png", "jpg", "jpeg")
    ]

    # Build message for Ollama
    msg_content = {'role': 'user', 'content': message}

    # Add images if present
    if images:
      msg_content['images'] = images

    # Use the client instance to make the request
    # Model can be configured via environment variable or default to qwen3-vl:2b
    model = os.getenv('OLLAMA_MODEL', 'qwen3-vl:2b')

    log.info(
      f"Ollama: Using model {model} with host {cls._client._client.base_url}")
    log.debug(f"Ollama: Message content has {len(images)} images")

    try:
      # Use streaming mode - timeout resets on each chunk received
      # This differentiates between "actively processing" vs "broken connection"
      # Add options to reduce overthinking/hallucination
      options = {
        'temperature': 0.1,  # Lower temperature = more focused, less creative
        'top_p': 0.9,  # Nucleus sampling
        'num_predict': 500,  # Limit output length to prevent rambling
      }

      stream = cls._client.chat(model=model,
                                messages=[msg_content],
                                stream=True,
                                options=options)

      # Collect the streamed response
      content = ""
      last_response = None
      chunk_count = 0

      for chunk in stream:
        chunk_count += 1
        if chunk_count % 1000 == 0:
          log.debug(
            f"Ollama: Received chunk {chunk_count}, content length: {len(content)}"
          )

        content += chunk['message']['content']
        last_response = chunk  # Keep last chunk for metadata

      log.info(
        f"Ollama: Received {chunk_count} chunks, total {len(content)} characters"
      )

      # Extract usage information from final chunk
      prompt_tokens = last_response.get(
        'prompt_eval_count') or 0 if last_response else 0
      completion_tokens = last_response.get(
        'eval_count') or 0 if last_response else 0
      usage_info = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "provider": "ollama"
      }

      return content, usage_info

    except httpx.ReadTimeout:
      timeout = os.getenv('OLLAMA_TIMEOUT', '30')
      log.error(
        f"Ollama request timed out after {timeout}s (no data received)")
      raise
    except Exception as e:
      log.error(f"Ollama error ({type(e).__name__}): {str(e)}")
      raise
