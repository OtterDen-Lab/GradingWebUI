"""
Startup configuration validation helpers.
"""
import os
from typing import List


def _is_truthy(value: str) -> bool:
  return value.strip().lower() in ("1", "true", "yes", "on")


def validate_startup_configuration() -> List[str]:
  """
  Validate critical startup settings.

  Returns warnings that should be logged.
  Raises RuntimeError when strict validation is enabled and required config is missing.
  """
  strict = _is_truthy(os.getenv("GRADING_STRICT_STARTUP_CONFIG", "true"))
  errors: List[str] = []
  warnings: List[str] = []

  dev_url = os.getenv("CANVAS_API_URL", "").strip()
  dev_key = os.getenv("CANVAS_API_KEY", "").strip()
  prod_url = (os.getenv("CANVAS_API_URL_PROD", "").strip() or
              os.getenv("CANVAS_API_URL_prod", "").strip())
  prod_key = (os.getenv("CANVAS_API_KEY_PROD", "").strip() or
              os.getenv("CANVAS_API_KEY_prod", "").strip())

  # URLs identify the institution's Canvas environments; API keys are now
  # normally stored encrypted per user. A legacy environment key remains
  # harmless, but must still have a matching URL when supplied.
  if dev_key and not dev_url:
    errors.append(
      "CANVAS_API_KEY requires CANVAS_API_URL for dev Canvas access."
    )
  if prod_key and not prod_url:
    errors.append(
      "CANVAS_API_KEY_PROD requires CANVAS_API_URL_PROD for prod Canvas access."
    )
  if not (dev_url or prod_url):
    errors.append(
      "Canvas credentials are missing. Set CANVAS_API_URL (or CANVAS_API_URL_PROD)."
    )

  if not _is_truthy(os.getenv("AUTH_COOKIE_SECURE", "true")):
    warnings.append(
      "AUTH_COOKIE_SECURE is false. Session cookies may be exposed on non-HTTPS connections."
    )

  if not _is_truthy(os.getenv("QUIZGEN_ALLOW_GENERATOR", "false")):
    warnings.append(
      "QUIZGEN_ALLOW_GENERATOR is false. Quiz answer regeneration for generator-backed questions will fail."
    )

  if strict and errors:
    raise RuntimeError("Startup configuration validation failed: " + " ".join(errors))

  warnings.extend(errors)
  return warnings
