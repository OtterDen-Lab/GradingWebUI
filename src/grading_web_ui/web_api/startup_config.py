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

  if bool(dev_url) != bool(dev_key):
    errors.append(
      "Both CANVAS_API_URL and CANVAS_API_KEY must be set together for dev Canvas access."
    )
  if bool(prod_url) != bool(prod_key):
    errors.append(
      "Both CANVAS_API_URL_PROD and CANVAS_API_KEY_PROD must be set together for prod Canvas access."
    )
  if not ((dev_url and dev_key) or (prod_url and prod_key)):
    errors.append(
      "Canvas credentials are missing. Set CANVAS_API_URL and CANVAS_API_KEY (or prod equivalents)."
    )

  if not _is_truthy(os.getenv("AUTH_COOKIE_SECURE", "true")):
    warnings.append(
      "AUTH_COOKIE_SECURE is false. Session cookies may be exposed on non-HTTPS connections."
    )

  if strict and errors:
    raise RuntimeError("Startup configuration validation failed: " + " ".join(errors))

  warnings.extend(errors)
  return warnings

