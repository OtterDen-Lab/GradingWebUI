"""QuizGenerator release-status checks used before answer regeneration."""
from importlib.metadata import PackageNotFoundError, version as package_version
import threading
import time
from typing import Optional

import requests
from packaging.version import InvalidVersion, Version


_PYPI_URL = "https://pypi.org/pypi/QuizGenerator/json"
_CHECK_TTL_SECONDS = 60 * 60
_ERROR_TTL_SECONDS = 5 * 60
_cache_lock = threading.Lock()
_cached_status: Optional[dict] = None
_cached_at = 0.0


def installed_quizgenerator_version() -> str:
  """Return the QuizGenerator version available to this running server."""
  try:
    return package_version("QuizGenerator")
  except PackageNotFoundError:
    return "unavailable"


def _is_current(installed: str, latest: str) -> bool:
  try:
    return Version(installed) >= Version(latest)
  except InvalidVersion:
    return installed == latest


def get_quizgenerator_version_status() -> dict:
  """Compare the installed version with PyPI, caching the remote result.

  A failed network check is deliberately non-fatal: regeneration remains available,
  while callers can tell that freshness could not be verified.
  """
  global _cached_status, _cached_at

  installed = installed_quizgenerator_version()
  now = time.monotonic()
  with _cache_lock:
    if (_cached_status and _cached_status["installed_version"] == installed
        and now - _cached_at < _cached_status["cache_ttl_seconds"]):
      return dict(_cached_status)

  try:
    response = requests.get(_PYPI_URL, timeout=3)
    response.raise_for_status()
    latest = response.json()["info"]["version"]
    if not isinstance(latest, str) or not latest:
      raise ValueError("PyPI did not provide a QuizGenerator version")
    status = {
      "installed_version": installed,
      "latest_version": latest,
      "is_latest": _is_current(installed, latest),
      "check_error": None,
      "cache_ttl_seconds": _CHECK_TTL_SECONDS,
    }
  except (requests.RequestException, KeyError, TypeError, ValueError) as error:
    status = {
      "installed_version": installed,
      "latest_version": None,
      "is_latest": None,
      "check_error": str(error),
      "cache_ttl_seconds": _ERROR_TTL_SECONDS,
    }

  with _cache_lock:
    _cached_status = status
    _cached_at = now
  return dict(status)
