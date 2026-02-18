"""
Runtime encryption-key management for QuizGenerator without mutating os.environ.
"""
import logging
import os
from threading import Lock
from typing import Optional

log = logging.getLogger(__name__)

_LOCK = Lock()
_RUNTIME_KEY: Optional[bytes] = None
_GENERATED_KEY: Optional[bytes] = None
_PATCHED = False


def set_runtime_encryption_key(key: str) -> None:
  """Set in-memory encryption key for the current process."""
  key_bytes = (key or "").strip().encode()
  if not key_bytes:
    raise ValueError("Encryption key cannot be empty")
  with _LOCK:
    global _RUNTIME_KEY
    _RUNTIME_KEY = key_bytes


def get_runtime_encryption_key() -> Optional[bytes]:
  """Get in-memory encryption key if one has been set."""
  with _LOCK:
    return _RUNTIME_KEY


def clear_runtime_encryption_key() -> None:
  """Clear in-memory runtime key."""
  with _LOCK:
    global _RUNTIME_KEY
    _RUNTIME_KEY = None


def _resolve_effective_key() -> bytes:
  runtime_key = get_runtime_encryption_key()
  if runtime_key:
    return runtime_key

  env_value = os.getenv("QUIZ_ENCRYPTION_KEY")
  if env_value:
    return env_value.encode()

  with _LOCK:
    global _GENERATED_KEY
    if _GENERATED_KEY is None:
      from cryptography.fernet import Fernet
      _GENERATED_KEY = Fernet.generate_key()
      log.warning(
        "QUIZ_ENCRYPTION_KEY is not set. Using in-memory temporary key for this process."
      )
    return _GENERATED_KEY


def install_quizgenerator_key_provider() -> bool:
  """
  Patch QuizGenerator to use our key resolver and avoid writing to os.environ.
  """
  global _PATCHED
  if _PATCHED:
    return True

  try:
    from QuizGenerator.qrcode_generator import QuestionQRCode
  except Exception as exc:
    log.warning("QuizGenerator not available for key-provider patch: %s", exc)
    return False

  def _get_encryption_key(cls) -> bytes:
    return _resolve_effective_key()

  QuestionQRCode.get_encryption_key = classmethod(_get_encryption_key)
  _PATCHED = True
  return True
