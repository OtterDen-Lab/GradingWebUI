"""
Compatibility helpers for QuizGenerator answer regeneration APIs.
"""
import inspect
from typing import Any, Dict, Optional


def regenerate_from_encrypted_compat(encrypted_data: str,
                                     points: float = 1.0,
                                     yaml_text: Optional[str] = None,
                                     image_mode: Optional[str] = None
                                     ) -> Dict[str, Any]:
  """
  Call QuizGenerator.regenerate_from_encrypted across multiple API versions.

  Some QuizGenerator releases only accept `(encrypted_data, points)` while
  newer releases may accept `yaml_text` and/or `image_mode`.
  """
  from QuizGenerator.regenerate import regenerate_from_encrypted

  kwargs: Dict[str, Any] = {
    "encrypted_data": encrypted_data,
    "points": points,
  }

  parameters = inspect.signature(regenerate_from_encrypted).parameters
  if yaml_text is not None and "yaml_text" in parameters:
    kwargs["yaml_text"] = yaml_text
  if image_mode is not None and "image_mode" in parameters:
    kwargs["image_mode"] = image_mode

  return regenerate_from_encrypted(**kwargs)

