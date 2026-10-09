"""
Compatibility helpers for QuizGenerator answer regeneration APIs.
"""
import inspect
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


def _quiet_quizgenerator_loggers() -> None:
  """Suppress noisy QuizGenerator info logs during regeneration."""
  for logger_name in (
    "QuizGenerator",
    "QuizGenerator.quiz",
    "QuizGenerator.regenerate",
    "QuizGenerator.question",
  ):
    logger = logging.getLogger(logger_name)
    if logger.level == logging.NOTSET or logger.level < logging.WARNING:
      logger.setLevel(logging.WARNING)


def _restore_app_logging_after_quizgenerator_import() -> None:
  """
  QuizGenerator applies its own logging config at import time.
  Re-apply host logging so QuizGenerator INFO chatter stays suppressed.
  """
  try:
    from grading_web_ui import setup_logging as setup_app_logging
    setup_app_logging()
  except Exception:
    # If host logging cannot be restored, still clamp QuizGenerator levels.
    pass
  _quiet_quizgenerator_loggers()


def _configure_matplotlib_for_worker_context() -> None:
  """
  Ensure QuizGenerator image rendering uses a non-interactive backend.

  Finalization and some grading flows run in worker threads. GUI backends
  (e.g., macOS backend) can fail in non-main threads when premade questions
  generate plots for explanations.
  """
  os.environ.setdefault("MPLBACKEND", "Agg")

  # Keep matplotlib cache writable in environments where $HOME is read-only.
  if "MPLCONFIGDIR" not in os.environ:
    mpl_cache_dir = Path(tempfile.gettempdir()) / "gradingwebui-matplotlib"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

  try:
    import matplotlib  # type: ignore
    backend = (matplotlib.get_backend() or "").lower()
    if "agg" not in backend:
      matplotlib.use("Agg", force=True)
  except Exception:
    # If matplotlib is unavailable or already in a safe state, defer to
    # QuizGenerator behavior and let existing error handling surface issues.
    return


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
  _configure_matplotlib_for_worker_context()
  from QuizGenerator.regenerate import regenerate_from_encrypted
  _restore_app_logging_after_quizgenerator_import()

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


def regenerate_question_html_from_encrypted(
    encrypted_data: str,
    points: float = 1.0,
    yaml_text: Optional[str] = None,
    image_mode: str = "inline"
) -> str:
  """Regenerate only the student-visible question body, without answers.

  QuizGenerator's public regeneration API intentionally returns an answer key.
  The feedback-example export needs the same seeded question, but must not
  reveal the answer through that question snippet.  This mirrors that API's
  regeneration path and renders the question body in QuizGenerator's
  ``review_mode``. That mode renders visible answer fields instead of exposing
  the internal UUID used to associate each blank with its answer.
  """
  _configure_matplotlib_for_worker_context()
  from QuizGenerator.regenerate import (
    QuestionQRCode,
    QuestionRegistry,
    _find_questions_by_id,
    _load_quizzes_for_question_id,
    _load_yaml_docs,
    _render_html,
    _resolve_upload_func,
  )
  _restore_app_logging_after_quizgenerator_import()

  regenerated = QuestionQRCode.decrypt_question_data(encrypted_data)
  seed = regenerated.get("seed")
  question_id = regenerated.get("question_id")
  upload_func = _resolve_upload_func(image_mode, None)

  if question_id:
    if not yaml_text:
      raise ValueError("Frozen quiz YAML is required to regenerate this question")
    docs = _load_yaml_docs(yaml_text=yaml_text)
    quizzes = _load_quizzes_for_question_id(docs, question_id)
    matches = _find_questions_by_id(quizzes, question_id)
    if not matches:
      raise ValueError(f"Question ID '{question_id}' was not found in the frozen YAML")
    question = matches[0]
    instance = question.instantiate(rng_seed=seed)
  else:
    question = QuestionRegistry.create(
      regenerated["question_type"],
      name=f"FeedbackExample_{seed}",
      points_value=points,
      **regenerated.get("config", {}),
    )
    instance = question.instantiate(
      rng_seed=seed,
      **regenerated.get("context", {}),
    )

  question_ast = question._build_question_ast(instance)
  return _render_html(
    question_ast.body,
    show_answers=False,
    review_mode=True,
    upload_func=upload_func,
  )
