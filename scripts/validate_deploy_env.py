#!/usr/bin/env python3
"""
Validate deployment env files before Docker deploy targets run.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _parse_env_file(env_path: Path) -> dict[str, str]:
  values: dict[str, str] = {}
  for raw_line in env_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
      continue
    if line.startswith("export "):
      line = line[7:].strip()
    if "=" not in line:
      continue
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
      continue
    if len(value) >= 2 and ((value[0] == value[-1] == '"')
                            or (value[0] == value[-1] == "'")):
      value = value[1:-1]
    values[key] = value
  return values


def _is_truthy(value: str) -> bool:
  return value.strip().lower() in {"1", "true", "yes", "on"}


def _validate(values: dict[str, str], *, require_prod_pair: bool) -> tuple[list[str], list[str]]:
  errors: list[str] = []
  warnings: list[str] = []

  if not values.get("CANVAS_API_URL", "").strip():
    errors.append("Set CANVAS_API_URL for this deployment.")

  if not _is_truthy(values.get("GRADING_STRICT_STARTUP_CONFIG", "true")):
    warnings.append(
      "GRADING_STRICT_STARTUP_CONFIG is not true; startup config errors may not fail fast."
    )
  if not _is_truthy(values.get("AUTH_COOKIE_SECURE", "true")):
    warnings.append(
      "AUTH_COOKIE_SECURE is not true; session cookies may be exposed on non-HTTPS connections."
    )
  if not _is_truthy(values.get("QUIZGEN_ALLOW_GENERATOR", "false")):
    warnings.append(
      "QUIZGEN_ALLOW_GENERATOR is not true; quiz answer regeneration for generator-backed questions will fail."
    )

  return errors, warnings


def main() -> int:
  parser = argparse.ArgumentParser(description="Validate deploy env file")
  parser.add_argument("env_file", help="Path to .env-style file")
  parser.add_argument("--require-prod-pair",
                      action="store_true",
                      help="Deprecated compatibility option; CANVAS_API_URL is always required")
  args = parser.parse_args()

  env_path = Path(args.env_file)
  if not env_path.exists():
    print(f"[ERROR] Env file not found: {env_path}")
    return 1

  values = _parse_env_file(env_path)
  errors, warnings = _validate(values,
                               require_prod_pair=args.require_prod_pair)

  for warning in warnings:
    print(f"[WARN] {warning}")

  if errors:
    for error in errors:
      print(f"[ERROR] {error}")
    return 1

  print(f"[OK] Env validation passed: {env_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
