#!/usr/bin/env python3
"""Create a verified, consistent backup of a GradingWebUI SQLite database."""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from grading_web_ui.web_api.database import create_database_backup, get_db_path


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Create a verified SQLite snapshot safe to run while GradingWebUI is active.")
  parser.add_argument("--db-path", help="Source database (defaults to GRADING_DB_PATH).")
  parser.add_argument(
    "--output",
    required=True,
    help="Destination backup file. It must not already exist.")
  parser.add_argument(
    "--manifest",
    help="Optional JSON manifest path. Defaults to <output>.json.")
  args = parser.parse_args()

  if args.db_path:
    os.environ["GRADING_DB_PATH"] = args.db_path
  source_path = get_db_path()
  destination_path = Path(args.output)
  manifest_path = Path(args.manifest) if args.manifest else Path(
    f"{destination_path}.json")
  if manifest_path.exists():
    parser.error(f"Refusing to overwrite existing manifest: {manifest_path}")

  try:
    result = create_database_backup(source_path, destination_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  except (FileNotFoundError, FileExistsError, OSError, RuntimeError) as error:
    print(f"Backup failed: {error}", file=sys.stderr)
    return 1

  print(json.dumps(result, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
