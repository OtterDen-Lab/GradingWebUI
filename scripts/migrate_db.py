#!/usr/bin/env python3
"""
Database migration helper for controlled upgrades.
"""
import argparse
import os

from grading_web_ui.web_api.database import (DB_CREATE_MIGRATION_BACKUP_ENV,
                                             DB_MIGRATION_BACKUP_DIR_ENV,
                                             get_db_connection,
                                             get_schema_version, init_database)


def _current_schema_version() -> int:
  with get_db_connection() as conn:
    return get_schema_version(conn.cursor())


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Run GradingWebUI database migrations with preflight backup.")
  parser.add_argument("--db-path",
                      help="Path to SQLite database (sets GRADING_DB_PATH).")
  parser.add_argument(
    "--backup-dir",
    help=
    "Directory for pre-migration backups (sets GRADING_DB_MIGRATION_BACKUP_DIR)."
  )
  parser.add_argument("--no-backup",
                      action="store_true",
                      help="Disable pre-migration backup creation.")
  args = parser.parse_args()

  if args.db_path:
    os.environ["GRADING_DB_PATH"] = args.db_path
  if args.backup_dir:
    os.environ[DB_MIGRATION_BACKUP_DIR_ENV] = args.backup_dir
  if args.no_backup:
    os.environ[DB_CREATE_MIGRATION_BACKUP_ENV] = "false"

  before_version = _current_schema_version()
  print(f"Schema version before migration: {before_version}")
  init_database()
  after_version = _current_schema_version()
  print(f"Schema version after migration: {after_version}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
