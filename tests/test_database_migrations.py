"""
Upgrade-path and preflight tests for database migrations.
"""
from pathlib import Path
import sqlite3
import shutil
from types import SimpleNamespace

import pytest

from grading_web_ui.web_api import database as db_module
from grading_web_ui.web_api.database import CURRENT_SCHEMA_VERSION, init_database


def _create_v23_database(path: Path) -> None:
  conn = sqlite3.connect(str(path))
  cursor = conn.cursor()
  cursor.execute("""
    CREATE TABLE _schema_version (
      version INTEGER PRIMARY KEY,
      applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
  """)
  cursor.execute("INSERT INTO _schema_version (version) VALUES (23)")
  cursor.execute("""
    CREATE TABLE problem_metadata (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id INTEGER NOT NULL,
      problem_number INTEGER NOT NULL,
      max_points REAL,
      question_text TEXT,
      grading_rubric TEXT,
      default_feedback TEXT,
      default_feedback_threshold REAL DEFAULT 100.0,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(session_id, problem_number)
    )
  """)
  conn.commit()
  conn.close()


def _create_future_version_database(path: Path) -> None:
  conn = sqlite3.connect(str(path))
  cursor = conn.cursor()
  cursor.execute("""
    CREATE TABLE _schema_version (
      version INTEGER PRIMARY KEY,
      applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
  """)
  cursor.execute(
    "INSERT INTO _schema_version (version) VALUES (?)",
    (CURRENT_SCHEMA_VERSION + 1, ),
  )
  conn.commit()
  conn.close()


def test_migration_from_v23_creates_backup_and_upgrades(monkeypatch, tmp_path):
  db_path = tmp_path / "legacy_v23.db"
  backup_dir = tmp_path / "backups"
  _create_v23_database(db_path)

  monkeypatch.setenv("GRADING_DB_PATH", str(db_path))
  monkeypatch.setenv("GRADING_DB_MIGRATION_BACKUP_DIR", str(backup_dir))
  monkeypatch.setenv("GRADING_DB_CREATE_MIGRATION_BACKUP", "true")

  init_database()

  conn = sqlite3.connect(str(db_path))
  cursor = conn.cursor()
  cursor.execute("SELECT version FROM _schema_version ORDER BY version DESC LIMIT 1")
  assert cursor.fetchone()[0] == CURRENT_SCHEMA_VERSION
  cursor.execute("PRAGMA table_info(problem_metadata)")
  columns = {row[1] for row in cursor.fetchall()}
  conn.close()
  assert "ai_grading_notes" in columns

  backup_files = list(backup_dir.glob("legacy_v23.db.v23.bak-*"))
  assert backup_files, "Expected pre-migration backup to be created"


def test_migration_preflight_fails_with_insufficient_backup_space(monkeypatch,
                                                                  tmp_path):
  db_path = tmp_path / "legacy_v23_nospace.db"
  backup_dir = tmp_path / "backups"
  _create_v23_database(db_path)

  monkeypatch.setenv("GRADING_DB_PATH", str(db_path))
  monkeypatch.setenv("GRADING_DB_MIGRATION_BACKUP_DIR", str(backup_dir))
  monkeypatch.setenv("GRADING_DB_CREATE_MIGRATION_BACKUP", "true")
  monkeypatch.setattr(
    db_module.shutil, "disk_usage",
    lambda _path: SimpleNamespace(total=1, used=1, free=0))

  with pytest.raises(RuntimeError, match="Insufficient disk space"):
    init_database()


def test_init_database_rejects_newer_schema_version(monkeypatch, tmp_path):
  db_path = tmp_path / "future.db"
  _create_future_version_database(db_path)
  monkeypatch.setenv("GRADING_DB_PATH", str(db_path))

  with pytest.raises(RuntimeError, match="newer than supported"):
    init_database()


def test_pre_migration_backup_includes_uncheckpointed_wal_changes(monkeypatch,
                                                                  tmp_path):
  db_path = tmp_path / "wal_source.db"
  backup_dir = tmp_path / "backups"
  conn = sqlite3.connect(str(db_path))
  try:
    cursor = conn.cursor()
    assert cursor.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
    cursor.execute("PRAGMA wal_autocheckpoint=0")
    cursor.execute("CREATE TABLE wal_state_test (value INTEGER)")
    conn.commit()
    cursor.execute("INSERT INTO wal_state_test (value) VALUES (42)")
    conn.commit()

    wal_path = Path(f"{db_path}-wal")
    assert wal_path.exists()
    assert wal_path.stat().st_size > 0

    monkeypatch.setenv("GRADING_DB_MIGRATION_BACKUP_DIR", str(backup_dir))
    monkeypatch.setenv("GRADING_DB_CREATE_MIGRATION_BACKUP", "true")

    backup_path = db_module._create_pre_migration_backup(db_path, from_version=23)
    assert backup_path is not None
    assert backup_path.exists()

    backup_conn = sqlite3.connect(str(backup_path))
    try:
      backup_count = backup_conn.execute(
        "SELECT COUNT(*) FROM wal_state_test").fetchone()[0]
      assert backup_count == 1
    finally:
      backup_conn.close()

    # Document expected failure mode of file-only copy when WAL contains commits.
    plain_copy_path = tmp_path / "plain_copy.db"
    shutil.copy2(str(db_path), str(plain_copy_path))
    plain_copy_conn = sqlite3.connect(str(plain_copy_path))
    try:
      with pytest.raises(sqlite3.OperationalError):
        plain_copy_conn.execute("SELECT COUNT(*) FROM wal_state_test").fetchone()
    finally:
      plain_copy_conn.close()
  finally:
    conn.close()
