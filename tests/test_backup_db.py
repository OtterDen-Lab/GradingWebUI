import sqlite3

from grading_web_ui.web_api.database import (create_database_backup,
                                              create_schema)


def test_create_database_backup_includes_wal_data_and_manifest_fields(tmp_path):
  source_path = tmp_path / "grading.db"
  destination_path = tmp_path / "backups" / "grading-backup.db"
  source_conn = sqlite3.connect(source_path)
  source_conn.execute("PRAGMA journal_mode = WAL")
  create_schema(source_conn.cursor())
  source_conn.execute(
    "INSERT INTO grading_sessions (assignment_id, assignment_name, course_id, status) "
    "VALUES (1, 'Backup test', 2, 'created')")
  source_conn.commit()

  result = create_database_backup(source_path, destination_path)

  assert destination_path.exists()
  assert result["schema_version"] > 0
  assert len(result["sha256"]) == 64
  assert result["size_bytes"] == destination_path.stat().st_size
  backup_conn = sqlite3.connect(destination_path)
  try:
    assert backup_conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert backup_conn.execute("SELECT count(*) FROM grading_sessions").fetchone()[0] == 1
  finally:
    backup_conn.close()
    source_conn.close()


def test_create_database_backup_refuses_to_overwrite(tmp_path):
  source_path = tmp_path / "grading.db"
  destination_path = tmp_path / "backup.db"
  sqlite3.connect(source_path).close()
  destination_path.write_bytes(b"existing backup")

  try:
    create_database_backup(source_path, destination_path)
  except FileExistsError:
    pass
  else:
    raise AssertionError("Expected backup creation to refuse an existing file")
