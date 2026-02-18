"""
Basic API tests to verify setup.
"""
import io
import os
from pathlib import Path
import zipfile
import pytest
from fastapi.testclient import TestClient
from grading_web_ui.web_api.main import app
from grading_web_ui.web_api.database import get_db_connection
from grading_web_ui.web_api.repositories.session_repository import SessionRepository


@pytest.fixture
def client(tmp_path, monkeypatch):
  """Create authenticated test client with isolated DB"""
  db_path = tmp_path / "test_api.db"
  monkeypatch.setenv("GRADING_DB_PATH", str(db_path))
  monkeypatch.setenv("GRADING_BOOTSTRAP_ADMIN_USERNAME", "admin")
  monkeypatch.setenv("GRADING_BOOTSTRAP_ADMIN_PASSWORD", "changeme123")
  monkeypatch.setenv("GRADING_BOOTSTRAP_ADMIN_EMAIL", "admin@example.com")
  monkeypatch.setenv("AUTH_COOKIE_SECURE", "false")
  monkeypatch.setenv("CANVAS_API_URL", "https://canvas.example.test")
  monkeypatch.setenv("CANVAS_API_KEY", "test-canvas-key")
  monkeypatch.setenv("GRADING_STRICT_STARTUP_CONFIG", "true")

  with TestClient(app) as test_client:
    login_response = test_client.post(
      "/api/auth/login",
      json={
        "username": "admin",
        "password": "changeme123"
      }
    )
    assert login_response.status_code == 200
    yield test_client


def test_no_default_admin_without_bootstrap(tmp_path, monkeypatch):
  """Without bootstrap env vars, database should not include known default admin."""
  db_path = tmp_path / "test_no_default_admin.db"
  monkeypatch.setenv("GRADING_DB_PATH", str(db_path))
  monkeypatch.delenv("GRADING_BOOTSTRAP_ADMIN_USERNAME", raising=False)
  monkeypatch.delenv("GRADING_BOOTSTRAP_ADMIN_PASSWORD", raising=False)
  monkeypatch.delenv("GRADING_BOOTSTRAP_ADMIN_EMAIL", raising=False)
  monkeypatch.setenv("AUTH_COOKIE_SECURE", "false")
  monkeypatch.setenv("CANVAS_API_URL", "https://canvas.example.test")
  monkeypatch.setenv("CANVAS_API_KEY", "test-canvas-key")
  monkeypatch.setenv("GRADING_STRICT_STARTUP_CONFIG", "true")

  with TestClient(app) as test_client:
    login_response = test_client.post(
      "/api/auth/login",
      json={
        "username": "admin",
        "password": "changeme123"
      }
    )
  assert login_response.status_code == 401


def test_login_sets_secure_cookie_when_enabled(tmp_path, monkeypatch):
  """Auth cookie should include Secure flag when AUTH_COOKIE_SECURE=true."""
  db_path = tmp_path / "test_cookie_secure.db"
  monkeypatch.setenv("GRADING_DB_PATH", str(db_path))
  monkeypatch.setenv("GRADING_BOOTSTRAP_ADMIN_USERNAME", "admin")
  monkeypatch.setenv("GRADING_BOOTSTRAP_ADMIN_PASSWORD", "changeme123")
  monkeypatch.setenv("GRADING_BOOTSTRAP_ADMIN_EMAIL", "admin@example.com")
  monkeypatch.setenv("AUTH_COOKIE_SECURE", "true")
  monkeypatch.setenv("CANVAS_API_URL", "https://canvas.example.test")
  monkeypatch.setenv("CANVAS_API_KEY", "test-canvas-key")
  monkeypatch.setenv("GRADING_STRICT_STARTUP_CONFIG", "true")

  with TestClient(app) as test_client:
    response = test_client.post(
      "/api/auth/login",
      json={
        "username": "admin",
        "password": "changeme123"
      }
    )

  assert response.status_code == 200
  set_cookie = response.headers.get("set-cookie", "")
  assert "Secure" in set_cookie


def test_health_check(client):
  """Test health endpoint"""
  response = client.get("/api/health")
  assert response.status_code == 200
  data = response.json()
  assert data["status"] == "healthy"
  assert "version" in data


def test_create_session(client):
  """Test session creation"""
  response = client.post("/api/sessions",
                         json={
                           "course_id": 12345,
                           "assignment_id": 67890,
                           "assignment_name": "Test Exam"
                         })
  assert response.status_code == 200
  data = response.json()
  assert data["assignment_name"] == "Test Exam"
  assert data["status"] == "preprocessing"


def test_get_nonexistent_session(client):
  """Test getting non-existent session returns 404"""
  response = client.get("/api/sessions/99999")
  assert response.status_code == 404


def test_list_sessions(client):
  """Test listing sessions"""
  # Create a session first
  client.post("/api/sessions",
              json={
                "course_id": 12345,
                "assignment_id": 67890,
                "assignment_name": "Test Exam"
              })

  # List sessions
  response = client.get("/api/sessions")
  assert response.status_code == 200
  data = response.json()
  assert isinstance(data, list)
  assert len(data) > 0


def test_upload_quiz_yaml_for_session(client):
  """Test uploading quiz YAML directly to session metadata"""
  session_response = client.post("/api/sessions",
                                 json={
                                   "course_id": 12345,
                                   "assignment_id": 67890,
                                   "assignment_name": "YAML Test"
                                 })
  assert session_response.status_code == 200
  session_id = session_response.json()["id"]

  yaml_text = "yaml_id: demo-exam\nquestions: []\n"
  upload_response = client.post(
    f"/api/sessions/{session_id}/quiz-yaml",
    files={
      "yaml_file": ("exam.yaml", yaml_text, "application/x-yaml")
    })

  assert upload_response.status_code == 200
  upload_data = upload_response.json()
  assert upload_data["status"] == "uploaded"
  assert upload_data["filename"] == "exam.yaml"
  assert upload_data["yaml_ids"] == ["demo-exam"]

  status_response = client.get(f"/api/sessions/{session_id}/quiz-yaml")
  assert status_response.status_code == 200
  status_data = status_response.json()
  assert status_data["has_quiz_yaml"] is True
  assert status_data["filename"] == "exam.yaml"


def test_upload_yaml_only_via_upload_endpoint(client):
  """Test YAML-only upload through exam upload endpoint."""
  session_response = client.post("/api/sessions",
                                 json={
                                   "course_id": 12345,
                                   "assignment_id": 67890,
                                   "assignment_name": "Upload YAML Test"
                                 })
  assert session_response.status_code == 200
  session_id = session_response.json()["id"]

  yaml_text = "yaml_id: upload-exam\nquestions: []\n"
  response = client.post(
    f"/api/uploads/{session_id}/upload",
    files=[("files", ("exam.yaml", yaml_text, "application/x-yaml"))])
  assert response.status_code == 200
  data = response.json()
  assert data["status"] == "yaml_uploaded"
  assert data["files_uploaded"] == 0

  status_response = client.get(f"/api/sessions/{session_id}/quiz-yaml")
  assert status_response.status_code == 200
  assert status_response.json()["has_quiz_yaml"] is True


def test_upload_filename_sanitized_for_storage(client):
  """Upload path traversal names should be sanitized before filesystem writes."""
  session_response = client.post("/api/sessions",
                                 json={
                                   "course_id": 12345,
                                   "assignment_id": 67890,
                                   "assignment_name": "Filename Sanitization Test"
                                 })
  assert session_response.status_code == 200
  session_id = session_response.json()["id"]

  pdf_content = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
  response = client.post(
    f"/api/uploads/{session_id}/upload",
    files=[("files", ("../../../../exam.pdf", pdf_content, "application/pdf"))])

  assert response.status_code == 200

  session_repo = SessionRepository()
  metadata = session_repo.get_metadata(session_id)
  assert metadata is not None
  stored_paths = [Path(path_str) for path_str in metadata["file_paths"]]
  assert stored_paths
  assert all(".." not in path.parts for path in stored_paths)
  assert all(path.name == "exam.pdf" for path in stored_paths)


def test_zip_upload_rejects_unsafe_paths(client):
  """Zip archives with traversal entries should be rejected."""
  session_response = client.post("/api/sessions",
                                 json={
                                   "course_id": 12345,
                                   "assignment_id": 67890,
                                   "assignment_name": "Zip Safety Test"
                                 })
  assert session_response.status_code == 200
  session_id = session_response.json()["id"]

  zip_buffer = io.BytesIO()
  with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
    zip_file.writestr("../evil.pdf", b"%PDF-1.4\n")
  zip_buffer.seek(0)

  response = client.post(
    f"/api/uploads/{session_id}/upload",
    files=[("files", ("unsafe.zip", zip_buffer.getvalue(), "application/zip"))])

  assert response.status_code == 400
  assert "unsafe file paths" in response.json()["detail"]


def test_delete_session_with_dependent_rows(client):
  """Deleting a session should cascade through dependent rows without FK errors."""
  session_response = client.post("/api/sessions",
                                 json={
                                   "course_id": 12345,
                                   "assignment_id": 67890,
                                   "assignment_name": "Delete Cascade Test"
                                 })
  assert session_response.status_code == 200
  session_id = session_response.json()["id"]

  with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
      """
      INSERT INTO submissions (session_id, document_id, page_mappings, student_name)
      VALUES (?, ?, ?, ?)
      """, (session_id, 1, "{}", "Student One"))
    submission_id = cursor.lastrowid

    cursor.execute(
      """
      INSERT INTO problems (session_id, submission_id, problem_number, graded)
      VALUES (?, ?, ?, ?)
      """, (session_id, submission_id, 1, 0))

    cursor.execute(
      """
      INSERT INTO problem_metadata (session_id, problem_number, max_points)
      VALUES (?, ?, ?)
      """, (session_id, 1, 10.0))

    cursor.execute(
      """
      INSERT INTO problem_stats (session_id, problem_number, num_total, num_graded)
      VALUES (?, ?, ?, ?)
      """, (session_id, 1, 1, 0))

    cursor.execute(
      """
      INSERT INTO feedback_tags (session_id, problem_number, short_name, comment_text)
      VALUES (?, ?, ?, ?)
      """, (session_id, 1, "tag1", "comment"))

    cursor.execute("SELECT id FROM users WHERE username = 'admin' LIMIT 1")
    admin_row = cursor.fetchone()
    assert admin_row is not None
    cursor.execute(
      """
      INSERT INTO session_assignments (session_id, user_id, assigned_by)
      VALUES (?, ?, ?)
      """, (session_id, admin_row["id"], admin_row["id"]))

  delete_response = client.delete(f"/api/sessions/{session_id}")
  assert delete_response.status_code == 200
  assert delete_response.json()["status"] == "deleted"

  with get_db_connection() as conn:
    cursor = conn.cursor()
    table_filters = {
      "grading_sessions": "id = ?",
      "submissions": "session_id = ?",
      "problems": "session_id = ?",
      "problem_metadata": "session_id = ?",
      "problem_stats": "session_id = ?",
      "feedback_tags": "session_id = ?",
      "session_assignments": "session_id = ?",
    }
    for table, where_clause in table_filters.items():
      cursor.execute(
        f"SELECT COUNT(*) AS count FROM {table} WHERE {where_clause}",
        (session_id, ))
      assert cursor.fetchone()["count"] == 0


def test_regenerate_answer_avoids_signature_mismatch_error(client):
  """Answer regeneration should not fail with unexpected keyword-argument errors."""
  session_response = client.post("/api/sessions",
                                 json={
                                   "course_id": 12345,
                                   "assignment_id": 67890,
                                   "assignment_name": "Regeneration Signature Test"
                                 })
  assert session_response.status_code == 200
  session_id = session_response.json()["id"]

  with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
      """
      INSERT INTO submissions (session_id, document_id, page_mappings, student_name)
      VALUES (?, ?, ?, ?)
      """, (session_id, 1, "{}", "Student One"))
    submission_id = cursor.lastrowid

    cursor.execute(
      """
      INSERT INTO problems (session_id, submission_id, problem_number, graded, qr_encrypted_data, max_points)
      VALUES (?, ?, ?, ?, ?, ?)
      """, (session_id, submission_id, 1, 0, "invalid-encrypted-payload", 5.0))
    problem_id = cursor.lastrowid

  response = client.get(f"/api/problems/{problem_id}/regenerate-answer")
  assert response.status_code in (400, 500)
  detail = response.json().get("detail", "")
  assert "unexpected keyword argument" not in detail


def test_set_encryption_key_uses_runtime_store_not_env(client):
  """Setting runtime encryption key should not mutate QUIZ_ENCRYPTION_KEY env var."""
  from grading_web_ui.web_api.services.quiz_encryption import (
    get_runtime_encryption_key,
    clear_runtime_encryption_key,
  )

  original_env = os.environ.get("QUIZ_ENCRYPTION_KEY")
  os.environ.pop("QUIZ_ENCRYPTION_KEY", None)
  clear_runtime_encryption_key()
  try:
    response = client.post("/api/sessions/encryption-key/set",
                           params={"encryption_key": "runtime-key-123"})
    assert response.status_code == 200
    assert get_runtime_encryption_key() == b"runtime-key-123"
    assert os.environ.get("QUIZ_ENCRYPTION_KEY") is None
  finally:
    clear_runtime_encryption_key()
    if original_env is not None:
      os.environ["QUIZ_ENCRYPTION_KEY"] = original_env


def test_encryption_key_test_endpoint_handles_invalid_payload(client):
  """Key test endpoint should return failure payload for invalid data, not import errors."""
  response = client.post("/api/sessions/encryption-key/test",
                         params={
                           "encrypted_data": "not-valid",
                           "encryption_key": "runtime-key-123"
                         })
  assert response.status_code == 200
  assert response.json().get("status") == "failed"
