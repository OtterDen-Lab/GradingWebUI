"""
Basic API tests to verify setup.
"""
import asyncio
import csv
import io
import os
from pathlib import Path
import zipfile
from concurrent.futures import ThreadPoolExecutor
import pytest
from fastapi.testclient import TestClient
from grading_web_ui.web_api.main import app
from grading_web_ui.web_api.database import get_db_connection
from grading_web_ui.web_api.repositories.session_repository import SessionRepository
from grading_web_ui.web_api import workflow_locks


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
  assert "requests_total" in data
  assert "requests_5xx_total" in data


def test_metrics_endpoint(client):
  """Metrics endpoint should return runtime counters for instructors."""
  response = client.get("/api/metrics")
  assert response.status_code == 200
  data = response.json()
  assert "requests_total" in data
  assert "status_buckets" in data
  assert "route_totals" in data


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


def test_session_export_includes_stats_metadata_and_tags(client):
  """Session export should include problem_stats, problem_metadata, and feedback_tags."""
  session_response = client.post("/api/sessions",
                                 json={
                                   "course_id": 12345,
                                   "assignment_id": 67890,
                                   "assignment_name": "Export Coverage Test"
                                 })
  assert session_response.status_code == 200
  session_id = session_response.json()["id"]

  with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
      """
      INSERT INTO submissions (session_id, document_id, page_mappings, student_name, file_hash)
      VALUES (?, ?, ?, ?, ?)
      """, (session_id, 1, "{}", "Student One", "hash-export-1"))
    submission_id = cursor.lastrowid

    cursor.execute(
      """
      INSERT INTO problems (session_id, submission_id, problem_number, graded, score, feedback)
      VALUES (?, ?, ?, ?, ?, ?)
      """, (session_id, submission_id, 1, 1, 4.0, "nice work"))

    cursor.execute(
      """
      INSERT INTO problem_stats (session_id, problem_number, avg_score, min_score, max_score, num_graded, num_total)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      """, (session_id, 1, 4.0, 4.0, 4.0, 1, 1))

    cursor.execute(
      """
      INSERT INTO problem_metadata (session_id, problem_number, max_points, default_feedback, default_feedback_threshold)
      VALUES (?, ?, ?, ?, ?)
      """, (session_id, 1, 5.0, "default comment", 90.0))

    cursor.execute(
      """
      INSERT INTO feedback_tags (session_id, problem_number, short_name, comment_text, use_count)
      VALUES (?, ?, ?, ?, ?)
      """, (session_id, 1, "tag-a", "comment a", 2))

  export_response = client.get(f"/api/sessions/{session_id}/export")
  assert export_response.status_code == 200
  payload = export_response.json()
  assert payload["session"]["id"] == session_id
  assert len(payload["problem_stats"]) == 1
  assert payload["problem_stats"][0]["problem_number"] == 1
  assert len(payload["problem_metadata"]) == 1
  assert payload["problem_metadata"][0]["problem_number"] == 1
  assert len(payload["feedback_tags"]) == 1
  assert payload["feedback_tags"][0]["short_name"] == "tag-a"


def test_session_compare_export_csv_contains_rows(client):
  """Comparison export should return CSV rows for matching file_hash/problem pairs."""
  session_one = client.post("/api/sessions",
                            json={
                              "course_id": 100,
                              "assignment_id": 200,
                              "assignment_name": "Compare One"
                            })
  session_two = client.post("/api/sessions",
                            json={
                              "course_id": 101,
                              "assignment_id": 201,
                              "assignment_name": "Compare Two"
                            })
  assert session_one.status_code == 200
  assert session_two.status_code == 200
  session_one_id = session_one.json()["id"]
  session_two_id = session_two.json()["id"]

  with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
      """
      INSERT INTO submissions (session_id, document_id, page_mappings, student_name, file_hash)
      VALUES (?, ?, ?, ?, ?)
      """, (session_one_id, 1, "{}", "Student One", "shared-hash-1"))
    sub_one = cursor.lastrowid
    cursor.execute(
      """
      INSERT INTO submissions (session_id, document_id, page_mappings, student_name, file_hash)
      VALUES (?, ?, ?, ?, ?)
      """, (session_two_id, 1, "{}", "Student One", "shared-hash-1"))
    sub_two = cursor.lastrowid

    cursor.execute(
      """
      INSERT INTO problems (session_id, submission_id, problem_number, graded, score, feedback, is_blank)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      """, (session_one_id, sub_one, 1, 1, 3.0, "ok", 0))
    cursor.execute(
      """
      INSERT INTO problems (session_id, submission_id, problem_number, graded, score, feedback, is_blank)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      """, (session_two_id, sub_two, 1, 1, 4.0, "better", 0))

  response = client.post("/api/sessions/compare-export",
                         json={
                           "session_ids": [session_one_id, session_two_id]
                         })
  assert response.status_code == 200

  rows = list(csv.reader(io.StringIO(response.text)))
  assert len(rows) >= 3
  assert rows[0][0] == "file_hash"
  assert rows[2][0] == "shared-hash-1"
  assert rows[2][1] == "1"


def test_finalize_rejects_when_finalize_lock_active(client):
  """Finalize endpoint should reject duplicate finalize runs for the same session."""
  session_response = client.post("/api/sessions",
                                 json={
                                   "course_id": 12345,
                                   "assignment_id": 67890,
                                   "assignment_name": "Finalize Lock Test"
                                 })
  assert session_response.status_code == 200
  session_id = session_response.json()["id"]

  assert workflow_locks.acquire("finalize", session_id) is True
  try:
    response = client.post(f"/api/finalize/{session_id}/finalize")
    assert response.status_code == 409
    assert "already running" in response.json()["detail"].lower()
  finally:
    workflow_locks.release("finalize", session_id)


def test_autograde_all_rejects_when_finalize_lock_active(client):
  """Autograde endpoint should reject while finalization lock is active for session."""
  session_response = client.post("/api/sessions",
                                 json={
                                   "course_id": 12345,
                                   "assignment_id": 67890,
                                   "assignment_name": "Autograde Lock Test"
                                 })
  assert session_response.status_code == 200
  session_id = session_response.json()["id"]

  with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
      """
      INSERT INTO submissions (session_id, document_id, page_mappings, student_name, file_hash)
      VALUES (?, ?, ?, ?, ?)
      """, (session_id, 1, "{}", "Student Lock", "lock-hash"))
    submission_id = cursor.lastrowid
    cursor.execute(
      """
      INSERT INTO problems (session_id, submission_id, problem_number, graded, is_blank)
      VALUES (?, ?, ?, ?, ?)
      """, (session_id, submission_id, 1, 0, 0))

  assert workflow_locks.acquire("finalize", session_id) is True
  try:
    response = client.post(f"/api/ai-grader/{session_id}/autograde-all",
                           json={
                             "mode": "image-only",
                             "settings": {
                               "batch_size": "small",
                               "image_quality": "medium",
                               "include_answer": False,
                               "include_default_feedback": False,
                               "auto_accept": False,
                               "dry_run": True
                             }
                           })
    assert response.status_code == 409
    assert "finalization is in progress" in response.json()["detail"].lower()
  finally:
    workflow_locks.release("finalize", session_id)


def test_finalize_concurrent_requests_allow_only_one_start(client, monkeypatch):
  """Concurrent finalize starts should allow one start and reject the overlap."""
  from grading_web_ui.web_api.routes import finalize as finalize_routes

  session_response = client.post("/api/sessions",
                                 json={
                                   "course_id": 12345,
                                   "assignment_id": 67890,
                                   "assignment_name": "Finalize Concurrency Test"
                                 })
  assert session_response.status_code == 200
  session_id = session_response.json()["id"]

  with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
      """
      INSERT INTO submissions (session_id, document_id, page_mappings, student_name, canvas_user_id)
      VALUES (?, ?, ?, ?, ?)
      """, (session_id, 1, "{}", "Student Concurrent", 9002))
    submission_id = cursor.lastrowid
    cursor.execute(
      """
      INSERT INTO problems (session_id, submission_id, problem_number, graded, score, is_blank)
      VALUES (?, ?, ?, ?, ?, ?)
      """, (session_id, submission_id, 1, 1, 5.0, 0))

  async def slow_fake_run_finalization(target_session_id: int, stream_id: str):
    await asyncio.sleep(0.25)
    workflow_locks.release("finalize", target_session_id)

  monkeypatch.setattr(finalize_routes, "run_finalization",
                      slow_fake_run_finalization)

  def start_finalize() -> int:
    return client.post(f"/api/finalize/{session_id}/finalize").status_code

  with ThreadPoolExecutor(max_workers=2) as executor:
    statuses = list(executor.map(lambda _: start_finalize(), range(2)))

  assert sorted(statuses) == [200, 409]


def test_finalize_lifecycle_starts_and_completes_with_background_task(client,
                                                                      monkeypatch):
  """Finalize endpoint should start and background task should drive status completion."""
  from grading_web_ui.web_api.routes import finalize as finalize_routes
  from grading_web_ui.web_api.repositories.session_repository import SessionRepository
  from grading_web_ui.web_api.domain.common import SessionStatus as DomainSessionStatus

  session_response = client.post("/api/sessions",
                                 json={
                                   "course_id": 12345,
                                   "assignment_id": 67890,
                                   "assignment_name": "Finalize Lifecycle Test"
                                 })
  assert session_response.status_code == 200
  session_id = session_response.json()["id"]

  with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
      """
      INSERT INTO submissions (session_id, document_id, page_mappings, student_name, canvas_user_id)
      VALUES (?, ?, ?, ?, ?)
      """, (session_id, 1, "{}", "Student Final", 9001))
    submission_id = cursor.lastrowid
    cursor.execute(
      """
      INSERT INTO problems (session_id, submission_id, problem_number, graded, score, is_blank)
      VALUES (?, ?, ?, ?, ?, ?)
      """, (session_id, submission_id, 1, 1, 4.0, 0))

  async def fake_run_finalization(target_session_id: int, stream_id: str):
    repo = SessionRepository()
    repo.update_status(target_session_id, DomainSessionStatus.FINALIZED,
                       "Finalized in test")
    workflow_locks.release("finalize", target_session_id)

  monkeypatch.setattr(finalize_routes, "run_finalization", fake_run_finalization)

  response = client.post(f"/api/finalize/{session_id}/finalize")
  assert response.status_code == 200
  assert response.json()["status"] == "started"

  status_response = client.get(f"/api/finalize/{session_id}/finalization-status")
  assert status_response.status_code == 200
  assert status_response.json()["status"] in ("finalizing", "finalized")

  # In TestClient, background tasks should complete by response return.
  final_status = client.get(
    f"/api/finalize/{session_id}/finalization-status").json()["status"]
  assert final_status == "finalized"
  assert workflow_locks.is_active("finalize", session_id) is False


def test_autograde_image_lifecycle_starts_job(client, monkeypatch):
  """Autograde image endpoint should start a background job when setup is valid."""
  from grading_web_ui.web_api.routes import ai_grader as ai_routes

  session_response = client.post("/api/sessions",
                                 json={
                                   "course_id": 12345,
                                   "assignment_id": 67890,
                                   "assignment_name": "Autograde Lifecycle Test"
                                 })
  assert session_response.status_code == 200
  session_id = session_response.json()["id"]

  with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
      """
      INSERT INTO submissions (session_id, document_id, page_mappings, student_name, file_hash)
      VALUES (?, ?, ?, ?, ?)
      """, (session_id, 1, "{}", "Student Auto", "autograde-hash"))
    submission_id = cursor.lastrowid
    cursor.execute(
      """
      INSERT INTO problems (session_id, submission_id, problem_number, graded, is_blank)
      VALUES (?, ?, ?, ?, ?)
      """, (session_id, submission_id, 1, 0, 0))

  async def fake_run_autograding_image(target_session_id: int, problem_number: int,
                                       settings: dict, stream_id: str):
    workflow_locks.release("autograde", target_session_id)

  monkeypatch.setattr(ai_routes, "run_autograding_image",
                      fake_run_autograding_image)

  response = client.post(f"/api/ai-grader/{session_id}/autograde",
                         json={
                           "mode": "image-only",
                           "problem_number": 1,
                           "settings": {
                             "batch_size": "small",
                             "image_quality": "medium",
                             "include_answer": False,
                             "include_default_feedback": False,
                             "auto_accept": False,
                             "dry_run": True
                           }
                         })
  assert response.status_code == 200
  assert response.json()["status"] == "started"
  assert workflow_locks.is_active("autograde", session_id) is False


def test_run_autograding_image_releases_lock_on_failure(monkeypatch):
  """AI autograde worker should release workflow lock and emit error event on exceptions."""
  from grading_web_ui.web_api.routes import ai_grader as ai_routes

  session_id = 9898
  assert workflow_locks.acquire("autograde", session_id) is True
  captured_events = []

  class FailingAIGrader:

    def autograde_problem_image_only(self, *args, **kwargs):
      raise RuntimeError("forced test failure")

  async def capture_event(stream_id: str, event_type: str, data: dict):
    captured_events.append((stream_id, event_type, data))

  monkeypatch.setattr(ai_routes, "AIGraderService", lambda: FailingAIGrader())
  monkeypatch.setattr(ai_routes.sse, "send_event", capture_event)

  asyncio.run(
    ai_routes.run_autograding_image(session_id, 1, {
      "auto_accept": False,
      "dry_run": True
    }, "test_stream"))

  assert workflow_locks.is_active("autograde", session_id) is False
  assert any(event_type == "error" for _, event_type, _ in captured_events)
