"""
Basic API tests to verify setup.
"""
import asyncio
import csv
import io
import os
import time
from pathlib import Path
import zipfile
from concurrent.futures import ThreadPoolExecutor
import pytest
from fastapi.testclient import TestClient
from grading_web_ui.web_api.main import app
from grading_web_ui.web_api.database import get_db_connection
from grading_web_ui.web_api.repositories.session_repository import SessionRepository
from grading_web_ui.web_api.repositories.problem_metadata_repository import ProblemMetadataRepository
from grading_web_ui.web_api.repositories.feedback_tag_repository import FeedbackTagRepository
from grading_web_ui.web_api.repositories.problem_stats_repository import ProblemStatsRepository
from grading_web_ui.web_api.repositories.session_assignment_repository import SessionAssignmentRepository
from grading_web_ui.web_api.repositories import with_transaction
from grading_web_ui.web_api.domain.submission import Submission
from grading_web_ui.web_api.domain.problem import Problem
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


def create_test_session(client, assignment_name: str = "Test Session") -> int:
  """Create a session through API and return session_id."""
  response = client.post("/api/sessions",
                         json={
                           "course_id": 12345,
                           "assignment_id": 67890,
                           "assignment_name": assignment_name
                         })
  assert response.status_code == 200
  return response.json()["id"]


def seed_submission_with_problem(session_id: int,
                                 *,
                                 document_id: int = 1,
                                 student_name: str = "Student One",
                                 canvas_user_id: int | None = None,
                                 file_hash: str | None = None,
                                 problem_number: int = 1,
                                 graded: bool = False,
                                 score: float | None = None,
                                 feedback: str | None = None,
                                 is_blank: bool = False,
                                 blank_method: str | None = None,
                                 blank_reasoning: str | None = None,
                                 qr_encrypted_data: str | None = None,
                                 max_points: float | None = None) -> tuple[int, int]:
  """Create one submission and one problem via repositories."""
  with with_transaction() as repos:
    created_submission = repos.submissions.create(
      Submission(
        id=0,
        session_id=session_id,
        document_id=document_id,
        approximate_name=None,
        name_image_data=None,
        student_name=student_name,
        display_name=student_name,
        canvas_user_id=canvas_user_id,
        page_mappings={},
        file_hash=file_hash,
        original_filename=None,
        exam_pdf_data=None,
      ))
    created_problem = repos.problems.create(
      Problem(
        id=0,
        session_id=session_id,
        submission_id=created_submission.id,
        problem_number=problem_number,
        graded=graded,
        score=score,
        feedback=feedback,
        is_blank=is_blank,
        blank_method=blank_method,
        blank_reasoning=blank_reasoning,
        qr_encrypted_data=qr_encrypted_data,
        max_points=max_points,
      ))
    return created_submission.id, created_problem.id


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
  session_id = create_test_session(client, "Delete Cascade Test")
  seed_submission_with_problem(session_id, graded=False)
  ProblemMetadataRepository().bulk_insert_for_session(
    session_id,
    [{
      "problem_number": 1,
      "max_points": 10.0
    }],
  )
  ProblemStatsRepository().bulk_insert_for_session(
    session_id,
    [{
      "problem_number": 1,
      "num_total": 1,
      "num_graded": 0
    }],
  )
  FeedbackTagRepository().bulk_insert_for_session(
    session_id,
    [{
      "problem_number": 1,
      "short_name": "tag1",
      "comment_text": "comment"
    }],
  )

  with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = 'admin' LIMIT 1")
    admin_row = cursor.fetchone()
    assert admin_row is not None
    admin_id = admin_row["id"]

  SessionAssignmentRepository().assign_user_to_session(session_id, admin_id,
                                                       admin_id)

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
  session_id = create_test_session(client, "Regeneration Signature Test")
  _, problem_id = seed_submission_with_problem(
    session_id,
    qr_encrypted_data="invalid-encrypted-payload",
    max_points=5.0,
  )

  response = client.get(f"/api/problems/{problem_id}/regenerate-answer")
  assert response.status_code in (400, 500)
  detail = response.json().get("detail", "")
  assert "unexpected keyword argument" not in detail


def test_prefetch_regeneration_returns_no_qr_data_when_empty(client):
  """Session-level regeneration prefetch should no-op when no QR-backed problems exist."""
  session_id = create_test_session(client, "Regeneration Prefetch Empty")

  response = client.post(f"/api/problems/session/{session_id}/prefetch-regeneration")
  assert response.status_code == 200
  payload = response.json()
  assert payload["status"] == "no_qr_data"
  assert payload["total_qr_problems"] == 0


def test_prefetch_regeneration_warms_cache_for_problem(client, monkeypatch):
  """Session-level regeneration prefetch should warm cache used by regenerate-answer."""
  from grading_web_ui.web_api.routes import problems as problems_routes

  session_id = create_test_session(client, "Regeneration Prefetch Warm")
  _, problem_id = seed_submission_with_problem(
    session_id,
    qr_encrypted_data="fake-encrypted",
    max_points=5.0,
  )

  call_count = {"count": 0}

  def fake_regenerate(**kwargs):
    call_count["count"] += 1
    return {
      "question_type": "numeric",
      "seed": 123,
      "version": "test",
      "answer_objects": [{"value": "42"}],
      "explanation_markdown": "Test explanation"
    }

  monkeypatch.setattr(
    problems_routes,
    "regenerate_from_encrypted_compat",
    fake_regenerate
  )

  prefetch_response = client.post(
    f"/api/problems/session/{session_id}/prefetch-regeneration"
  )
  assert prefetch_response.status_code == 200
  assert prefetch_response.json()["status"] == "started"

  for _ in range(40):
    if call_count["count"] >= 1:
      break
    time.sleep(0.05)
  assert call_count["count"] >= 1

  call_count_before = call_count["count"]
  regen_response = client.get(f"/api/problems/{problem_id}/regenerate-answer")
  assert regen_response.status_code == 200
  regen_payload = regen_response.json()
  assert regen_payload["answers"][0]["value"] == "42"
  assert regen_payload["explanation_markdown"] == "Test explanation"
  assert call_count["count"] == call_count_before


def test_manual_qr_payload_updates_problem_and_metadata(client):
  """Manual QR payload endpoint should persist max_points and encrypted data."""
  session_id = create_test_session(client, "Manual QR Payload")
  _, problem_id = seed_submission_with_problem(
    session_id,
    problem_number=4,
    qr_encrypted_data=None,
    max_points=None,
  )

  response = client.post(
    f"/api/problems/{problem_id}/manual-qr",
    json={"payload_text": '{"q": 4, "pts": 9, "s": "manual-encrypted"}'}
  )
  assert response.status_code == 200
  payload = response.json()
  assert payload["status"] == "success"
  assert payload["problem_number"] == 4
  assert payload["max_points"] == 9.0
  assert payload["has_qr_data"] is True

  with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
      "SELECT max_points, qr_encrypted_data FROM problems WHERE id = ?",
      (problem_id,)
    )
    row = cursor.fetchone()
    assert row["max_points"] == 9.0
    assert row["qr_encrypted_data"] == "manual-encrypted"

    cursor.execute(
      """
      SELECT max_points FROM problem_metadata
      WHERE session_id = ? AND problem_number = ?
      """,
      (session_id, 4)
    )
    metadata_row = cursor.fetchone()
    assert metadata_row is not None
    assert metadata_row["max_points"] == 9.0


def test_manual_qr_payload_rejects_question_number_mismatch(client):
  """Manual QR payload should reject JSON for the wrong problem number."""
  session_id = create_test_session(client, "Manual QR Mismatch")
  _, problem_id = seed_submission_with_problem(session_id, problem_number=2)

  response = client.post(
    f"/api/problems/{problem_id}/manual-qr",
    json={"payload_text": '{"q": 7, "pts": 8, "s": "wrong-problem"}'}
  )
  assert response.status_code == 400
  assert "does not match current problem number" in response.json()["detail"]


def test_subjective_settings_and_triage_flow(client, monkeypatch):
  """Subjective mode should support bucket edits and triage iteration."""
  from grading_web_ui.web_api.routes import problems as problems_routes

  monkeypatch.setattr(
    problems_routes,
    "get_problem_image_data",
    lambda problem, submission_repo=None: ""
  )

  session_id = create_test_session(client, "Subjective Flow")
  _, problem_id_a = seed_submission_with_problem(
    session_id, document_id=1, problem_number=3
  )
  _, problem_id_b = seed_submission_with_problem(
    session_id, document_id=2, problem_number=3
  )
  _, problem_id_c = seed_submission_with_problem(
    session_id, document_id=3, problem_number=3
  )

  update_response = client.put(
    f"/api/sessions/{session_id}/subjective-settings",
    json={
      "problem_number": 3,
      "grading_mode": "subjective",
      "buckets": [
        {"id": "perfect", "label": "Perfect"},
        {"id": "good", "label": "Good"},
      ]
    }
  )
  assert update_response.status_code == 200
  assert update_response.json()["grading_mode"] == "subjective"

  triage_response = client.post(
    f"/api/problems/{problem_id_a}/subjective-triage",
    json={"bucket_id": "perfect", "notes": "Strong work"}
  )
  assert triage_response.status_code == 200
  assert triage_response.json()["status"] == "triaged"

  next_response = client.get(f"/api/problems/{session_id}/3/next")
  assert next_response.status_code == 200
  next_payload = next_response.json()
  assert next_payload["grading_mode"] == "subjective"
  assert next_payload["id"] in {problem_id_b, problem_id_c}
  assert next_payload["id"] != problem_id_a

  previous_response = client.get(f"/api/problems/{session_id}/3/previous")
  assert previous_response.status_code == 200
  previous_payload = previous_response.json()
  assert previous_payload["id"] == problem_id_a
  assert previous_payload["subjective_bucket_id"] == "perfect"
  assert previous_payload["subjective_triaged"] is True

  # Removing a bucket that is actively used should fail
  invalid_update = client.put(
    f"/api/sessions/{session_id}/subjective-settings",
    json={
      "problem_number": 3,
      "grading_mode": "subjective",
      "buckets": [
        {"id": "good", "label": "Good"},
      ]
    }
  )
  assert invalid_update.status_code == 400
  assert "Cannot remove buckets with active triaged responses" in invalid_update.json()["detail"]

  # Adding buckets after triaging should work (bucket-count change support)
  valid_update = client.put(
    f"/api/sessions/{session_id}/subjective-settings",
    json={
      "problem_number": 3,
      "grading_mode": "subjective",
      "buckets": [
        {"id": "perfect", "label": "Perfect"},
        {"id": "excellent", "label": "Excellent"},
        {"id": "good", "label": "Good"},
      ]
    }
  )
  assert valid_update.status_code == 200
  updated_payload = valid_update.json()
  assert len(updated_payload["buckets"]) == 3


def test_subjective_finalize_applies_bucket_scores(client):
  """Finalizing subjective buckets should grade all triaged responses."""
  session_id = create_test_session(client, "Subjective Finalize")
  _, problem_id_a = seed_submission_with_problem(
    session_id, document_id=1, problem_number=4, max_points=8.0
  )
  _, problem_id_b = seed_submission_with_problem(
    session_id, document_id=2, problem_number=4, max_points=8.0
  )
  _, problem_id_c = seed_submission_with_problem(
    session_id, document_id=3, problem_number=4, max_points=8.0
  )

  mode_response = client.put(
    f"/api/sessions/{session_id}/subjective-settings",
    json={
      "problem_number": 4,
      "grading_mode": "subjective",
      "buckets": [
        {"id": "perfect", "label": "Perfect"},
        {"id": "good", "label": "Good"},
      ]
    }
  )
  assert mode_response.status_code == 200

  for problem_id, bucket_id in [
    (problem_id_a, "perfect"),
    (problem_id_b, "good"),
    (problem_id_c, "good"),
  ]:
    triage_response = client.post(
      f"/api/problems/{problem_id}/subjective-triage",
      json={"bucket_id": bucket_id}
    )
    assert triage_response.status_code == 200

  finalize_response = client.post(
    f"/api/sessions/{session_id}/subjective-finalize",
    json={
      "problem_number": 4,
      "bucket_scores": [
        {"bucket_id": "perfect", "score": 8.0, "feedback": "Excellent work"},
        {"bucket_id": "good", "score": 6.0, "feedback": "Mostly correct"},
      ]
    }
  )
  assert finalize_response.status_code == 200
  payload = finalize_response.json()
  assert payload["status"] == "finalized"
  assert payload["graded_count"] == 3
  assert payload["remaining_triaged"] == 0

  with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("""
      SELECT id, graded, score, feedback
      FROM problems
      WHERE session_id = ? AND problem_number = ?
    """, (session_id, 4))
    rows = {row["id"]: dict(row) for row in cursor.fetchall()}

    assert rows[problem_id_a]["graded"] == 1
    assert rows[problem_id_a]["score"] == 8.0
    assert rows[problem_id_a]["feedback"] == "Excellent work"

    assert rows[problem_id_b]["graded"] == 1
    assert rows[problem_id_b]["score"] == 6.0
    assert rows[problem_id_b]["feedback"] == "Mostly correct"

    assert rows[problem_id_c]["graded"] == 1
    assert rows[problem_id_c]["score"] == 6.0
    assert rows[problem_id_c]["feedback"] == "Mostly correct"

    cursor.execute("""
      SELECT COUNT(*) AS count
      FROM subjective_triage
      WHERE session_id = ? AND problem_number = ?
    """, (session_id, 4))
    assert cursor.fetchone()["count"] == 3


def test_subjective_reopen_restores_triaged_state(client):
  """Reopen should clear finalized grades and return to triaged/ungraded."""
  session_id = create_test_session(client, "Subjective Reopen")
  _, problem_id_a = seed_submission_with_problem(
    session_id, document_id=1, problem_number=5, max_points=8.0
  )
  _, problem_id_b = seed_submission_with_problem(
    session_id, document_id=2, problem_number=5, max_points=8.0
  )

  mode_response = client.put(
    f"/api/sessions/{session_id}/subjective-settings",
    json={
      "problem_number": 5,
      "grading_mode": "subjective",
      "buckets": [
        {"id": "good", "label": "Good"},
      ]
    }
  )
  assert mode_response.status_code == 200

  for problem_id in [problem_id_a, problem_id_b]:
    triage_response = client.post(
      f"/api/problems/{problem_id}/subjective-triage",
      json={"bucket_id": "good"}
    )
    assert triage_response.status_code == 200

  finalize_response = client.post(
    f"/api/sessions/{session_id}/subjective-finalize",
    json={
      "problem_number": 5,
      "bucket_scores": [
        {"bucket_id": "good", "score": 6.0}
      ]
    }
  )
  assert finalize_response.status_code == 200

  reopen_response = client.post(
    f"/api/sessions/{session_id}/subjective-reopen",
    json={"problem_number": 5}
  )
  assert reopen_response.status_code == 200
  payload = reopen_response.json()
  assert payload["status"] == "reopened"
  assert payload["reopened_count"] == 2
  assert payload["triaged_count"] == 2
  assert payload["graded_count"] == 0

  with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("""
      SELECT id, graded, score, feedback
      FROM problems
      WHERE session_id = ? AND problem_number = ?
      ORDER BY id
    """, (session_id, 5))
    rows = [dict(row) for row in cursor.fetchall()]
    assert len(rows) == 2
    for row in rows:
      assert row["graded"] == 0
      assert row["score"] is None
      assert row["feedback"] is None


def test_subjective_bucket_navigation_endpoints(client, monkeypatch):
  """Bucket navigation should iterate triaged responses within a bucket."""
  from grading_web_ui.web_api.routes import problems as problems_routes

  monkeypatch.setattr(
    problems_routes,
    "get_problem_image_data",
    lambda problem, submission_repo=None: ""
  )

  session_id = create_test_session(client, "Subjective Bucket Nav")
  _, problem_id_a = seed_submission_with_problem(
    session_id, document_id=1, problem_number=6
  )
  _, problem_id_b = seed_submission_with_problem(
    session_id, document_id=2, problem_number=6
  )
  _, problem_id_c = seed_submission_with_problem(
    session_id, document_id=3, problem_number=6
  )

  mode_response = client.put(
    f"/api/sessions/{session_id}/subjective-settings",
    json={
      "problem_number": 6,
      "grading_mode": "subjective",
      "buckets": [
        {"id": "good", "label": "Good"},
        {"id": "poor_blank", "label": "Poor/Blank"},
      ]
    }
  )
  assert mode_response.status_code == 200

  for problem_id, bucket_id in [
    (problem_id_a, "good"),
    (problem_id_b, "good"),
    (problem_id_c, "poor_blank"),
  ]:
    triage_response = client.post(
      f"/api/problems/{problem_id}/subjective-triage",
      json={"bucket_id": bucket_id}
    )
    assert triage_response.status_code == 200

  first = client.get(f"/api/problems/{session_id}/6/bucket/good/next")
  assert first.status_code == 200
  first_id = first.json()["id"]
  assert first_id in {problem_id_a, problem_id_b}

  second = client.get(
    f"/api/problems/{session_id}/6/bucket/good/next?current_problem_id={first_id}"
  )
  assert second.status_code == 200
  second_id = second.json()["id"]
  assert second_id in {problem_id_a, problem_id_b}
  assert second_id != first_id

  previous = client.get(
    f"/api/problems/{session_id}/6/bucket/good/previous?current_problem_id={second_id}"
  )
  assert previous.status_code == 200
  assert previous.json()["id"] == first_id

  sample = client.get(f"/api/problems/{session_id}/6/bucket/good/sample")
  assert sample.status_code == 200
  assert sample.json()["id"] in {problem_id_a, problem_id_b}


def test_session_stats_blank_rate_counts_manual_blanks_only(client):
  """Stats blank metrics should use manual '-' marks, not AI/heuristic blank flags."""
  session_id = create_test_session(client, "Manual Blank Stats Test")

  # Manual blank entry (dash grading path equivalent)
  seed_submission_with_problem(session_id,
                               document_id=1,
                               graded=True,
                               score=0.0,
                               is_blank=True,
                               blank_method="manual",
                               blank_reasoning="Manually marked as blank by grader")

  # AI/estimated blank should NOT count toward manual blank-rate tracking.
  seed_submission_with_problem(session_id,
                               document_id=2,
                               graded=True,
                               score=0.0,
                               is_blank=True,
                               blank_method="ai",
                               blank_reasoning="Accepted as blank by AI autograder")

  # Heuristic ungraded blank should also be excluded from stats blank tracking.
  seed_submission_with_problem(session_id,
                               document_id=3,
                               graded=False,
                               score=None,
                               is_blank=True,
                               blank_method="heuristic",
                               blank_reasoning="No writing detected")

  # Non-blank graded example
  seed_submission_with_problem(session_id,
                               document_id=4,
                               graded=True,
                               score=4.0,
                               is_blank=False)

  response = client.get(f"/api/sessions/{session_id}/stats")
  assert response.status_code == 200
  payload = response.json()
  stats_row = next(ps for ps in payload["problem_stats"]
                   if ps["problem_number"] == 1)

  assert stats_row["num_total"] == 4
  assert stats_row["num_graded"] == 3
  assert stats_row["num_blank"] == 1
  assert stats_row["num_blank_ungraded"] == 0
  assert stats_row["pct_blank"] == pytest.approx((1 / 3) * 100, rel=1e-3)


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
  session_id = create_test_session(client, "Export Coverage Test")
  seed_submission_with_problem(session_id,
                               file_hash="hash-export-1",
                               graded=True,
                               score=4.0,
                               feedback="nice work")

  ProblemStatsRepository().bulk_insert_for_session(
    session_id,
    [{
      "problem_number": 1,
      "avg_score": 4.0,
      "min_score": 4.0,
      "max_score": 4.0,
      "num_graded": 1,
      "num_total": 1,
    }],
  )
  ProblemMetadataRepository().bulk_insert_for_session(
    session_id,
    [{
      "problem_number": 1,
      "max_points": 5.0,
      "default_feedback": "default comment",
      "default_feedback_threshold": 90.0,
    }],
  )
  FeedbackTagRepository().bulk_insert_for_session(
    session_id,
    [{
      "problem_number": 1,
      "short_name": "tag-a",
      "comment_text": "comment a",
      "use_count": 2,
    }],
  )

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
  session_one_id = create_test_session(client, "Compare One")
  session_two_id = create_test_session(client, "Compare Two")
  seed_submission_with_problem(session_one_id,
                               file_hash="shared-hash-1",
                               graded=True,
                               score=3.0,
                               feedback="ok")
  seed_submission_with_problem(session_two_id,
                               file_hash="shared-hash-1",
                               graded=True,
                               score=4.0,
                               feedback="better")

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


def test_finalize_rejects_when_autograde_lock_active(client):
  """Finalize endpoint should reject when autograding lock is active."""
  session_id = create_test_session(client, "Finalize vs Autograde Lock Test")
  seed_submission_with_problem(session_id,
                               student_name="Student Lock",
                               canvas_user_id=9010,
                               graded=True,
                               score=4.0)

  assert workflow_locks.acquire("autograde", session_id) is True
  try:
    response = client.post(f"/api/finalize/{session_id}/finalize")
    assert response.status_code == 409
    assert "autograding is in progress" in response.json()["detail"].lower()
  finally:
    workflow_locks.release("autograde", session_id)


def test_autograde_all_rejects_when_finalize_lock_active(client):
  """Autograde endpoint should reject while finalization lock is active for session."""
  session_id = create_test_session(client, "Autograde Lock Test")
  seed_submission_with_problem(session_id,
                               student_name="Student Lock",
                               file_hash="lock-hash",
                               graded=False)

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

  session_id = create_test_session(client, "Finalize Concurrency Test")
  seed_submission_with_problem(session_id,
                               student_name="Student Concurrent",
                               canvas_user_id=9002,
                               graded=True,
                               score=5.0)

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


def test_autograde_all_concurrent_requests_allow_only_one_start(client,
                                                                 monkeypatch):
  """Concurrent autograde-all starts should allow one start and reject overlap."""
  from grading_web_ui.web_api.routes import ai_grader as ai_routes

  session_id = create_test_session(client, "Autograde-All Concurrency Test")
  for doc_id in range(1, 4):
    seed_submission_with_problem(session_id,
                                 document_id=doc_id,
                                 student_name=f"Student Auto {doc_id}",
                                 file_hash=f"autograde-concurrency-{doc_id}",
                                 problem_number=doc_id,
                                 graded=False)

  async def slow_fake_run_autograding_all(target_session_id: int,
                                          problem_numbers: list,
                                          totals_by_problem: dict,
                                          settings: dict,
                                          stream_id: str):
    await asyncio.sleep(0.25)
    workflow_locks.release("autograde", target_session_id)

  monkeypatch.setattr(ai_routes, "run_autograding_all",
                      slow_fake_run_autograding_all)

  payload = {
    "mode": "image-only",
    "settings": {
      "batch_size": "small",
      "image_quality": "medium",
      "include_answer": False,
      "include_default_feedback": False,
      "auto_accept": False,
      "dry_run": True
    }
  }

  def start_autograde_all() -> int:
    return client.post(f"/api/ai-grader/{session_id}/autograde-all",
                       json=payload).status_code

  with ThreadPoolExecutor(max_workers=2) as executor:
    statuses = list(executor.map(lambda _: start_autograde_all(), range(2)))

  assert sorted(statuses) == [200, 409]


def test_concurrent_grade_submissions_do_not_fail_with_db_locked(client):
  """Concurrent grading requests should not fail with SQLite lock errors."""
  session_id = create_test_session(client, "Concurrent Grading Lock Test")
  problem_ids = []

  for doc_id in range(1, 9):
    _, problem_id = seed_submission_with_problem(session_id,
                                                 document_id=doc_id,
                                                 student_name=f"Student {doc_id}",
                                                 file_hash=f"grade-lock-{doc_id}",
                                                 graded=False,
                                                 problem_number=1)
    problem_ids.append(problem_id)

  def submit_grade(problem_id: int) -> tuple[int, str]:
    response = client.post(f"/api/problems/{problem_id}/grade",
                           json={
                             "score": 1.0,
                             "feedback": "ok"
                           })
    body = response.text if hasattr(response, "text") else ""
    return response.status_code, body

  with ThreadPoolExecutor(max_workers=6) as executor:
    results = list(executor.map(submit_grade, problem_ids))

  statuses = [status for status, _ in results]
  bodies = [body.lower() for _, body in results]

  assert all(status == 200 for status in statuses)
  assert all("database is locked" not in body for body in bodies)


def test_finalize_lifecycle_starts_and_completes_with_background_task(client,
                                                                      monkeypatch):
  """Finalize endpoint should start and background task should drive status completion."""
  from grading_web_ui.web_api.routes import finalize as finalize_routes
  from grading_web_ui.web_api.repositories.session_repository import SessionRepository
  from grading_web_ui.web_api.domain.common import SessionStatus as DomainSessionStatus

  session_id = create_test_session(client, "Finalize Lifecycle Test")
  seed_submission_with_problem(session_id,
                               student_name="Student Final",
                               canvas_user_id=9001,
                               graded=True,
                               score=4.0)

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

  session_id = create_test_session(client, "Autograde Lifecycle Test")
  seed_submission_with_problem(session_id,
                               student_name="Student Auto",
                               file_hash="autograde-hash",
                               graded=False)

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


def test_autograde_all_lifecycle_starts_job(client, monkeypatch):
  """Autograde-all endpoint should start a background job for valid sessions."""
  from grading_web_ui.web_api.routes import ai_grader as ai_routes

  session_id = create_test_session(client, "Autograde-All Lifecycle Test")
  seed_submission_with_problem(session_id,
                               student_name="Student Auto 1",
                               file_hash="autograde-all-hash-1",
                               problem_number=1,
                               graded=False)
  seed_submission_with_problem(session_id,
                               document_id=2,
                               student_name="Student Auto 2",
                               file_hash="autograde-all-hash-2",
                               problem_number=2,
                               graded=False)

  async def fake_run_autograding_all(target_session_id: int, problem_numbers: list,
                                     totals_by_problem: dict, settings: dict,
                                     stream_id: str):
    workflow_locks.release("autograde", target_session_id)

  monkeypatch.setattr(ai_routes, "run_autograding_all",
                      fake_run_autograding_all)

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


def test_run_autograding_all_releases_lock_on_failure(monkeypatch):
  """Autograde-all worker should release lock and emit error event on exceptions."""
  from grading_web_ui.web_api.routes import ai_grader as ai_routes

  session_id = 9899
  assert workflow_locks.acquire("autograde", session_id) is True
  captured_events = []

  class FailingAIGrader:

    def autograde_problem_image_only(self, *args, **kwargs):
      raise RuntimeError("forced autograde-all failure")

  async def capture_event(stream_id: str, event_type: str, data: dict):
    captured_events.append((stream_id, event_type, data))

  monkeypatch.setattr(ai_routes, "AIGraderService", lambda: FailingAIGrader())
  monkeypatch.setattr(ai_routes.sse, "send_event", capture_event)

  asyncio.run(
    ai_routes.run_autograding_all(session_id, [1], {
      1: 1
    }, {
      "auto_accept": False,
      "dry_run": True
    }, "test_stream_all"))

  assert workflow_locks.is_active("autograde", session_id) is False
  assert any(event_type == "error" for _, event_type, _ in captured_events)


def test_run_finalization_sets_error_status_and_releases_lock(client,
                                                              monkeypatch):
  """Finalization worker should set ERROR status and release lock on failures."""
  from grading_web_ui.web_api.routes import finalize as finalize_routes
  session_repo = SessionRepository()

  session_id = create_test_session(client, "Finalize Failure Lifecycle Test")
  assert workflow_locks.acquire("finalize", session_id) is True
  captured_events = []

  class FailingFinalizer:

    def __init__(self, session_id, temp_dir, stream_id, loop):
      self.session_id = session_id

    def finalize(self):
      raise RuntimeError("forced finalization failure")

  async def capture_event(stream_id: str, event_type: str, data: dict):
    captured_events.append((stream_id, event_type, data))

  monkeypatch.setattr(finalize_routes, "FinalizationService", FailingFinalizer)
  monkeypatch.setattr(finalize_routes.sse, "send_event", capture_event)

  asyncio.run(finalize_routes.run_finalization(session_id, "finalize_stream"))

  session = session_repo.get_by_id(session_id)
  assert session is not None
  assert session.status.value == "error"
  assert "finalization failed" in (session.processing_message or "").lower()
  assert workflow_locks.is_active("finalize", session_id) is False
  assert any(event_type == "error" for _, event_type, _ in captured_events)
