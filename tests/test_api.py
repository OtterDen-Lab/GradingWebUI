"""
Basic API tests to verify setup.
"""
import pytest
from fastapi.testclient import TestClient
from grading_web_ui.web_api.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
  """Create authenticated test client with isolated DB"""
  db_path = tmp_path / "test_api.db"
  monkeypatch.setenv("GRADING_DB_PATH", str(db_path))

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
