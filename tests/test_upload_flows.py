"""
Upload/name/alignment flow tests.
"""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import fitz
import pytest
from fastapi.testclient import TestClient

from grading_web_ui.web_api.main import app
from grading_web_ui.web_api.repositories.session_repository import SessionRepository
from grading_web_ui.web_api.repositories.submission_repository import SubmissionRepository
from grading_web_ui.web_api.repositories.problem_repository import ProblemRepository
from grading_web_ui.web_api.repositories import with_transaction
from grading_web_ui.web_api.domain.submission import Submission


@pytest.fixture
def client(tmp_path, monkeypatch):
  """Create authenticated test client with isolated DB."""
  db_path = tmp_path / "test_upload_flows.db"
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
      },
    )
    assert login_response.status_code == 200
    yield test_client


def create_test_session(client, assignment_name: str = "Upload Flow Test") -> int:
  """Create session via API and return ID."""
  response = client.post("/api/sessions",
                         json={
                           "course_id": 12345,
                           "assignment_id": 67890,
                           "assignment_name": assignment_name
                         })
  assert response.status_code == 200
  return response.json()["id"]


def test_prepare_alignment_without_saved_splits_sets_awaiting_alignment(
    client, tmp_path, monkeypatch):
  """prepare-alignment should build composites and move session to awaiting_alignment."""
  from grading_web_ui.web_api.routes import uploads as uploads_routes
  from grading_web_ui.web_api.services import manual_alignment

  session_id = create_test_session(client, "Prepare Alignment Flow")
  pdf_path = tmp_path / "one.pdf"
  pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")

  session_repo = SessionRepository()
  session_repo.update_metadata(
    session_id, {
      "file_paths": [str(pdf_path)],
      "file_metadata": {
        str(pdf_path): {
          "hash": "align-hash-1",
          "original_filename": "one.pdf"
        }
      }
    })

  class FakeQRScanner:
    available = False

    def __init__(self):
      pass

  class FakeAlignmentService:

    def create_composite_images(self, *args, **kwargs):
      transforms = {
        pdf_path: {
          0: {
            "target_width": 100,
            "target_height": 200
          }
        }
      }
      return ({0: "ZmFrZQ=="}, {0: (100, 200)}, transforms)

    def suggest_split_points_from_composites(self, *args, **kwargs):
      return {0: [42]}

  monkeypatch.setattr(uploads_routes, "QRScanner", FakeQRScanner)
  monkeypatch.setattr(manual_alignment, "ManualAlignmentService",
                      FakeAlignmentService)

  response = client.post(f"/api/uploads/{session_id}/prepare-alignment")
  assert response.status_code == 200
  data = response.json()
  assert data["status"] == "awaiting_alignment"
  assert data["suggested_split_points"] == {"0": [42]}

  session = session_repo.get_by_id(session_id)
  assert session is not None
  assert session.status.value == "awaiting_alignment"
  metadata = session_repo.get_metadata(session_id) or {}
  assert metadata["composite_dimensions"]["0"] == [100, 200]
  assert "align-hash-1" in metadata["page_transforms"]


def test_prepare_alignment_with_saved_splits_starts_split_processing(
    client, tmp_path, monkeypatch):
  """prepare-alignment should auto-process when split_points already exist."""
  from grading_web_ui.web_api.routes import uploads as uploads_routes

  session_id = create_test_session(client, "Prepare Alignment Saved Splits")
  pdf_path = tmp_path / "two.pdf"
  pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")

  session_repo = SessionRepository()
  session_repo.update_metadata(
    session_id, {
      "file_paths": [str(pdf_path)],
      "file_metadata": {
        str(pdf_path): {
          "hash": "align-hash-2",
          "original_filename": "two.pdf"
        }
      },
      "split_points": {
        "0": [0.4]
      },
      "skip_first_region": False,
      "last_page_blank": True,
      "ai_provider": "openai",
    })

  captured = {}

  async def fake_process_exam_splits(session_id, file_paths, file_metadata,
                                     stream_id, manual_split_points,
                                     skip_first_region, last_page_blank,
                                     ai_provider):
    captured["session_id"] = session_id
    captured["manual_split_points"] = manual_split_points
    captured["skip_first_region"] = skip_first_region
    captured["last_page_blank"] = last_page_blank
    captured["ai_provider"] = ai_provider

  monkeypatch.setattr(uploads_routes, "process_exam_splits",
                      fake_process_exam_splits)

  response = client.post(f"/api/uploads/{session_id}/prepare-alignment")
  assert response.status_code == 200
  data = response.json()
  assert data["status"] == "processing"
  assert data["auto_processed"] is True
  assert captured["session_id"] == session_id
  assert captured["manual_split_points"] == {0: [0.4]}
  assert captured["skip_first_region"] is False
  assert captured["last_page_blank"] is True
  assert captured["ai_provider"] == "openai"

  session = session_repo.get_by_id(session_id)
  assert session is not None
  assert session.status.value == "preprocessing"


def test_submit_alignment_converts_pixel_splits_to_percentages(client, tmp_path,
                                                               monkeypatch):
  """submit-alignment should store percentage split points and dispatch split worker."""
  from grading_web_ui.web_api.routes import uploads as uploads_routes

  session_id = create_test_session(client, "Submit Alignment Flow")
  pdf_path = tmp_path / "three.pdf"
  pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")

  session_repo = SessionRepository()
  session_repo.update_metadata(
    session_id, {
      "file_paths": [str(pdf_path)],
      "file_metadata": {
        str(pdf_path): {
          "hash": "align-hash-3",
          "original_filename": "three.pdf"
        }
      },
      "composite_dimensions": {
        "0": [200, 400]
      },
    })

  captured = {}

  async def fake_process_exam_splits(session_id, file_paths, file_metadata,
                                     stream_id, manual_split_points,
                                     skip_first_region, last_page_blank,
                                     ai_provider):
    captured["manual_split_points"] = manual_split_points
    captured["skip_first_region"] = skip_first_region
    captured["last_page_blank"] = last_page_blank
    captured["ai_provider"] = ai_provider

  monkeypatch.setattr(uploads_routes, "process_exam_splits",
                      fake_process_exam_splits)

  response = client.post(f"/api/uploads/{session_id}/submit-alignment",
                         json={
                           "split_points": {
                             "0": [100, 200]
                           },
                           "skip_first_region": False,
                           "last_page_blank": True,
                           "ai_provider": "openai"
                         })
  assert response.status_code == 200
  assert response.json()["status"] == "processing"
  assert captured["manual_split_points"] == {0: [0.25, 0.5]}
  assert captured["skip_first_region"] is False
  assert captured["last_page_blank"] is True
  assert captured["ai_provider"] == "openai"

  metadata = session_repo.get_metadata(session_id) or {}
  assert metadata["split_points"]["0"] == [0.25, 0.5]
  assert metadata["skip_first_region"] is False
  assert metadata["last_page_blank"] is True
  assert metadata["ai_provider"] == "openai"


def test_process_exam_names_mock_roster_sets_awaiting_alignment(
    client, tmp_path, monkeypatch):
  """Name extraction worker should create submissions and move mock-roster sessions to alignment."""
  from grading_web_ui.web_api.routes import uploads as uploads_routes

  session_id = create_test_session(client, "Name Extraction Worker Flow")
  session_repo = SessionRepository()
  session_repo.update_metadata(session_id, {"mock_roster": True})

  pdf_path = tmp_path / "four.pdf"
  pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
  file_metadata = {pdf_path: {"hash": "name-hash-1", "original_filename": "four.pdf"}}

  class FakeProcessor:

    def __init__(self, ai_provider):
      pass

    def extract_name_image(self, pdf_path):
      return "name-image-base64"

  async def noop_event(stream_id: str, event_type: str, data: dict):
    return None

  monkeypatch.setattr(uploads_routes, "ExamProcessor", FakeProcessor)
  monkeypatch.setattr(uploads_routes.sse, "send_event", noop_event)

  asyncio.run(
    uploads_routes.process_exam_names(session_id, [pdf_path], file_metadata,
                                      "name_stream"))

  submissions = SubmissionRepository().get_by_session(session_id)
  assert len(submissions) == 1
  assert submissions[0].name_image_data == "name-image-base64"
  assert submissions[0].canvas_user_id is not None

  session = session_repo.get_by_id(session_id)
  assert session is not None
  assert session.status.value == "awaiting_alignment"


def test_process_exam_splits_creates_problems_and_marks_session_ready(
    client, tmp_path, monkeypatch):
  """Split worker should attach problems to existing submissions and mark session ready."""
  from grading_web_ui.web_api.routes import uploads as uploads_routes

  session_id = create_test_session(client, "Split Worker Flow")
  session_repo = SessionRepository()
  session_repo.update_metadata(session_id, {})

  pdf_path = tmp_path / "five.pdf"
  doc = fitz.open()
  doc.new_page(width=300, height=400)
  doc.save(str(pdf_path))
  doc.close()

  file_hash = "split-hash-1"
  with with_transaction() as repos:
    repos.submissions.create(
      Submission(
        id=0,
        session_id=session_id,
        document_id=1,
        approximate_name="Student Split",
        name_image_data="x",
        student_name="Student Split",
        display_name="Student Split",
        canvas_user_id=101,
        page_mappings={},
        file_hash=file_hash,
        original_filename="five.pdf",
        exam_pdf_data=None,
      ))

  class FakeProcessor:

    def __init__(self, ai_provider):
      pass

    def process_exams(self, **kwargs):
      problem_dto = SimpleNamespace(
        problem_number=1,
        is_blank=False,
        blank_confidence=0.0,
        blank_method=None,
        blank_reasoning=None,
        max_points=8.0,
        region_coords={
          "page_number": 0,
          "region_y_start": 10,
          "region_y_end": 50
        },
        qr_encrypted_data=None,
      )
      sub_dto = SimpleNamespace(
        file_hash=file_hash,
        page_mappings={"1": [0]},
        pdf_data="pdf-base64",
        problems=[problem_dto],
      )
      return [sub_dto], []

  async def noop_event(stream_id: str, event_type: str, data: dict):
    return None

  monkeypatch.setattr(uploads_routes, "ExamProcessor", FakeProcessor)
  monkeypatch.setattr(uploads_routes.sse, "send_event", noop_event)

  asyncio.run(
    uploads_routes.process_exam_splits(
      session_id,
      [pdf_path],
      {pdf_path: {"hash": file_hash, "original_filename": "five.pdf"}},
      "split_stream",
      manual_split_points={0: [0.2]},
      skip_first_region=False,
      last_page_blank=False,
      ai_provider="anthropic",
    ))

  problems = ProblemRepository().get_by_session_batch(session_id)
  assert len(problems) == 1
  assert problems[0].problem_number == 1

  session = session_repo.get_by_id(session_id)
  assert session is not None
  assert session.status.value == "ready"
