from pathlib import Path

import fitz

from grading_web_ui.web_api.services.exam_processor import ExamProcessor


def _stub_redaction(*args, **kwargs):
  return "pdf-base64", []


def test_process_exams_keeps_auto_match_at_threshold_as_suggestion(monkeypatch):
  processor = ExamProcessor()

  monkeypatch.setattr(
    processor,
    "extract_name",
    lambda pdf_path, student_names=None: ("Ethan Peregoy", "name-image")
  )
  monkeypatch.setattr(processor, "redact_and_extract_regions", _stub_redaction)
  monkeypatch.setattr(
    processor,
    "_find_suggested_match",
    lambda approximate_name, unmatched_students: (unmatched_students[0], 98)
  )

  matched, unmatched = processor.process_exams(
    input_files=[Path("exam1.pdf")],
    canvas_students=[{"name": "Ethan Peregoy", "user_id": 101}],
  )

  assert len(matched) == 0
  assert len(unmatched) == 1
  assert unmatched[0].canvas_user_id is None
  assert unmatched[0].student_name is None
  assert unmatched[0].suggested_canvas_user_id == 101


def test_process_exams_does_not_auto_match_below_threshold(monkeypatch):
  processor = ExamProcessor()

  monkeypatch.setattr(
    processor,
    "extract_name",
    lambda pdf_path, student_names=None: ("Ethan Peregoy", "name-image")
  )
  monkeypatch.setattr(processor, "redact_and_extract_regions", _stub_redaction)
  monkeypatch.setattr(
    processor,
    "_find_suggested_match",
    lambda approximate_name, unmatched_students: (unmatched_students[0], 97)
  )

  matched, unmatched = processor.process_exams(
    input_files=[Path("exam1.pdf")],
    canvas_students=[{"name": "Ethan Peregoy", "user_id": 101}],
  )

  assert len(matched) == 0
  assert len(unmatched) == 1
  assert unmatched[0].canvas_user_id is None
  assert unmatched[0].student_name is None


def test_process_exams_does_not_auto_assign_same_student_twice(monkeypatch):
  processor = ExamProcessor()

  monkeypatch.setattr(
    processor,
    "extract_name",
    lambda pdf_path, student_names=None: ("Ethan Peregoy", "name-image")
  )
  monkeypatch.setattr(processor, "redact_and_extract_regions", _stub_redaction)

  matched, unmatched = processor.process_exams(
    input_files=[Path("exam1.pdf"), Path("exam2.pdf")],
    canvas_students=[{"name": "Ethan Peregoy", "user_id": 101}],
  )

  assert len(matched) == 0
  assert len(unmatched) == 2
  assert unmatched[0].suggested_canvas_user_id == 101
  assert unmatched[1].canvas_user_id is None
  assert unmatched[1].suggested_canvas_user_id is None


def test_extract_regions_reports_handwriting_progress_without_qr_scan(
    tmp_path, monkeypatch):
  """Handwriting progress must not rely on the optional QR pre-scan helper."""
  pdf_path = tmp_path / "exam.pdf"
  document = fitz.open()
  document.new_page(width=300, height=400)
  document.save(pdf_path)
  document.close()

  processor = ExamProcessor(
    qr_prescan_dpi_steps=[], handwriting_analysis_enabled=True)
  monkeypatch.setattr(
    processor.problem_service,
    "extract_image_from_document",
    lambda *args, **kwargs: ("image-base64", 400))
  messages = []

  _, problems = processor.redact_and_extract_regions(
    pdf_path,
    split_points={0: [0.0]},
    skip_first_region=False,
    message_callback=lambda message, step_increment=1: messages.append(
      (message, step_increment)))

  assert len(problems) == 1
  assert messages == [("Handwriting analysis complete for problem 1/1", 1)]
