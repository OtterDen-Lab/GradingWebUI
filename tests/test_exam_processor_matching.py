from pathlib import Path

from grading_web_ui.web_api.services.exam_processor import ExamProcessor


def _stub_redaction(*args, **kwargs):
  return "pdf-base64", []


def test_process_exams_persists_auto_match_at_threshold(monkeypatch):
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

  assert len(matched) == 1
  assert len(unmatched) == 0
  assert matched[0].canvas_user_id == 101
  assert matched[0].student_name == "Ethan Peregoy"


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

  assert len(matched) == 1
  assert len(unmatched) == 1
  assert matched[0].canvas_user_id == 101
  assert unmatched[0].canvas_user_id is None
