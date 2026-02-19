from __future__ import annotations

import os
from types import SimpleNamespace

import canvasapi.exceptions

from lms_interface.canvas_interface import CanvasAssignment


class _FakeSubmission:
  def __init__(self, *, existing_score=None, fail_comment_edit=False):
    self.score = existing_score
    self.fail_comment_edit = fail_comment_edit
    self.submission_comments = []
    self.edits = []
    self.uploaded_paths = []

  def edit(self, **kwargs):
    self.edits.append(kwargs)
    if self.fail_comment_edit and "comment" in kwargs:
      raise canvasapi.exceptions.CanvasException("comment update failed")

  def upload_comment(self, path):
    self.uploaded_paths.append(path)


class _FakeAssignmentAPI:
  def __init__(self, submission):
    self._submission = submission
    self.bulk_updates = []

  def get_submission(self, _user_id):
    return self._submission

  def submissions_bulk_update(self, grade_data, student_ids):
    self.bulk_updates.append((grade_data, student_ids))


def _build_assignment(fake_submission):
  fake_assignment_api = _FakeAssignmentAPI(fake_submission)
  fake_canvas_interface = SimpleNamespace(
    canvas=SimpleNamespace(
      _Canvas__requester=SimpleNamespace(
        request=lambda *_args, **_kwargs: SimpleNamespace(status_code=200)
      )
    )
  )
  fake_canvas_course = SimpleNamespace(
    course=SimpleNamespace(id=42),
    get_user=lambda user_id: f"user-{user_id}",
  )
  assignment = CanvasAssignment(
    canvasapi_interface=fake_canvas_interface,
    canvasapi_course=fake_canvas_course,
    canvasapi_assignment=fake_assignment_api,
  )
  return assignment, fake_assignment_api


def test_push_feedback_uploads_html_attachment_without_inline_summary():
  submission = _FakeSubmission(existing_score=10.0)
  assignment, fake_assignment_api = _build_assignment(submission)

  comment_html = "<html><body><h1>Feedback</h1><p>Great work.</p></body></html>"
  result = assignment.push_feedback(
    user_id=123,
    score=95.0,
    comments=comment_html,
    attachments=[],
  )

  assert result is True
  assert fake_assignment_api.bulk_updates == [
    (
      {"submission[posted_grade]": 95.0},
      [123],
    )
  ]
  assert not any("comment" in edit for edit in submission.edits)
  assert any(os.path.basename(path) == "feedback.html"
             for path in submission.uploaded_paths)
  assert not any(path.endswith(".txt") for path in submission.uploaded_paths)


def test_push_feedback_posts_plain_text_inline_comment():
  submission = _FakeSubmission()
  assignment, _ = _build_assignment(submission)

  result = assignment.push_feedback(
    user_id=123,
    score=88.0,
    comments="Plain text feedback",
    attachments=[],
  )

  assert result is True
  assert any(
    edit.get("comment", {}).get("text_comment") == "Plain text feedback"
    for edit in submission.edits
  )
  assert submission.uploaded_paths == []


def test_push_feedback_uploads_html_with_tag_attributes():
  submission = _FakeSubmission()
  assignment, _ = _build_assignment(submission)

  result = assignment.push_feedback(
    user_id=123,
    score=88.0,
    comments='<p class="feedback">Formatted feedback</p>',
    attachments=[],
  )

  assert result is True
  assert not any("comment" in edit for edit in submission.edits)
  assert any(os.path.basename(path) == "feedback.html"
             for path in submission.uploaded_paths)


def test_push_feedback_falls_back_to_txt_attachment_for_plain_text():
  submission = _FakeSubmission(fail_comment_edit=True)
  assignment, _ = _build_assignment(submission)

  result = assignment.push_feedback(
    user_id=123,
    score=88.0,
    comments="fallback",
    attachments=[],
  )

  assert result is True
  assert any(path.endswith(".txt") for path in submission.uploaded_paths)
