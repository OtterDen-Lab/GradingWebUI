"""
Repository for subjective triage assignments.
"""
from typing import Dict, List, Optional
import sqlite3

from .base import BaseRepository


class SubjectiveTriageRepository(BaseRepository):
  """Data access for subjective_triage table."""

  def _row_to_domain(self, row: sqlite3.Row):
    return dict(row)

  def get_for_problem(self, problem_id: int) -> Optional[Dict]:
    """Get triage assignment for one problem."""
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute("""
        SELECT * FROM subjective_triage
        WHERE problem_id = ?
      """, (problem_id,))
      row = cursor.fetchone()
      return dict(row) if row else None

  def upsert(self, problem_id: int, session_id: int, problem_number: int,
             bucket_id: str, notes: Optional[str]) -> None:
    """Create or update a triage assignment."""
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute("""
        INSERT INTO subjective_triage
        (problem_id, session_id, problem_number, bucket_id, notes)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(problem_id)
        DO UPDATE SET
          bucket_id = excluded.bucket_id,
          notes = excluded.notes,
          updated_at = CURRENT_TIMESTAMP
      """, (problem_id, session_id, problem_number, bucket_id, notes))

  def clear(self, problem_id: int) -> None:
    """Remove a triage assignment for one problem."""
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute("""
        DELETE FROM subjective_triage
        WHERE problem_id = ?
      """, (problem_id,))

  def count_ungraded_for_problem_number(self, session_id: int,
                                        problem_number: int) -> int:
    """Count triaged-but-ungraded responses for a problem number."""
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute("""
        SELECT COUNT(*) AS count
        FROM subjective_triage st
        JOIN problems p ON p.id = st.problem_id
        WHERE st.session_id = ? AND st.problem_number = ? AND p.graded = 0
      """, (session_id, problem_number))
      row = cursor.fetchone()
      return int(row["count"] or 0)

  def get_bucket_counts(self, session_id: int, problem_number: int) -> Dict[str, int]:
    """Get counts by bucket_id for triaged-but-ungraded responses."""
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute("""
        SELECT st.bucket_id, COUNT(*) AS count
        FROM subjective_triage st
        JOIN problems p ON p.id = st.problem_id
        WHERE st.session_id = ? AND st.problem_number = ? AND p.graded = 0
        GROUP BY st.bucket_id
      """, (session_id, problem_number))
      return {row["bucket_id"]: int(row["count"] or 0) for row in cursor.fetchall()}

  def get_used_bucket_ids(self, session_id: int, problem_number: int) -> List[str]:
    """List bucket ids currently used by triaged-but-ungraded responses."""
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute("""
        SELECT DISTINCT st.bucket_id
        FROM subjective_triage st
        JOIN problems p ON p.id = st.problem_id
        WHERE st.session_id = ? AND st.problem_number = ? AND p.graded = 0
      """, (session_id, problem_number))
      return [row["bucket_id"] for row in cursor.fetchall() if row["bucket_id"]]

  def get_ungraded_problem_ids_by_bucket(self, session_id: int,
                                         problem_number: int) -> Dict[str, List[int]]:
    """Return triaged-ungraded problem IDs grouped by bucket_id."""
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute("""
        SELECT st.bucket_id, st.problem_id
        FROM subjective_triage st
        JOIN problems p ON p.id = st.problem_id
        WHERE st.session_id = ? AND st.problem_number = ? AND p.graded = 0
        ORDER BY st.bucket_id, st.problem_id
      """, (session_id, problem_number))

      grouped: Dict[str, List[int]] = {}
      for row in cursor.fetchall():
        bucket_id = row["bucket_id"]
        if not bucket_id:
          continue
        grouped.setdefault(bucket_id, []).append(int(row["problem_id"]))
      return grouped

  def count_graded_for_problem_number(self, session_id: int,
                                      problem_number: int) -> int:
    """Count triage rows whose problems are currently graded."""
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute("""
        SELECT COUNT(*) AS count
        FROM subjective_triage st
        JOIN problems p ON p.id = st.problem_id
        WHERE st.session_id = ? AND st.problem_number = ? AND p.graded = 1
      """, (session_id, problem_number))
      row = cursor.fetchone()
      return int(row["count"] or 0)

  def get_graded_problem_ids_for_problem_number(self, session_id: int,
                                                problem_number: int) -> List[int]:
    """Return problem ids that are graded and have subjective triage rows."""
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute("""
        SELECT st.problem_id
        FROM subjective_triage st
        JOIN problems p ON p.id = st.problem_id
        WHERE st.session_id = ? AND st.problem_number = ? AND p.graded = 1
        ORDER BY st.problem_id
      """, (session_id, problem_number))
      return [int(row["problem_id"]) for row in cursor.fetchall()]

  def clear_for_problem_ids(self, problem_ids: List[int]) -> int:
    """Remove triage assignments for the provided problem ids."""
    if not problem_ids:
      return 0
    with self._get_connection() as conn:
      cursor = conn.cursor()
      placeholders = ",".join("?" for _ in problem_ids)
      cursor.execute(f"""
        DELETE FROM subjective_triage
        WHERE problem_id IN ({placeholders})
      """, tuple(problem_ids))
      return int(cursor.rowcount or 0)
