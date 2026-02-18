"""
Repository for problem_stats table.
"""
from typing import List, Dict
import sqlite3

from .base import BaseRepository


class ProblemStatsRepository(BaseRepository):
  """
  Data access for problem_stats table.

  This table stores aggregate per-problem statistics for a session.
  """

  def _row_to_domain(self, row: sqlite3.Row) -> Dict:
    """Represent rows as plain dictionaries."""
    return dict(row)

  def list_by_session(self, session_id: int) -> List[Dict]:
    """
    Get all problem stat rows for a session.
    """
    with self._get_connection() as conn:
      return self._execute_and_fetch_all(
        conn,
        """
        SELECT * FROM problem_stats
        WHERE session_id = ?
        ORDER BY problem_number
        """,
        (session_id, )
      )

  def bulk_insert_for_session(self, session_id: int, stats_rows: List[Dict]) -> int:
    """
    Insert problem stat rows for a target session.

    Existing ids/session_ids from input rows are ignored.
    """
    if not stats_rows:
      return 0

    with self._get_connection() as conn:
      cursor = conn.cursor()
      inserted = 0
      for row in stats_rows:
        cursor.execute(
          """
          INSERT INTO problem_stats
          (session_id, problem_number, avg_score, min_score, max_score, num_graded, num_total, updated_at)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?)
          """,
          (
            session_id,
            row["problem_number"],
            row.get("avg_score"),
            row.get("min_score"),
            row.get("max_score"),
            row.get("num_graded", 0),
            row.get("num_total", 0),
            row.get("updated_at"),
          ),
        )
        inserted += 1
      return inserted
