"""
Database connection and schema management.
"""
import sqlite3
import os
import shutil
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
import logging
from datetime import datetime

log = logging.getLogger(__name__)

# Default database path (can be overridden via environment variable)
DEFAULT_DB_PATH = Path.home() / ".autograder" / "grading.db"
CURRENT_SCHEMA_VERSION = 26
BOOTSTRAP_ADMIN_USERNAME_ENV = "GRADING_BOOTSTRAP_ADMIN_USERNAME"
BOOTSTRAP_ADMIN_PASSWORD_ENV = "GRADING_BOOTSTRAP_ADMIN_PASSWORD"
BOOTSTRAP_ADMIN_EMAIL_ENV = "GRADING_BOOTSTRAP_ADMIN_EMAIL"
BOOTSTRAP_ADMIN_FULL_NAME_ENV = "GRADING_BOOTSTRAP_ADMIN_FULL_NAME"
DB_MIGRATION_BACKUP_DIR_ENV = "GRADING_DB_MIGRATION_BACKUP_DIR"
DB_CREATE_MIGRATION_BACKUP_ENV = "GRADING_DB_CREATE_MIGRATION_BACKUP"


def get_db_path() -> Path:
  """Get database path from environment or use default"""
  db_path = os.getenv("GRADING_DB_PATH", str(DEFAULT_DB_PATH))
  path = Path(db_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  return path


def _is_truthy(value: str) -> bool:
  return value.strip().lower() in ("1", "true", "yes", "on")


def _should_create_migration_backup() -> bool:
  return _is_truthy(os.getenv(DB_CREATE_MIGRATION_BACKUP_ENV, "true"))


def _get_migration_backup_dir(db_path: Path) -> Path:
  configured = os.getenv(DB_MIGRATION_BACKUP_DIR_ENV, "").strip()
  backup_dir = Path(configured) if configured else db_path.parent
  backup_dir.mkdir(parents=True, exist_ok=True)
  return backup_dir


def _create_pre_migration_backup(db_path: Path, from_version: int) -> Optional[Path]:
  """
  Create a pre-migration backup for rollback safety.

  Returns backup path when a backup is created, otherwise None.
  """
  if not _should_create_migration_backup():
    log.warning(
      "Skipping pre-migration backup because %s=false",
      DB_CREATE_MIGRATION_BACKUP_ENV)
    return None

  if not db_path.exists():
    return None

  backup_dir = _get_migration_backup_dir(db_path)
  db_related_paths = [
    db_path,
    Path(f"{db_path}-wal"),
    Path(f"{db_path}-shm"),
  ]
  db_size = sum(path.stat().st_size for path in db_related_paths if path.exists())
  # Require at least 2x DB size (or 5MB minimum) in backup destination.
  required_free_bytes = max(db_size * 2, 5 * 1024 * 1024)
  free_bytes = shutil.disk_usage(str(backup_dir)).free
  if free_bytes < required_free_bytes:
    raise RuntimeError(
      f"Insufficient disk space for pre-migration backup in {backup_dir}. "
      f"Required {required_free_bytes} bytes, available {free_bytes} bytes.")

  timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
  backup_path = backup_dir / f"{db_path.name}.v{from_version}.bak-{timestamp}"
  source_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
  backup_conn = sqlite3.connect(str(backup_path))
  try:
    # SQLite backup API creates a consistent snapshot and includes WAL state.
    source_conn.backup(backup_conn)
    backup_conn.commit()
  finally:
    backup_conn.close()
    source_conn.close()
  log.warning("Created pre-migration backup at %s", backup_path)
  return backup_path


def _get_busy_timeout_ms() -> int:
  raw_value = os.getenv("GRADING_DB_BUSY_TIMEOUT_MS", "30000").strip()
  try:
    return max(1000, int(raw_value))
  except ValueError:
    return 30000


def _get_connect_timeout_seconds() -> float:
  raw_value = os.getenv("GRADING_DB_CONNECT_TIMEOUT_SECONDS", "30").strip()
  try:
    return max(1.0, float(raw_value))
  except ValueError:
    return 30.0


def maybe_create_bootstrap_admin(cursor, email_required: bool = False):
  """
  Create initial admin user only when explicitly configured via environment.
  """
  password = os.getenv(BOOTSTRAP_ADMIN_PASSWORD_ENV, "").strip()
  if not password:
    log.info(
      "Skipping bootstrap admin creation because %s is not set",
      BOOTSTRAP_ADMIN_PASSWORD_ENV)
    return

  username = os.getenv(BOOTSTRAP_ADMIN_USERNAME_ENV, "admin").strip() or "admin"
  email = os.getenv(BOOTSTRAP_ADMIN_EMAIL_ENV, "").strip() or None
  full_name = (os.getenv(BOOTSTRAP_ADMIN_FULL_NAME_ENV, "Administrator").strip()
               or "Administrator")

  if email_required and not email:
    email = f"{username}@example.com"

  cursor.execute("SELECT id FROM users WHERE username = ? LIMIT 1", (username, ))
  if cursor.fetchone():
    log.info("Bootstrap admin user '%s' already exists; skipping", username)
    return

  import bcrypt
  password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
  cursor.execute(
    """
      INSERT INTO users (username, email, password_hash, full_name, role)
      VALUES (?, ?, ?, ?, ?)
    """, (username, email, password_hash, full_name, "instructor"))
  log.info("Created bootstrap admin user '%s' from environment config", username)


@contextmanager
def get_db_connection():
  """Context manager for database connections"""
  db_path = get_db_path()
  conn = sqlite3.connect(str(db_path), timeout=_get_connect_timeout_seconds())
  conn.row_factory = sqlite3.Row  # Enable column access by name
  conn.execute("PRAGMA foreign_keys = ON")
  conn.execute(f"PRAGMA busy_timeout = {_get_busy_timeout_ms()}")
  try:
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
  except sqlite3.OperationalError:
    # WAL may be unavailable on some filesystems; continue with defaults.
    log.warning("Could not enable SQLite WAL mode for %s", db_path)
  try:
    yield conn
    conn.commit()
  except Exception:
    conn.rollback()
    raise
  finally:
    conn.close()


def init_database():
  """Initialize database with schema"""
  db_path = get_db_path()
  log.info(f"Initializing database at {db_path}")

  with get_db_connection() as conn:
    cursor = conn.cursor()

    # Check current schema version
    current_version = get_schema_version(cursor)

    if current_version == 0:
      # Create new database
      create_schema(cursor)
    elif current_version > CURRENT_SCHEMA_VERSION:
      raise RuntimeError(
        f"Database schema version {current_version} is newer than supported "
        f"application schema version {CURRENT_SCHEMA_VERSION}. "
        "Downgrades are not supported. Use a matching/newer application version.")
    elif current_version < CURRENT_SCHEMA_VERSION:
      # Preflight backup before in-place migrations for rollback safety.
      _create_pre_migration_backup(db_path, current_version)
      # Run migrations
      run_migrations(cursor, current_version)

    # Always allow bootstrap-admin creation on startup when users table exists.
    # This provides an operator recovery path if the DB was initialized without
    # bootstrap credentials and currently has no instructor account.
    cursor.execute(
      "SELECT 1 FROM sqlite_master WHERE type='table' AND name='users' LIMIT 1")
    if cursor.fetchone():
      maybe_create_bootstrap_admin(cursor)

    log.info(f"Database ready (schema version {CURRENT_SCHEMA_VERSION})")


def get_schema_version(cursor) -> int:
  """Get current schema version"""
  try:
    cursor.execute(
      "SELECT version FROM _schema_version ORDER BY version DESC LIMIT 1")
    result = cursor.fetchone()
    return result[0] if result else 0
  except sqlite3.OperationalError:
    # Table doesn't exist yet
    return 0


def create_schema(cursor):
  """Create initial database schema"""

  # Schema version tracking
  cursor.execute("""
        CREATE TABLE _schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

  # Grading sessions
  cursor.execute("""
        CREATE TABLE grading_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assignment_id INTEGER NOT NULL,
            assignment_name TEXT NOT NULL,
            course_id INTEGER NOT NULL,
            course_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL,
            canvas_points REAL,
            metadata TEXT,
            total_exams INTEGER DEFAULT 0,
            processed_exams INTEGER DEFAULT 0,
            matched_exams INTEGER DEFAULT 0,
            processing_message TEXT,
            use_prod_canvas INTEGER DEFAULT 0
        )
    """)

  # Student submissions
  cursor.execute("""
        CREATE TABLE submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            document_id INTEGER NOT NULL,
            approximate_name TEXT,
            name_image_data TEXT,
            student_name TEXT,
            display_name TEXT,
            canvas_user_id INTEGER,
            page_mappings TEXT NOT NULL,
            total_score REAL,
            graded_at TIMESTAMP,
            file_hash TEXT,
            original_filename TEXT,
            exam_pdf_data TEXT,
            FOREIGN KEY (session_id) REFERENCES grading_sessions(id)
        )
    """)

  # Individual problems (PDF-based storage only, no image_data column)
  cursor.execute("""
        CREATE TABLE problems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            submission_id INTEGER NOT NULL,
            problem_number INTEGER NOT NULL,
            score REAL,
            feedback TEXT,
            graded INTEGER DEFAULT 0,
            graded_at TIMESTAMP,
            is_blank INTEGER DEFAULT 0,
            blank_confidence REAL DEFAULT 0.0,
            blank_method TEXT,
            blank_reasoning TEXT,
            max_points REAL,
            ai_reasoning TEXT,
            region_coords TEXT,
            qr_encrypted_data TEXT,
            regeneration_cache_key TEXT,
            regeneration_response_json TEXT,
            regeneration_cached_at TIMESTAMP,
            transcription TEXT,
            transcription_model TEXT,
            transcription_cached_at TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES grading_sessions(id),
            FOREIGN KEY (submission_id) REFERENCES submissions(id)
        )
    """)

  # Create indexes for performance
  cursor.execute("""
        CREATE INDEX idx_problems_session_problem
        ON problems(session_id, problem_number)
    """)

  cursor.execute("""
        CREATE INDEX idx_problems_graded
        ON problems(session_id, graded)
    """)

  cursor.execute("""
        CREATE INDEX idx_submissions_session
        ON submissions(session_id)
    """)

  cursor.execute("""
        CREATE INDEX idx_submissions_file_hash
        ON submissions(session_id, file_hash)
    """)

  # Problem statistics (computed view)
  cursor.execute("""
        CREATE TABLE problem_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            problem_number INTEGER NOT NULL,
            avg_score REAL,
            min_score REAL,
            max_score REAL,
            num_graded INTEGER,
            num_total INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES grading_sessions(id),
            UNIQUE(session_id, problem_number)
        )
    """)

  # Problem metadata (for storing max_points, rubrics, etc. per problem number)
  cursor.execute("""
        CREATE TABLE problem_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            problem_number INTEGER NOT NULL,
            max_points REAL,
            question_text TEXT,
            grading_rubric TEXT,
            default_feedback TEXT,
            default_feedback_threshold REAL DEFAULT 100.0,
            ai_grading_notes TEXT,
            grading_mode TEXT DEFAULT 'calculation',
            subjective_buckets_json TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES grading_sessions(id),
            UNIQUE(session_id, problem_number)
        )
    """)

  # Subjective triage assignments (per-response bucket sorting prior to scoring)
  cursor.execute("""
        CREATE TABLE subjective_triage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            problem_id INTEGER NOT NULL UNIQUE,
            session_id INTEGER NOT NULL,
            problem_number INTEGER NOT NULL,
            bucket_id TEXT NOT NULL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE,
            FOREIGN KEY (session_id) REFERENCES grading_sessions(id) ON DELETE CASCADE
        )
    """)
  cursor.execute("""
        CREATE INDEX idx_subjective_triage_session_problem
        ON subjective_triage(session_id, problem_number)
    """)

  # Feedback tags (reusable grading comments)
  cursor.execute("""
        CREATE TABLE feedback_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            problem_number INTEGER NOT NULL,
            short_name TEXT NOT NULL,
            comment_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            use_count INTEGER DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES grading_sessions(id),
            UNIQUE(session_id, problem_number, short_name)
        )
    """)

  cursor.execute("""
        CREATE INDEX idx_feedback_tags_session_problem
        ON feedback_tags(session_id, problem_number)
    """)

  # Authentication and RBAC tables (v22, email made optional in v23)
  cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            role TEXT NOT NULL CHECK(role IN ('instructor', 'ta')),
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by INTEGER,
            FOREIGN KEY (created_by) REFERENCES users(id)
        )
    """)

  cursor.execute("""
        CREATE TABLE auth_sessions (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            ip_address TEXT,
            user_agent TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

  cursor.execute("""
        CREATE TABLE session_assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            assigned_by INTEGER,
            FOREIGN KEY (session_id) REFERENCES grading_sessions(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (assigned_by) REFERENCES users(id),
            UNIQUE(session_id, user_id)
        )
    """)

  # Create indexes for auth tables
  cursor.execute(
    "CREATE INDEX idx_auth_sessions_user_id ON auth_sessions(user_id)")
  cursor.execute(
    "CREATE INDEX idx_auth_sessions_expires ON auth_sessions(expires_at)")
  cursor.execute(
    "CREATE INDEX idx_session_assignments_user ON session_assignments(user_id)")
  cursor.execute(
    "CREATE INDEX idx_session_assignments_session ON session_assignments(session_id)"
  )

  maybe_create_bootstrap_admin(cursor)

  # Record schema version
  cursor.execute("INSERT INTO _schema_version (version) VALUES (?)",
                 (CURRENT_SCHEMA_VERSION, ))

  log.info(f"Created database schema version {CURRENT_SCHEMA_VERSION}")


def run_migrations(cursor, from_version: int):
  """Run database migrations from current version to latest"""
  log.info(
    f"Running migrations from version {from_version} to {CURRENT_SCHEMA_VERSION}"
  )

  if from_version < 2:
    migrate_to_v2(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (2)")

  if from_version < 3:
    migrate_to_v3(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (3)")

  if from_version < 4:
    migrate_to_v4(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (4)")

  if from_version < 5:
    migrate_to_v5(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (5)")

  if from_version < 6:
    migrate_to_v6(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (6)")

  if from_version < 7:
    migrate_to_v7(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (7)")

  if from_version < 8:
    migrate_to_v8(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (8)")

  if from_version < 9:
    migrate_to_v9(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (9)")

  if from_version < 10:
    migrate_to_v10(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (10)")

  if from_version < 11:
    migrate_to_v11(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (11)")

  if from_version < 12:
    migrate_to_v12(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (12)")

  if from_version < 13:
    migrate_to_v13(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (13)")

  if from_version < 14:
    migrate_to_v14(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (14)")

  if from_version < 15:
    migrate_to_v15(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (15)")

  if from_version < 16:
    migrate_to_v16(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (16)")

  if from_version < 17:
    migrate_to_v17(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (17)")

  if from_version < 18:
    migrate_to_v18(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (18)")

  if from_version < 19:
    migrate_to_v19(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (19)")

  if from_version < 20:
    migrate_to_v20(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (20)")

  if from_version < 21:
    migrate_to_v21(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (21)")

  if from_version < 22:
    migrate_to_v22(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (22)")

  if from_version < 23:
    migrate_to_v23(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (23)")

  if from_version < 24:
    migrate_to_v24(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (24)")

  if from_version < 25:
    migrate_to_v25(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (25)")

  if from_version < 26:
    migrate_to_v26(cursor)
    cursor.execute("INSERT INTO _schema_version (version) VALUES (26)")


def migrate_to_v2(cursor):
  """Add progress tracking columns to grading_sessions"""
  log.info("Migrating to schema version 2: adding progress tracking")

  cursor.execute(
    "ALTER TABLE grading_sessions ADD COLUMN total_exams INTEGER DEFAULT 0")
  cursor.execute(
    "ALTER TABLE grading_sessions ADD COLUMN processed_exams INTEGER DEFAULT 0"
  )
  cursor.execute(
    "ALTER TABLE grading_sessions ADD COLUMN matched_exams INTEGER DEFAULT 0")
  cursor.execute(
    "ALTER TABLE grading_sessions ADD COLUMN processing_message TEXT")


def migrate_to_v3(cursor):
  """Add approximate_name column to submissions"""
  log.info(
    "Migrating to schema version 3: adding approximate_name to submissions")

  cursor.execute("ALTER TABLE submissions ADD COLUMN approximate_name TEXT")


def migrate_to_v4(cursor):
  """Add name_image_data column to submissions"""
  log.info(
    "Migrating to schema version 4: adding name_image_data to submissions")

  cursor.execute("ALTER TABLE submissions ADD COLUMN name_image_data TEXT")


def migrate_to_v5(cursor):
  """Add blank detection columns to problems"""
  log.info(
    "Migrating to schema version 5: adding blank detection columns to problems"
  )

  cursor.execute("ALTER TABLE problems ADD COLUMN is_blank INTEGER DEFAULT 0")
  cursor.execute(
    "ALTER TABLE problems ADD COLUMN blank_confidence REAL DEFAULT 0.0")
  cursor.execute("ALTER TABLE problems ADD COLUMN blank_method TEXT")
  cursor.execute("ALTER TABLE problems ADD COLUMN blank_reasoning TEXT")


def migrate_to_v6(cursor):
  """Add file hash tracking to submissions"""
  log.info(
    "Migrating to schema version 6: adding file_hash and original_filename to submissions"
  )

  cursor.execute("ALTER TABLE submissions ADD COLUMN file_hash TEXT")
  cursor.execute("ALTER TABLE submissions ADD COLUMN original_filename TEXT")

  # Create index for fast duplicate detection
  cursor.execute(
    "CREATE INDEX IF NOT EXISTS idx_submissions_file_hash ON submissions(session_id, file_hash)"
  )


def migrate_to_v7(cursor):
  """Add Canvas environment setting to sessions"""
  log.info(
    "Migrating to schema version 7: adding use_prod_canvas to grading_sessions"
  )

  cursor.execute(
    "ALTER TABLE grading_sessions ADD COLUMN use_prod_canvas INTEGER DEFAULT 0"
  )


def migrate_to_v8(cursor):
  """Add min/max score tracking to problem_stats"""
  log.info(
    "Migrating to schema version 8: adding min_score and max_score to problem_stats"
  )

  cursor.execute("ALTER TABLE problem_stats ADD COLUMN min_score REAL")
  cursor.execute("ALTER TABLE problem_stats ADD COLUMN max_score REAL")


def migrate_to_v9(cursor):
  """Add max_points column to problems"""
  log.info("Migrating to schema version 9: adding max_points to problems")

  cursor.execute("ALTER TABLE problems ADD COLUMN max_points REAL")


def migrate_to_v10(cursor):
  """Create problem_metadata table for storing max_points per problem number"""
  log.info("Migrating to schema version 10: creating problem_metadata table")

  cursor.execute("""
        CREATE TABLE IF NOT EXISTS problem_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            problem_number INTEGER NOT NULL,
            max_points REAL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES grading_sessions(id),
            UNIQUE(session_id, problem_number)
        )
    """)


def migrate_to_v11(cursor):
  """Add question_text column to problem_metadata for autograding"""
  log.info(
    "Migrating to schema version 11: adding question_text to problem_metadata")

  cursor.execute("ALTER TABLE problem_metadata ADD COLUMN question_text TEXT")


def migrate_to_v12(cursor):
  """Add ai_reasoning column to problems for storing AI feedback separately"""
  log.info("Migrating to schema version 12: adding ai_reasoning to problems")

  cursor.execute("ALTER TABLE problems ADD COLUMN ai_reasoning TEXT")


def migrate_to_v13(cursor):
  """Add region_coords to problems for PDF-based storage (was added in actual v13)"""
  log.info("Migrating to schema version 13: adding region_coords to problems")

  # This migration was already applied, but we need the function for migration flow
  # Check if column exists before adding
  cursor.execute("PRAGMA table_info(problems)")
  columns = [row[1] for row in cursor.fetchall()]

  if "region_coords" not in columns:
    cursor.execute("ALTER TABLE problems ADD COLUMN region_coords TEXT")


def migrate_to_v14(cursor):
  """Make image_data nullable in problems table for PDF-based storage"""
  log.info("Migrating to schema version 14: making image_data nullable")

  # SQLite doesn't support ALTER COLUMN, so we need to recreate the table
  # Step 1: Get all existing data
  cursor.execute("SELECT * FROM problems")
  existing_data = cursor.fetchall()

  # Step 2: Get column names
  cursor.execute("PRAGMA table_info(problems)")
  columns_info = cursor.fetchall()

  # Step 3: Drop old table and recreate without NOT NULL on image_data
  cursor.execute("DROP TABLE IF EXISTS problems_backup")
  cursor.execute("ALTER TABLE problems RENAME TO problems_backup")

  # Recreate problems table with image_data as nullable
  cursor.execute("""
        CREATE TABLE problems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            submission_id INTEGER NOT NULL,
            problem_number INTEGER NOT NULL,
            image_data TEXT,
            score REAL,
            feedback TEXT,
            graded INTEGER DEFAULT 0,
            graded_at TIMESTAMP,
            is_blank INTEGER DEFAULT 0,
            blank_confidence REAL DEFAULT 0.0,
            blank_method TEXT,
            blank_reasoning TEXT,
            max_points REAL,
            transcription TEXT,
            transcription_model TEXT,
            ai_grading_status TEXT DEFAULT 'none',
            ai_suggested_score REAL,
            ai_suggested_feedback TEXT,
            ai_suggestion_received_at TIMESTAMP,
            is_example_submission INTEGER DEFAULT 0,
            example_priority INTEGER DEFAULT 0,
            ai_reasoning TEXT,
            region_coords TEXT,
            page_number INTEGER,
            region_y_start INTEGER,
            region_y_end INTEGER,
            region_height INTEGER,
            FOREIGN KEY (session_id) REFERENCES grading_sessions(id),
            FOREIGN KEY (submission_id) REFERENCES submissions(id)
        )
    """)

  # Step 4: Copy data back
  # Get column names from backup table
  cursor.execute("PRAGMA table_info(problems_backup)")
  backup_columns = [row[1] for row in cursor.fetchall()]
  column_list = ", ".join(backup_columns)

  cursor.execute(f"""
        INSERT INTO problems ({column_list})
        SELECT {column_list} FROM problems_backup
    """)

  # Step 5: Recreate indexes
  cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_problems_session_problem
        ON problems(session_id, problem_number)
    """)

  cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_problems_graded
        ON problems(session_id, graded)
    """)

  # Step 6: Drop backup table
  cursor.execute("DROP TABLE problems_backup")

  log.info("Successfully made image_data nullable in problems table")


def migrate_to_v15(cursor):
  """Add grading_rubric column to problem_metadata for AI-assisted rubric generation"""
  log.info(
    "Migrating to schema version 15: adding grading_rubric to problem_metadata"
  )

  cursor.execute("ALTER TABLE problem_metadata ADD COLUMN grading_rubric TEXT")


def migrate_to_v16(cursor):
  """Add QR code metadata columns to problems table"""
  log.info(
    "Migrating to schema version 16: adding QR code metadata columns to problems"
  )

  # Add columns for storing QR code metadata (question type, seed, version)
  cursor.execute("ALTER TABLE problems ADD COLUMN qr_question_type TEXT")
  cursor.execute("ALTER TABLE problems ADD COLUMN qr_seed INTEGER")
  cursor.execute("ALTER TABLE problems ADD COLUMN qr_version TEXT")


def migrate_to_v17(cursor):
  """Replace QR code fields with single encrypted data field"""
  log.info(
    "Migrating to schema version 17: replacing QR fields with qr_encrypted_data"
  )

  # Add new encrypted data column
  cursor.execute("ALTER TABLE problems ADD COLUMN qr_encrypted_data TEXT")

  # Note: Old columns (qr_question_type, qr_seed, qr_version) remain for backward compatibility
  # but new code will only use qr_encrypted_data


def migrate_to_v18(cursor):
  """Create feedback_tags table for reusable grading comments"""
  log.info("Migrating to schema version 18: creating feedback_tags table")

  cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            problem_number INTEGER NOT NULL,
            short_name TEXT NOT NULL,
            comment_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            use_count INTEGER DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES grading_sessions(id),
            UNIQUE(session_id, problem_number, short_name)
        )
    """)

  # Create index for fast lookup by session and problem
  cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_feedback_tags_session_problem
        ON feedback_tags(session_id, problem_number)
    """)


def migrate_to_v19(cursor):
  """Add default feedback columns to problem_metadata"""
  log.info(
    "Migrating to schema version 19: adding default_feedback to problem_metadata"
  )

  # Check if columns already exist
  cursor.execute("PRAGMA table_info(problem_metadata)")
  existing_columns = {row[1] for row in cursor.fetchall()}

  if 'default_feedback' not in existing_columns:
    cursor.execute(
      "ALTER TABLE problem_metadata ADD COLUMN default_feedback TEXT")
    log.info("Added default_feedback column")

  if 'default_feedback_threshold' not in existing_columns:
    cursor.execute(
      "ALTER TABLE problem_metadata ADD COLUMN default_feedback_threshold REAL DEFAULT 100.0"
    )
    log.info("Added default_feedback_threshold column")


def migrate_to_v20(cursor):
  """Add transcription caching columns to problems table"""
  log.info(
    "Migrating to schema version 20: adding transcription cache columns to problems"
  )

  # Check if columns already exist
  cursor.execute("PRAGMA table_info(problems)")
  existing_columns = {row[1] for row in cursor.fetchall()}

  if 'transcription' not in existing_columns:
    cursor.execute("ALTER TABLE problems ADD COLUMN transcription TEXT")
    log.info("Added transcription column")

  if 'transcription_model' not in existing_columns:
    cursor.execute("ALTER TABLE problems ADD COLUMN transcription_model TEXT")
    log.info("Added transcription_model column")

  if 'transcription_cached_at' not in existing_columns:
    cursor.execute(
      "ALTER TABLE problems ADD COLUMN transcription_cached_at TIMESTAMP")
    log.info("Added transcription_cached_at column")


def migrate_to_v21(cursor):
  """Remove image_data column from problems table (no longer used with PDF-based storage)"""
  log.info(
    "Migrating to schema version 21: removing image_data column from problems")

  # SQLite doesn't support DROP COLUMN directly, need to recreate table
  # Create new table without image_data column
  cursor.execute("""
        CREATE TABLE problems_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            submission_id INTEGER NOT NULL,
            problem_number INTEGER NOT NULL,
            score REAL,
            feedback TEXT,
            graded INTEGER DEFAULT 0,
            graded_at TIMESTAMP,
            is_blank INTEGER DEFAULT 0,
            blank_confidence REAL DEFAULT 0.0,
            blank_method TEXT,
            blank_reasoning TEXT,
            max_points REAL,
            ai_reasoning TEXT,
            region_coords TEXT,
            qr_encrypted_data TEXT,
            transcription TEXT,
            transcription_model TEXT,
            transcription_cached_at TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES grading_sessions(id),
            FOREIGN KEY (submission_id) REFERENCES submissions(id)
        )
    """)

  # Copy data from old table (excluding image_data)
  cursor.execute("""
        INSERT INTO problems_new
        (id, session_id, submission_id, problem_number, score, feedback, graded, graded_at,
         is_blank, blank_confidence, blank_method, blank_reasoning, max_points, ai_reasoning,
         region_coords, qr_encrypted_data, transcription, transcription_model, transcription_cached_at)
        SELECT
         id, session_id, submission_id, problem_number, score, feedback, graded, graded_at,
         is_blank, blank_confidence, blank_method, blank_reasoning, max_points, ai_reasoning,
         region_coords, qr_encrypted_data, transcription, transcription_model, transcription_cached_at
        FROM problems
    """)

  # Drop old table
  cursor.execute("DROP TABLE problems")

  # Rename new table
  cursor.execute("ALTER TABLE problems_new RENAME TO problems")

  # Recreate indexes
  cursor.execute("""
        CREATE INDEX idx_problems_session_problem
        ON problems(session_id, problem_number)
    """)

  cursor.execute("""
        CREATE INDEX idx_problems_graded
        ON problems(session_id, graded)
    """)

  log.info("Successfully removed image_data column from problems table")


def migrate_to_v22(cursor):
  """Add authentication and RBAC tables for multi-user support"""
  log.info(
    "Migrating to schema version 22: adding authentication and RBAC tables")

  # Users table
  cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            role TEXT NOT NULL CHECK(role IN ('instructor', 'ta')),
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by INTEGER,
            FOREIGN KEY (created_by) REFERENCES users(id)
        )
    """)

  # Authentication sessions table (not grading sessions!)
  cursor.execute("""
        CREATE TABLE auth_sessions (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            ip_address TEXT,
            user_agent TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

  # Session assignments table (which TAs can access which grading sessions)
  cursor.execute("""
        CREATE TABLE session_assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            assigned_by INTEGER,
            FOREIGN KEY (session_id) REFERENCES grading_sessions(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (assigned_by) REFERENCES users(id),
            UNIQUE(session_id, user_id)
        )
    """)

  # Create indexes for performance
  cursor.execute(
    "CREATE INDEX idx_auth_sessions_user_id ON auth_sessions(user_id)")
  cursor.execute(
    "CREATE INDEX idx_auth_sessions_expires ON auth_sessions(expires_at)")
  cursor.execute(
    "CREATE INDEX idx_session_assignments_user ON session_assignments(user_id)")
  cursor.execute(
    "CREATE INDEX idx_session_assignments_session ON session_assignments(session_id)"
  )

  maybe_create_bootstrap_admin(cursor, email_required=True)
  log.info("Successfully added authentication and RBAC tables")


def migrate_to_v23(cursor):
  """Make email field optional in users table"""
  log.info("Migrating to schema version 23: making email optional")

  # SQLite doesn't support ALTER COLUMN, so we need to recreate the table
  # Step 1: Create new users table with email as nullable
  cursor.execute("""
        CREATE TABLE users_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            role TEXT NOT NULL CHECK(role IN ('instructor', 'ta')),
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by INTEGER,
            FOREIGN KEY (created_by) REFERENCES users(id)
        )
    """)

  # Step 2: Copy data from old table
  cursor.execute("""
        INSERT INTO users_new (id, username, email, password_hash, full_name, role, is_active, created_at, created_by)
        SELECT id, username, email, password_hash, full_name, role, is_active, created_at, created_by
        FROM users
    """)

  # Step 3: Drop old table
  cursor.execute("DROP TABLE users")

  # Step 4: Rename new table
  cursor.execute("ALTER TABLE users_new RENAME TO users")

  log.info("Successfully made email optional in users table")


def migrate_to_v24(cursor):
  """Add ai_grading_notes column to problem_metadata"""
  log.info(
    "Migrating to schema version 24: adding ai_grading_notes to problem_metadata"
  )

  cursor.execute("PRAGMA table_info(problem_metadata)")
  existing_columns = {row[1] for row in cursor.fetchall()}

  if 'ai_grading_notes' not in existing_columns:
    cursor.execute(
      "ALTER TABLE problem_metadata ADD COLUMN ai_grading_notes TEXT")
    log.info("Added ai_grading_notes column")


def migrate_to_v25(cursor):
  """Add subjective grading metadata and triage assignment table."""
  log.info(
    "Migrating to schema version 25: adding subjective grading settings and triage table"
  )

  cursor.execute("PRAGMA table_info(problem_metadata)")
  existing_columns = {row[1] for row in cursor.fetchall()}

  if 'grading_mode' not in existing_columns:
    cursor.execute(
      "ALTER TABLE problem_metadata ADD COLUMN grading_mode TEXT DEFAULT 'calculation'"
    )
    log.info("Added grading_mode column")

  if 'subjective_buckets_json' not in existing_columns:
    cursor.execute(
      "ALTER TABLE problem_metadata ADD COLUMN subjective_buckets_json TEXT"
    )
    log.info("Added subjective_buckets_json column")

  cursor.execute("""
    CREATE TABLE IF NOT EXISTS subjective_triage (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      problem_id INTEGER NOT NULL UNIQUE,
      session_id INTEGER NOT NULL,
      problem_number INTEGER NOT NULL,
      bucket_id TEXT NOT NULL,
      notes TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (problem_id) REFERENCES problems(id) ON DELETE CASCADE,
      FOREIGN KEY (session_id) REFERENCES grading_sessions(id) ON DELETE CASCADE
    )
  """)
  cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_subjective_triage_session_problem
    ON subjective_triage(session_id, problem_number)
  """)


def migrate_to_v26(cursor):
  """Add persistent regeneration cache columns to problems table."""
  log.info(
    "Migrating to schema version 26: adding regeneration cache columns to problems"
  )

  cursor.execute("PRAGMA table_info(problems)")
  existing_columns = {row[1] for row in cursor.fetchall()}

  if 'regeneration_cache_key' not in existing_columns:
    cursor.execute("ALTER TABLE problems ADD COLUMN regeneration_cache_key TEXT")
    log.info("Added regeneration_cache_key column")

  if 'regeneration_response_json' not in existing_columns:
    cursor.execute(
      "ALTER TABLE problems ADD COLUMN regeneration_response_json TEXT")
    log.info("Added regeneration_response_json column")

  if 'regeneration_cached_at' not in existing_columns:
    cursor.execute(
      "ALTER TABLE problems ADD COLUMN regeneration_cached_at TIMESTAMP")
    log.info("Added regeneration_cached_at column")


def update_problem_stats(session_id: int):
  """Update computed statistics for a session"""
  with get_db_connection() as conn:
    cursor = conn.cursor()

    # Get all problem numbers for this session
    cursor.execute(
      """
            SELECT DISTINCT problem_number
            FROM problems
            WHERE session_id = ?
        """, (session_id, ))

    problem_numbers = [row[0] for row in cursor.fetchall()]

    for problem_num in problem_numbers:
      # Calculate statistics
      cursor.execute(
        """
                SELECT
                    AVG(score) as avg_score,
                    MIN(score) as min_score,
                    MAX(score) as max_score,
                    SUM(CASE WHEN graded = 1 THEN 1 ELSE 0 END) as num_graded,
                    COUNT(*) as num_total
                FROM problems
                WHERE session_id = ? AND problem_number = ? AND graded = 1
            """, (session_id, problem_num))

      row = cursor.fetchone()
      avg_score, min_score, max_score, num_graded, num_total_graded = row[
        0], row[1], row[2], row[3], row[4]

      # Get total count (including ungraded)
      cursor.execute(
        """
                SELECT COUNT(*) FROM problems
                WHERE session_id = ? AND problem_number = ?
            """, (session_id, problem_num))
      num_total = cursor.fetchone()[0]

      # Upsert statistics
      cursor.execute(
        """
                INSERT INTO problem_stats (session_id, problem_number, avg_score, min_score, max_score, num_graded, num_total)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, problem_number)
                DO UPDATE SET
                    avg_score = excluded.avg_score,
                    min_score = excluded.min_score,
                    max_score = excluded.max_score,
                    num_graded = excluded.num_graded,
                    num_total = excluded.num_total,
                    updated_at = CURRENT_TIMESTAMP
            """, (session_id, problem_num, avg_score, min_score, max_score,
                  num_graded, num_total))
