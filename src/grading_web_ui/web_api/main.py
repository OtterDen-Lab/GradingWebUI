"""
Main FastAPI application entry point.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from dotenv import load_dotenv
import subprocess
import tomllib
import logging

# Load environment variables from .env file
load_dotenv()
# Version helpers
def _find_pyproject(start_path: Path) -> Path | None:
  for parent in [start_path, *start_path.parents]:
    candidate = parent / "pyproject.toml"
    if candidate.is_file():
      return candidate
  return None


def _read_project_version() -> str:
  pyproject = _find_pyproject(Path(__file__).resolve())
  if not pyproject:
    return "0.0.0"
  try:
    with pyproject.open("rb") as handle:
      data = tomllib.load(handle)
    return data.get("project", {}).get("version", "0.0.0")
  except Exception:
    return "0.0.0"


def _is_ahead_of_tag(version: str) -> bool:
  pyproject = _find_pyproject(Path(__file__).resolve())
  if not pyproject:
    return False
  repo_root = pyproject.parent
  if not (repo_root / ".git").exists():
    return False
  tag_name = f"v{version}"
  try:
    result = subprocess.run(
      ["git", "describe", "--tags", "--exact-match"],
      cwd=repo_root,
      check=False,
      capture_output=True,
      text=True,
    )
    if result.returncode != 0:
      return True
    return result.stdout.strip() != tag_name
  except Exception:
    return False


PROJECT_VERSION = _read_project_version()
DISPLAY_VERSION = f"v{PROJECT_VERSION}" + ("+" if _is_ahead_of_tag(PROJECT_VERSION) else "")


from .database import init_database, get_db_connection
from .services.quiz_encryption import install_quizgenerator_key_provider
from .startup_config import validate_startup_configuration
from .routes import sessions, problems, uploads, canvas, matching, finalize, ai_grader, alignment, feedback_tags, auth, assignments

# Optional debug routes (may not exist on all deployments)
try:
  from .routes import debug
  has_debug_routes = True
except ImportError:
  has_debug_routes = False


@asynccontextmanager
async def lifespan(app: FastAPI):
  """Lifespan event handler for startup/shutdown"""
  log = logging.getLogger(__name__)

  # Ensure QuizGenerator key access does not mutate process env at runtime.
  install_quizgenerator_key_provider()

  for warning in validate_startup_configuration():
    log.warning("Startup config: %s", warning)

  # Startup: Initialize database
  init_database()

  # Cleanup expired authentication sessions
  from .services.auth_service import AuthService
  auth_service = AuthService()
  with get_db_connection() as conn:
    auth_service.cleanup_expired_sessions(conn)

  yield

  # Shutdown: cleanup if needed
  pass


# Initialize FastAPI app
app = FastAPI(
  title="Web Grading API",
  description="API for web-based exam grading interface",
  version=PROJECT_VERSION,
  docs_url="/api/docs",
  redoc_url="/api/redoc",
  lifespan=lifespan,
)

# CORS middleware for development
app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:3000", "http://localhost:8765"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


# Health check endpoint (must be before static files mount)
@app.get("/api/health")
async def health_check():
  """Health check endpoint"""
  return {"status": "healthy", "version": PROJECT_VERSION}


@app.get("/api/version")
async def version_info():
  """Return project version information."""
  return {
    "version": PROJECT_VERSION,
    "display": DISPLAY_VERSION,
    "tag": f"v{PROJECT_VERSION}"
  }


# Include routers
app.include_router(auth.router,           prefix="/api/auth",           tags=["auth"])
app.include_router(sessions.router,       prefix="/api/sessions",       tags=["sessions"])
app.include_router(assignments.router,    prefix="/api/sessions",       tags=["assignments"])
app.include_router(problems.router,       prefix="/api/problems",       tags=["problems"])
app.include_router(uploads.router,        prefix="/api/uploads",        tags=["uploads"])
app.include_router(canvas.router,         prefix="/api/canvas",         tags=["canvas"])
app.include_router(matching.router,       prefix="/api/matching",       tags=["matching"])
app.include_router(finalize.router,       prefix="/api/finalize",       tags=["finalize"])
app.include_router(ai_grader.router,      prefix="/api/ai-grader",      tags=["ai-grader"])
app.include_router(alignment.router,      prefix="/api/alignment",      tags=["alignment"])
app.include_router(feedback_tags.router,  prefix="/api/feedback-tags",  tags=["feedback-tags"])

# Conditionally include debug routes if available
if has_debug_routes:
  app.include_router(debug.router,        prefix="/api",                tags=["debug"])

# Mount static files (frontend) - MUST BE LAST as it catches all routes
frontend_path = Path(__file__).parent.parent / "web_frontend"
if frontend_path.exists():
  app.mount(
    "/",
    StaticFiles(directory=str(frontend_path), html=True),
    name="static"
  )

if __name__ == "__main__":
  import uvicorn
  uvicorn.run("grading_web_ui.web_api.main:app",
              host="127.0.0.1",
              port=8765,
              reload=True)
