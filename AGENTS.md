# Repository Guidelines

## Project Structure & Module Organization
- `src/grading_web_ui/` is the main Python package.
  - `web_api/` contains the FastAPI backend (routes, services, models, database).
  - `web_frontend/` holds the static frontend assets.
  - `lms_interface/` provides Canvas and AI helper integrations.
- `tests/` contains unit and integration tests.
- `docker/web-grading/` contains Docker build and compose files.
- `WebUI/` contains documentation and planning notes.

## Build, Test, and Development Commands
- `pip install -e .` installs the package in editable mode.
- `python -m grading_web_ui.web_api.main` runs the API locally.
- `docker compose -f docker/web-grading/docker-compose.yml up --build` builds and runs the Dockerized web UI.
- `pytest` runs the test suite.

## Coding Style & Naming Conventions
- Python: use 2-space indentation by default, and convert existing 4-space blocks to 2-space when you touch them.
- Module imports should use the package prefix, e.g. `from grading_web_ui.web_api import ...`.
- API routes live under `src/grading_web_ui/web_api/routes/`; services under `services/`.

## Testing Guidelines
- Framework: `pytest`.
- Test files live in `tests/` and are named `test_*.py`.
- Add unit tests for new services; add integration tests when changing API flows.
- Run locally with `pytest` from the repo root.

## Commit & Pull Request Guidelines
- Commit conventions are not established (git history was reset). Use concise, imperative messages without emojis (e.g., "Add session export").
- PRs should include: summary, testing notes, and any UI screenshots if frontend changes are involved.
- Link relevant issues when applicable.

## Security & Configuration Tips
- Do not commit secrets. Use `.env` for Canvas and AI credentials.
- If secrets are exposed, rotate keys immediately and rewrite history if needed.
