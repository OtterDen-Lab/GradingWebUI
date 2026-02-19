# GradingWebUI

A web-based interface for grading exams with Canvas LMS integration and AI-assisted workflows.

## Installation

```bash
pip install -e .
```

## LMSInterface Vendoring Workflow

Install local hooks and the `git bump` alias:

```bash
bash scripts/install_git_hooks.sh
```

Refresh vendored LMSInterface code manually:

```bash
python scripts/vendor_lms_interface.py --quiet
```

Bump version + vendor + test + commit:

```bash
git bump patch
```

## Quick Start

### 1. Set up environment variables

Copy `.env.example` to `.env` and set values:

```bash
cp .env.example .env

# Required Canvas settings
CANVAS_API_KEY=your_canvas_api_key_here
CANVAS_API_URL=https://your-institution.instructure.com

# Required for first login bootstrap
GRADING_BOOTSTRAP_ADMIN_PASSWORD=choose_a_strong_password
```

On first startup, an initial instructor user is created only when
`GRADING_BOOTSTRAP_ADMIN_PASSWORD` is set.

### 2. Run the server

```bash
python -m grading_web_ui.web_api.main
```

### 3. Docker quick start (optional)

From the repo root:

```bash
cd docker/web-grading
docker compose up --build
```

Then open:

```
http://localhost:8765
```

For more details, see `docker/web-grading/README.md`.

## Deployment Quick Reference

Primary workflows from the repo root:

### Run local dev server

```bash
make dev
```

(`make run` is kept as an alias.)

### Build a local Docker image

```bash
make image
```

Or set a custom tag:

```bash
make image DOCKER_IMAGE=autograder-web-grading:mytag
```

### Deploy container with env-file validation

```bash
make deploy DOCKER_ENV_FILE=/etc/grading-web/web.env
```

This validates env configuration, builds the image if needed, and starts
`docker/web-grading/docker-compose.prod.yml` with:
- `GRADING_WEB_IMAGE=$(DOCKER_IMAGE)`
- `GRADING_WEB_ENV_FILE=$(DOCKER_ENV_FILE)`

Named volumes (including `grading-data`) are preserved.

One-time bootstrap admin without editing env file:

```bash
GRADING_BOOTSTRAP_ADMIN_PASSWORD='use-a-strong-temp-secret' \
make deploy DOCKER_ENV_FILE=/etc/grading-web/web.env
```

After first successful login, remove that variable from your shell/session
before later deploys.

If you deploy from a remote registry image instead of a local build, use
direct Compose commands:

```bash
GRADING_WEB_IMAGE=ghcr.io/otterden-lab/gradingwebui:v0.5.4 \
GRADING_WEB_ENV_FILE=/etc/grading-web/web.env \
docker compose -f docker/web-grading/docker-compose.prod.yml pull

GRADING_WEB_IMAGE=ghcr.io/otterden-lab/gradingwebui:v0.5.4 \
GRADING_WEB_ENV_FILE=/etc/grading-web/web.env \
docker compose -f docker/web-grading/docker-compose.prod.yml up -d
```

## Features

### Web UI Capabilities

- **Problem-first grading**: Grade all Q1, then Q2, etc. with intelligent ordering
- **AI assistance**: Name extraction, blank detection, handwriting transcription
- **Canvas integration**: Dev/prod environment switching with safe defaults
- **Persistent sessions**: Resume grading anytime with database-backed state
- **Duplicate detection**: SHA256 hashing prevents re-processing the same files
- **Local storage**: SQLite database for FERPA-friendly workflows

## Configuration

The web UI reads Canvas credentials from `~/.env` by default. To use production Canvas, add:

```bash
USE_PROD_CANVAS=true
CANVAS_API_KEY_PROD=your_prod_key
CANVAS_API_URL_PROD=https://your-institution.instructure.com
```

Authentication cookie defaults can be controlled with:

```bash
AUTH_COOKIE_SECURE=true
AUTH_COOKIE_SAMESITE=lax
```

Startup config is validated at launch. Keep strict validation enabled in production:

```bash
GRADING_STRICT_STARTUP_CONFIG=true
```

## Dependency Management

Runtime dependencies are pinned in `pyproject.toml`, and CI installs via `uv sync --frozen` using `uv.lock`.

Recommended update cadence:

```bash
# Monthly (or before release)
uv lock --upgrade
uv sync --extra dev
uv run pytest -q
uv run pip-audit \
  --ignore-vuln GHSA-6vgw-5pg2-w6jp \
  --ignore-vuln GHSA-8rrh-rw8j-w5fx
```

During dependency updates, review upstream changelogs for FastAPI, Pydantic, Uvicorn, and QuizGenerator before merging.

## Database Migration Runbook

Run migrations explicitly before deployment cutovers:

```bash
python scripts/migrate_db.py --db-path /path/to/grading.db --backup-dir /path/to/backups
```

Rollback guidance:

1. Stop the app process.
2. Restore the most recent backup created before migration:
`cp /path/to/backups/grading.db.v<old>.bak-<timestamp> /path/to/grading.db`
3. Restart the app on the matching application version for that schema.

Preflight safety checks:

- By default, migrations create a backup (`GRADING_DB_CREATE_MIGRATION_BACKUP=true`).
- Backup location defaults to the DB directory and can be overridden with
`GRADING_DB_MIGRATION_BACKUP_DIR`.
- Migration aborts if there is not enough free disk for the backup copy.

## Requirements

- Python >= 3.12
- Docker (optional, for containerized runs)
- Canvas API access
- Optional: OpenAI/Anthropic/Ollama for AI-powered features

## Documentation

For detailed documentation, see `WebUI/docs`.

## License

This project is licensed under the GPL-3.0-or-later license. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/OtterDen-Lab/GradingWebUI).
