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

Use `docker/web-grading/docker-compose.yml` for development (local source mounted, fast iteration).
Use `docker/web-grading/docker-compose.prod.yml` for deployment-style runs (image-only, server env file).

### Local dev with Docker Compose (build from local source)

```bash
make docker-up-build
```

### Local server-style run (image + env file injection)

```bash
GRADING_WEB_IMAGE=ghcr.io/otterden-lab/gradingwebui:v0.5.3 \
GRADING_WEB_ENV_FILE=/etc/grading-web/web.env \
docker compose -f docker/web-grading/docker-compose.prod.yml up -d
```

### Update to a newer image tag

```bash
make docker-prod-pull DOCKER_IMAGE=ghcr.io/otterden-lab/gradingwebui:v0.5.4 DOCKER_ENV_FILE=/etc/grading-web/web.env
make docker-prod-up DOCKER_IMAGE=ghcr.io/otterden-lab/gradingwebui:v0.5.4 DOCKER_ENV_FILE=/etc/grading-web/web.env
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
