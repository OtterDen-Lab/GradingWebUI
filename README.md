# GradingWebUI

A web-based interface for grading exams with Canvas LMS integration and AI-assisted workflows.

## Installation

```bash
pip install -e .
```

## LMSInterface Dependency

Install local hooks and the `git bump` alias:

```bash
bash scripts/install_git_hooks.sh
```

LMSInterface is consumed as a pinned dependency in `pyproject.toml`:
`lms-interface @ git+https://github.com/OtterDen-Lab/LMSInterface.git@v0.5.0`

Bump version + test + commit:

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
make debug
```

(`make dev` is kept as an alias.)

### Build and run local Docker image

```bash
make run
```

Optional overrides:

```bash
make run RUN_IMAGE=autograder-web-grading:mytag RUN_ENV_FILE=.env
```

### Publish a release image

```bash
make publish v0.8.1
```

This pushes:
- `samogden/webgraderui:v0.8.1`
- `samogden/webgraderui:latest`

Default platform is `linux/amd64`.

To publish multi-arch (requires buildx/QEMU support on your host):

```bash
make publish v0.8.1 PUBLISH_PLATFORMS=linux/amd64,linux/arm64
```

### Deploy container with env-file validation

```bash
make deploy v0.8.1 DEPLOY_ENV_FILE=/etc/grading-web/web.env
```

Or deploy `latest`:

```bash
make deploy DEPLOY_ENV_FILE=/etc/grading-web/web.env
```

This validates env configuration and starts
`docker/web-grading/docker-compose.prod.yml` with:
- `GRADING_WEB_IMAGE=$(REGISTRY_IMAGE):$(DEPLOY_TAG)`
- `GRADING_WEB_ENV_FILE=$(DEPLOY_ENV_FILE)`

One-time bootstrap admin without editing env file:

```bash
GRADING_BOOTSTRAP_ADMIN_PASSWORD='use-a-strong-temp-secret' \
make deploy DEPLOY_ENV_FILE=/etc/grading-web/web.env
```

After first successful login, remove that variable from your shell/session
before later deploys.

If you deploy from a remote registry image instead of a local build, use
direct Compose commands:

```bash
GRADING_WEB_IMAGE=samogden/webgraderui:v0.8.1 \
GRADING_WEB_ENV_FILE=/etc/grading-web/web.env \
docker compose -f docker/web-grading/docker-compose.prod.yml pull

GRADING_WEB_IMAGE=samogden/webgraderui:v0.8.1 \
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

## Full backups, migration, and analysis

All persistent grading data—including submission PDFs, scores, feedback, and
session configuration—is stored in the SQLite database. For a complete move or
backup, use a **database snapshot**, not the per-session JSON export in the UI.
The snapshot is a self-contained `.db` file and can be made while the app is
running; it includes committed WAL data and is verified with SQLite's integrity
check.

### Docker server: make a backup before decommissioning

Run this from the repository on the server. Replace `/srv/grading-backups` with
a directory on storage that will survive the server (mounted backup disk or a
synced off-server directory). The copy step is deliberately outside Docker's
named volume:

```bash
make backup BACKUP_DIR=/srv/grading-backups
```

This is a host-side command; it invokes Docker to run the snapshot utility in
the container, then copies the resulting database and manifest to `BACKUP_DIR`.
If you need the equivalent individual commands, they are:

```bash
backup_dir=/srv/grading-backups
stamp=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$backup_dir"

docker compose -f docker/web-grading/docker-compose.prod.yml exec -T web \
  python /app/scripts/backup_db.py --output "/data/grading-backup-$stamp.db"

container_id=$(docker compose -f docker/web-grading/docker-compose.prod.yml ps -q web)
docker cp "$container_id:/data/grading-backup-$stamp.db" "$backup_dir/"
docker cp "$container_id:/data/grading-backup-$stamp.db.json" "$backup_dir/"
sha256sum "$backup_dir/grading-backup-$stamp.db"
```

The command prints the database SHA-256 and schema version, and writes the same
information to its adjacent JSON manifest. Keep both files. After copying, test
the backup on another machine with `sqlite3 backup.db 'PRAGMA integrity_check;'`;
the answer must be `ok`.

For regular backups, schedule the same commands with cron or your backup system
and retain several dated copies. The destination must be outside the Docker
volume and preferably on a different host/storage account; a backup only in
`grading-data` will be lost with the server.

If the running server has an older image that does not yet contain
`/app/scripts/backup_db.py`, use this one-time equivalent before upgrading it.
It also uses SQLite's consistent backup API; check that it prints `ok` before
copying the `.db` file with `docker cp` and recording its `sha256sum`.

```bash
container_id=$(docker compose -f docker/web-grading/docker-compose.prod.yml ps -q web)
stamp=$(date -u +%Y%m%dT%H%M%SZ)
docker exec "$container_id" python -c \
  "import sqlite3; source=sqlite3.connect('file:/data/grading.db?mode=ro', uri=True); target=sqlite3.connect('/data/grading-backup-$stamp.db'); source.backup(target); target.commit(); print(target.execute('PRAGMA integrity_check').fetchone()[0]); target.close(); source.close()"
```

### Restore or migrate to a new Docker host

1. Deploy a compatible or newer application image on the new host, then stop
   the `web` container.
2. Copy the saved `.db` into the new container at `/data/grading.db` (or into
   the named volume) and start the container. The application will migrate an
   older supported schema automatically and creates a pre-migration backup.
3. Sign in and verify a known session, submission PDF, and score before retiring
   the old server. Preserve the old backup until that check succeeds.

The instructor account records are in this database. Canvas/API credentials are
not: recreate the protected env file on the new host.

### Analysis

The backup is a standard SQLite database, so it can be opened read-only with
SQLite, DB Browser for SQLite, Python/pandas, or exported to CSV later without
touching the live grading system. For example:

```bash
sqlite3 -readonly grading-backup-YYYYMMDDTHHMMSSZ.db '.tables'
sqlite3 -readonly grading-backup-YYYYMMDDTHHMMSSZ.db \
  'SELECT session_id, problem_number, score, feedback FROM problems;' > problems.tsv
```

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
