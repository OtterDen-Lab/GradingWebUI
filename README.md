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

LMSInterface is consumed from PyPI as the pinned
[`otterden-lms-interface`](https://pypi.org/project/otterden-lms-interface/)
dependency in `pyproject.toml`.

Bump version + test + commit:

```bash
git bump patch
```

## Quick Start

### 1. Set up environment variables

Copy `.env.example` to `.env` and set values:

```bash
cp .env.example .env

# Required Canvas settings. Users add their own API keys after signing in.
CANVAS_API_URL=https://your-institution.instructure.com
CANVAS_CREDENTIAL_ENCRYPTION_KEY=choose_a_long_random_server_secret

# Required for first login bootstrap
GRADING_BOOTSTRAP_ADMIN_PASSWORD=choose_a_strong_password
```

On first startup, an initial instructor user is created only when
`GRADING_BOOTSTRAP_ADMIN_PASSWORD` is set.

For native LXC installs, `make lxc-install` generates and persists
`CANVAS_CREDENTIAL_ENCRYPTION_KEY` in the LXC's `web.env` when it is absent.
`make lxc-deploy` performs the same idempotent check. The value is not printed;
back up the persistent `web.env` file and do not replace this value after users
have saved Canvas keys. To test only this setup step, run
`make lxc-ensure-canvas-key` on the LXC host after `lxc-install` has created
the state configuration.

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

### Deploy natively in a Proxmox LXC

The LXC deployment is designed so the container and application checkout are
replaceable while one detachable Proxmox mount point holds all durable state.
Use a local Proxmox/ZFS/ext4-backed volume (not SMB/NFS/FUSE) and attach it in
the LXC at `/srv/grading-web`. The service refuses to start if that mount is
missing, preventing an accidental empty database on the LXC root filesystem.

Inside a Debian/Ubuntu LXC with Python 3.12 or newer, clone this repository at
`/opt/grading-web`, attach the storage volume at `/srv/grading-web`, and run:

```bash
cd /opt/grading-web
make lxc-install
sudoedit /srv/grading-web/config/web.env
sudo systemctl start grading-web
```

In a minimal LXC where you are already root and `sudo` is not installed, use
`editor` (or `vi`/`nano`) instead of `sudoedit` and run `systemctl` directly.
The native Make targets detect root and do not invoke `sudo` in that case. The
web process itself still runs as the dedicated unprivileged `grading-web` user.

The installer creates this state layout on the mounted volume:

```text
/srv/grading-web/
├── config/web.env       # secrets and service configuration, mode 0600
├── data/grading.db      # grading records, users, scores, and submission PDFs
├── logs/                 # application and error logs, owned by grading-web
├── tmp/                 # in-progress upload/alignment files
└── backup-staging/      # optional local staging only
```

The repository and virtual environment remain at `/opt/grading-web`; changing
or replacing them does not alter the state volume. By default, the native
service binds only to the current Tailscale IPv4 address, discovered with
`tailscale ip -4`. A reverse proxy on another Tailnet host can therefore target
that address without exposing the service on the LXC's LAN interface. Restrict
access further with Tailscale ACLs.

Override the bind policy in `config/web.env` only when needed:

```ini
# Default when omitted: tailnet (the address reported by `tailscale ip -4`)
GRADING_BIND_HOST=tailnet
# Alternatives: localhost, all, or an explicit IPv4/IPv6 address
# GRADING_BIND_HOST=localhost
GRADING_BIND_PORT=8765
```

Configure Caddy/nginx for HTTPS to the chosen address and keep
`AUTH_COOKIE_SECURE=true`. For a temporary HTTP-only internal deployment, set
`AUTH_COOKIE_SECURE=false` explicitly.

For an upgrade, update the checkout using your normal Git/release process, then
run `make lxc-deploy`. The native upgrade path refreshes the LMS interface and
verifies its Canvas import before restarting the service, so a damaged virtual
environment is caught during deployment. To deploy an exact immutable release tag in one step,
run `make lxc-deploy v0.12.1`; the command fetches that tag, refuses to replace
a checkout with uncommitted changes, switches to the tag in detached-HEAD mode,
and then synchronizes dependencies and restarts the service. An exact tag
checkout displays its release version without a `+` suffix in the UI. To create
a consistent database snapshot, run:

```bash
make lxc-backup BACKUP_DIR=/mnt/off-host-backups/grading-web
```

`BACKUP_DIR` must itself be the mount point of separate storage (or be copied
off-host afterward); the detachable state volume protects against LXC
replacement, not disk failure.

To schedule a daily backup, create the protected backup destination setting and
enable the supplied systemd timer:

```bash
sudo cp /srv/grading-web/config/backup.env.example /srv/grading-web/config/backup.env
sudoedit /srv/grading-web/config/backup.env
make lxc-enable-backups
```

Set `GRADING_BACKUP_DIR` to a separate mounted disk or directory synchronized
off-host. Check recent executions with `systemctl status grading-web-backup.timer`
and `journalctl -u grading-web-backup.service`.

To restore into a freshly attached state volume after installing the service:

```bash
make lxc-restore BACKUP_FILE=/path/to/grading-backup-YYYYMMDDTHHMMSSZ.db
```

Do not start `grading-web` before this restore command; the installer enables
the service but does not start it, so the fresh state volume has no database to
replace. The restore command refuses to overwrite a database that already
exists. This is intentional. On a non-empty destination, make a current backup
first, then use `scripts/restore_lxc_backup.sh --replace-existing` only after
confirming the target paths.

For an unprivileged LXC, configure the Proxmox mount point's UID/GID mapping so
the `grading-web` service user can write `/srv/grading-web/data` and `tmp`.
Verify this before installation with `sudo -u grading-web touch
/srv/grading-web/data/.write-test`, then remove the test file.

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
CI runs `pip-audit` as a report-only step: findings remain visible in the
workflow output but do not block a release. Review each finding for
reachability, document accepted risk by advisory ID, and revisit it during the
next dependency update.

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

On the new host, create its protected environment file first, then run:

```bash
make deploy-from-backup v0.10.1 \
  DEPLOY_ENV_FILE=/etc/grading-web/web.env \
  BACKUP_FILE=/path/to/grading-backup-YYYYMMDDTHHMMSSZ.db
```

This deploys the release, stops its newly created `web` container, replaces only
the fresh database and SQLite WAL sidecars in its volume, verifies
`PRAGMA integrity_check`, and starts it again. The application will migrate an
older supported schema automatically and creates a pre-migration backup.

Sign in and verify a known session, submission PDF, and score before retiring
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
- Optional: OpenAI/Anthropic for AI-powered features

## Documentation

For detailed documentation, see `WebUI/docs`.

## License

This project is licensed under the GPL-3.0-or-later license. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/OtterDen-Lab/GradingWebUI).
