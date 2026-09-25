#!/bin/sh
# Create a verified SQLite backup from a native LXC installation.
set -eu

usage() {
  echo "Usage: $0 --backup-dir DIRECTORY [--app-dir DIR] [--state-dir DIR]" >&2
  exit 2
}

backup_dir=""
app_dir="/opt/grading-web"
state_dir="/srv/grading-web"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --backup-dir) backup_dir=$2; shift 2 ;;
    --app-dir) app_dir=$2; shift 2 ;;
    --state-dir) state_dir=$2; shift 2 ;;
    *) usage ;;
  esac
done

[ -n "$backup_dir" ] || usage
[ -x "$app_dir/.venv/bin/python" ] || {
  echo "Native environment is not installed: $app_dir/.venv" >&2
  exit 1
}
[ -d "$backup_dir" ] && mountpoint -q "$backup_dir" || {
  echo "Backup directory must be an existing, separately mounted filesystem: $backup_dir" >&2
  exit 1
}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
"$app_dir/.venv/bin/python" "$app_dir/scripts/backup_db.py" \
  --db-path "$state_dir/data/grading.db" \
  --output "$backup_dir/grading-backup-$stamp.db"
