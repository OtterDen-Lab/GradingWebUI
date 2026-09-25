#!/bin/sh
# Restore a verified SQLite backup into a native LXC installation.
set -eu

usage() {
  echo "Usage: sudo $0 --backup-file FILE [--app-dir DIR] [--state-dir DIR] [--service-user USER] [--replace-existing]" >&2
  exit 2
}

backup_file=""
app_dir="/opt/grading-web"
state_dir="/srv/grading-web"
service_user="grading-web"
replace_existing=false
while [ "$#" -gt 0 ]; do
  case "$1" in
    --backup-file) backup_file=$2; shift 2 ;;
    --app-dir) app_dir=$2; shift 2 ;;
    --state-dir) state_dir=$2; shift 2 ;;
    --service-user) service_user=$2; shift 2 ;;
    --replace-existing) replace_existing=true; shift ;;
    *) usage ;;
  esac
done

[ "$(id -u)" -eq 0 ] || { echo "Run through sudo." >&2; exit 1; }
[ -n "$backup_file" ] && [ -f "$backup_file" ] || {
  echo "Backup file does not exist: $backup_file" >&2
  exit 1
}
mountpoint -q "$state_dir" || {
  echo "State directory is not a mounted filesystem: $state_dir" >&2
  exit 1
}
database_path="$state_dir/data/grading.db"
if [ -e "$database_path" ] && [ "$replace_existing" = false ]; then
  echo "Refusing to overwrite existing database: $database_path" >&2
  echo "Use a fresh state volume or pass --replace-existing after taking a backup." >&2
  exit 1
fi

systemctl stop grading-web.service
rm -f "$database_path" "$database_path-wal" "$database_path-shm"
install -o "$service_user" -g "$service_user" -m 0600 "$backup_file" "$database_path"
integrity=$(runuser -u "$service_user" -- env GRADING_RESTORE_DB_PATH="$database_path" \
  "$app_dir/.venv/bin/python" -c \
  "import os, sqlite3; connection=sqlite3.connect(os.environ['GRADING_RESTORE_DB_PATH']); print(connection.execute('PRAGMA integrity_check').fetchone()[0]); connection.close()")
[ "$integrity" = "ok" ] || {
  echo "Restore failed: SQLite integrity check returned: $integrity" >&2
  exit 1
}
systemctl start grading-web.service
echo "Restored and started GradingWebUI from $backup_file"
