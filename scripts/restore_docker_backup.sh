#!/bin/sh
# Restore a GradingWebUI SQLite backup into the running Compose deployment.
set -eu

usage() {
  echo "Usage: $0 --backup-file FILE [--compose-file FILE]" >&2
  exit 2
}

backup_file=""
compose_file="docker/web-grading/docker-compose.prod.yml"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --backup-file)
      [ "$#" -ge 2 ] || usage
      backup_file=$2
      shift 2
      ;;
    --compose-file)
      [ "$#" -ge 2 ] || usage
      compose_file=$2
      shift 2
      ;;
    *) usage ;;
  esac
done

[ -n "$backup_file" ] || usage
[ -f "$backup_file" ] || {
  echo "Restore failed: backup file does not exist: $backup_file" >&2
  exit 1
}

container_id=$(docker compose -f "$compose_file" ps -q web)
[ -n "$container_id" ] || {
  echo "Restore failed: could not find the web container." >&2
  exit 1
}
image=$(docker inspect -f '{{.Config.Image}}' "$container_id")

# The container must be stopped so no process has the database open while its
# main file and WAL sidecars are replaced.
docker compose -f "$compose_file" stop web

# This is a newly deployed destination volume. Limit deletion to SQLite's
# database and journal sidecars; no other files in /data are touched.
docker run --rm --volumes-from "$container_id" --entrypoint sh "$image" -c \
  'rm -f /data/grading.db /data/grading.db-wal /data/grading.db-shm'
docker cp "$backup_file" "$container_id:/data/grading.db"

integrity=$(docker run --rm --volumes-from "$container_id" --entrypoint python "$image" -c \
  "import sqlite3; connection=sqlite3.connect('/data/grading.db'); print(connection.execute('PRAGMA integrity_check').fetchone()[0]); connection.close()")
[ "$integrity" = "ok" ] || {
  echo "Restore failed: SQLite integrity check returned: $integrity" >&2
  exit 1
}

docker compose -f "$compose_file" start web
echo "Restored and started GradingWebUI from $backup_file"
