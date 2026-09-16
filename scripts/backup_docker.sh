#!/bin/sh
# Create a verified GradingWebUI backup from a running Docker deployment.
set -eu

usage() {
  echo "Usage: $0 --backup-dir DIRECTORY [--compose-file FILE]" >&2
  exit 2
}

backup_dir=""
compose_file="docker/web-grading/docker-compose.prod.yml"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --backup-dir)
      [ "$#" -ge 2 ] || usage
      backup_dir=$2
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

[ -n "$backup_dir" ] || usage
mkdir -p "$backup_dir"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
backup_name="grading-backup-$stamp.db"
container_path="/data/$backup_name"

container_id=$(docker compose -f "$compose_file" ps -q web)
[ -n "$container_id" ] || {
  echo "Backup failed: the web container is not running." >&2
  exit 1
}

docker compose -f "$compose_file" exec -T web \
  python /app/scripts/backup_db.py --output "$container_path"
docker cp "$container_id:$container_path" "$backup_dir/$backup_name"
docker cp "$container_id:$container_path.json" "$backup_dir/$backup_name.json"

echo "Copied verified backup to $backup_dir/$backup_name"
sha256sum "$backup_dir/$backup_name"
