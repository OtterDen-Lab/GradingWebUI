#!/bin/sh
# Update dependencies from an already-updated native LXC checkout and restart.
set -eu

app_dir="/opt/grading-web"
service_user="grading-web"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --app-dir) app_dir=$2; shift 2 ;;
    --service-user) service_user=$2; shift 2 ;;
    *) echo "Usage: sudo $0 [--app-dir DIR] [--service-user USER]" >&2; exit 2 ;;
  esac
done

[ "$(id -u)" -eq 0 ] || { echo "Run through sudo." >&2; exit 1; }
[ -x "$app_dir/.venv/bin/uv" ] || {
  echo "Native environment is not installed; run install_lxc_service.sh first." >&2
  exit 1
}
systemctl stop grading-web.service
runuser -u "$service_user" -- sh -c "cd '$app_dir' && '$app_dir/.venv/bin/uv' sync --frozen --no-dev"
systemctl start grading-web.service
systemctl --no-pager --full status grading-web.service
