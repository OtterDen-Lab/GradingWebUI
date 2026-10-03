#!/bin/sh
# Update dependencies from an already-updated native LXC checkout and restart.
set -eu

app_dir="/opt/grading-web"
state_dir="/srv/grading-web"
service_user="grading-web"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --app-dir) app_dir=$2; shift 2 ;;
    --state-dir) state_dir=$2; shift 2 ;;
    --service-user) service_user=$2; shift 2 ;;
    *) echo "Usage: sudo $0 [--app-dir DIR] [--state-dir DIR] [--service-user USER]" >&2; exit 2 ;;
  esac
done

[ "$(id -u)" -eq 0 ] || { echo "Run through sudo." >&2; exit 1; }
[ -x "$app_dir/.venv/bin/uv" ] || {
  echo "Native environment is not installed; run install_lxc_service.sh first." >&2
  exit 1
}
systemctl stop grading-web.service
runuser -u "$service_user" -- sh -c "cd '$app_dir' && '$app_dir/.venv/bin/uv' sync --frozen --no-dev"
# Re-render the unit so deployment changes to environment paths take effect.
install -d -m 0750 -o "$service_user" -g "$service_user" /var/log/grading-ui
sed \
  -e "s|__APP_DIR__|$app_dir|g" \
  -e "s|__STATE_DIR__|$state_dir|g" \
  -e "s|__SERVICE_USER__|$service_user|g" \
  "$app_dir/deploy/lxc/grading-web.service.template" \
  > /etc/systemd/system/grading-web.service
chmod 0644 /etc/systemd/system/grading-web.service
systemctl daemon-reload
systemctl start grading-web.service
systemctl --no-pager --full status grading-web.service
