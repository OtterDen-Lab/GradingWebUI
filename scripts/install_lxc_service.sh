#!/bin/sh
# Install GradingWebUI as a native systemd service in a Debian/Ubuntu LXC.
set -eu

usage() {
  echo "Usage: sudo $0 [--app-dir DIR] [--state-dir DIR] [--service-user USER] [--python-bin PATH] [--skip-packages]" >&2
  exit 2
}

app_dir="/opt/grading-web"
state_dir="/srv/grading-web"
service_user="grading-web"
python_bin="python3"
skip_packages=false
while [ "$#" -gt 0 ]; do
  case "$1" in
    --app-dir) app_dir=$2; shift 2 ;;
    --state-dir) state_dir=$2; shift 2 ;;
    --service-user) service_user=$2; shift 2 ;;
    --python-bin) python_bin=$2; shift 2 ;;
    --skip-packages) skip_packages=true; shift ;;
    *) usage ;;
  esac
done

[ "$(id -u)" -eq 0 ] || {
  echo "Run this installer as root (usually through sudo)." >&2
  exit 1
}
[ -f "$app_dir/pyproject.toml" ] || {
  echo "Application checkout not found at $app_dir" >&2
  exit 1
}
[ -f "$app_dir/deploy/lxc/grading-web.service.template" ] || {
  echo "Missing LXC systemd service template in $app_dir" >&2
  exit 1
}
[ -x "$app_dir/scripts/run_lxc_service.sh" ] || {
  echo "Missing native service runner in $app_dir/scripts" >&2
  exit 1
}
mountpoint -q "$state_dir" || {
  echo "State directory is not a mounted filesystem: $state_dir" >&2
  echo "Attach the Proxmox storage volume there before installing." >&2
  exit 1
}

if [ "$skip_packages" = false ]; then
  apt-get update
  apt-get install -y \
    python3 python3-venv git \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libzbar0
fi
command -v "$python_bin" >/dev/null 2>&1 || {
  echo "Python executable not found: $python_bin" >&2
  exit 1
}
"$python_bin" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' || {
  echo "Python 3.12 or newer is required; $python_bin is too old." >&2
  exit 1
}

if ! id "$service_user" >/dev/null 2>&1; then
  useradd --system --create-home --home-dir "/var/lib/$service_user" \
    --shell /usr/sbin/nologin "$service_user"
fi

install -d -m 0750 -o root -g "$service_user" "$state_dir"
install -d -m 0700 -o "$service_user" -g "$service_user" \
  "$state_dir/data" "$state_dir/tmp" "$state_dir/backup-staging" "$state_dir/logs"
# /var/log is root-owned, so the service needs its own writable subdirectory.
install -d -m 0750 -o "$service_user" -g "$service_user" /var/log/grading-ui
install -d -m 0750 -o root -g root "$state_dir/config"
if [ ! -e "$state_dir/config/web.env" ]; then
  install -m 0600 -o root -g root "$app_dir/.env.example" \
    "$state_dir/config/web.env"
  echo "Created $state_dir/config/web.env; set real credentials before starting the service."
fi
if [ ! -e "$state_dir/config/backup.env.example" ]; then
  install -m 0600 -o root -g root "$app_dir/deploy/lxc/backup.env.example" \
    "$state_dir/config/backup.env.example"
fi

"$python_bin" -m venv "$app_dir/.venv"
chown -R "$service_user:$service_user" "$app_dir/.venv"
runuser -u "$service_user" -- "$app_dir/.venv/bin/pip" install --upgrade pip uv
runuser -u "$service_user" -- sh -c "cd '$app_dir' && '$app_dir/.venv/bin/uv' sync --frozen --no-dev"

sed \
  -e "s|__APP_DIR__|$app_dir|g" \
  -e "s|__STATE_DIR__|$state_dir|g" \
  -e "s|__SERVICE_USER__|$service_user|g" \
  "$app_dir/deploy/lxc/grading-web.service.template" \
  > /etc/systemd/system/grading-web.service
chmod 0644 /etc/systemd/system/grading-web.service
sed \
  -e "s|__APP_DIR__|$app_dir|g" \
  -e "s|__STATE_DIR__|$state_dir|g" \
  "$app_dir/deploy/lxc/grading-web-backup.service.template" \
  > /etc/systemd/system/grading-web-backup.service
install -m 0644 "$app_dir/deploy/lxc/grading-web-backup.timer" \
  /etc/systemd/system/grading-web-backup.timer
systemctl daemon-reload
systemctl enable grading-web.service

echo "Installed grading-web.service. Edit $state_dir/config/web.env, then run:"
echo "  sudo systemctl start grading-web"
echo "To enable daily backups, create $state_dir/config/backup.env and run:"
echo "  sudo systemctl enable --now grading-web-backup.timer"
