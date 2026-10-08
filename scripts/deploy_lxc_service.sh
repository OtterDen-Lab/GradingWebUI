#!/bin/sh
# Update dependencies from a native LXC checkout and restart.
set -eu

app_dir="/opt/grading-web"
state_dir="/srv/grading-web"
service_user="grading-web"
tag=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --app-dir) app_dir=$2; shift 2 ;;
    --state-dir) state_dir=$2; shift 2 ;;
    --service-user) service_user=$2; shift 2 ;;
    --tag) tag=$2; shift 2 ;;
    *) echo "Usage: sudo $0 [--app-dir DIR] [--state-dir DIR] [--service-user USER] [--tag vX.Y.Z]" >&2; exit 2 ;;
  esac
done

[ "$(id -u)" -eq 0 ] || { echo "Run through sudo." >&2; exit 1; }
[ -x "$app_dir/.venv/bin/uv" ] || {
  echo "Native environment is not installed; run install_lxc_service.sh first." >&2
  exit 1
}
[ -f "$state_dir/config/web.env" ] || {
  echo "Environment file not found: $state_dir/config/web.env; run install_lxc_service.sh first." >&2
  exit 1
}

if [ -n "$tag" ]; then
  [ -d "$app_dir/.git" ] || {
    echo "Cannot deploy tag $tag: $app_dir is not a Git checkout." >&2
    exit 1
  }
  [ -z "$(git -C "$app_dir" status --porcelain)" ] || {
    echo "Cannot deploy tag $tag: $app_dir has uncommitted changes." >&2
    echo "Commit, stash, or discard them before deploying a release tag." >&2
    exit 1
  }
  git -C "$app_dir" fetch origin --tags --prune
  git -C "$app_dir" rev-parse --verify --quiet "refs/tags/$tag^{commit}" >/dev/null || {
    echo "Release tag not found on origin: $tag" >&2
    exit 1
  }
  echo "Switching native checkout to release tag $tag"
  git -C "$app_dir" checkout --detach "refs/tags/$tag"
fi

systemctl stop grading-web.service
"$app_dir/scripts/ensure_canvas_credential_key.sh" "$state_dir/config/web.env"
# QuizGenerator and this application both use the LMS interface.  Reinstall it
# during upgrades: uv otherwise trusts its dist-info metadata, which cannot
# detect a partially populated namespace package from an interrupted install.
runuser -u "$service_user" -- sh -c "cd '$app_dir' && '$app_dir/.venv/bin/uv' sync --frozen --no-dev --reinstall-package otterden-lms-interface"
runuser -u "$service_user" -- "$app_dir/.venv/bin/python" -c \
  "from lms_interface.canvas_interface import CanvasInterface; print('Verified CanvasInterface:', CanvasInterface.__name__)"
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
