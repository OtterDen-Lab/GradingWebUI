#!/bin/sh
# Resolve the native service bind address, then start Uvicorn.
set -eu

app_dir=${GRADING_APP_DIR:-/opt/grading-web}
bind_host=${GRADING_BIND_HOST:-tailnet}
bind_port=${GRADING_BIND_PORT:-8765}

case "$bind_host" in
  tailnet|tailscale|auto)
    command -v tailscale >/dev/null 2>&1 || {
      echo "GRADING_BIND_HOST=$bind_host requires the tailscale command." >&2
      exit 1
    }
    bind_host=$(tailscale ip -4 2>/dev/null | sed -n '1p' || true)
    if [ -z "$bind_host" ] && command -v ip >/dev/null 2>&1; then
      bind_host=$(ip -4 -o addr show dev tailscale0 2>/dev/null | \
        awk 'NR == 1 { sub(/\/.*/, "", $4); print $4 }')
    fi
    [ -n "$bind_host" ] || {
      echo "Could not determine a Tailscale IPv4 address." >&2
      exit 1
    }
    ;;
  localhost)
    bind_host=127.0.0.1
    ;;
  all)
    bind_host=0.0.0.0
    ;;
esac

echo "Starting GradingWebUI on $bind_host:$bind_port" >&2
exec "$app_dir/.venv/bin/python" -m uvicorn grading_web_ui.web_api.main:app \
  --host "$bind_host" --port "$bind_port"
