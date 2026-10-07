#!/bin/sh
# Create the persistent encryption secret for per-user Canvas credentials.
set -eu

usage() {
  echo "Usage: $0 ENV_FILE" >&2
  exit 2
}

[ "$#" -eq 1 ] || usage
env_file=$1

[ -f "$env_file" ] || {
  echo "Environment file not found: $env_file" >&2
  exit 1
}

# A non-empty, uncommented assignment is the only acceptable existing value.
if grep -Eq '^[[:space:]]*CANVAS_CREDENTIAL_ENCRYPTION_KEY=.+$' "$env_file"; then
  chmod 0600 "$env_file"
  exit 0
fi

key=$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')
umask 077
printf '\n# Generated automatically; keep stable or saved Canvas keys cannot be decrypted.\nCANVAS_CREDENTIAL_ENCRYPTION_KEY=%s\n' "$key" >> "$env_file"
chmod 0600 "$env_file"
echo "Created Canvas credential encryption key in $env_file."
