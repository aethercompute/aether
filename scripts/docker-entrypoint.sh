#!/bin/sh
set -eu

export CONTROL_HOST="${CONTROL_HOST:-0.0.0.0}"
export CONTROL_USERNAME="${CONTROL_USERNAME:-admin}"

if [ -z "${CONTROL_PASSWORD:-}" ]; then
  password_file="${CONTROL_PASSWORD_FILE:-/app/.aether-control/control-password}"
  password_dir="$(dirname "$password_file")"
  mkdir -p "$password_dir"

  if [ ! -s "$password_file" ]; then
    umask 077
    python3 -c 'import secrets; print(secrets.token_urlsafe(24))' > "$password_file"
  fi

  CONTROL_PASSWORD="$(tr -d '\r\n' < "$password_file")"
  if [ -z "$CONTROL_PASSWORD" ]; then
    echo "control dashboard password file is empty: $password_file" >&2
    exit 1
  fi
  export CONTROL_PASSWORD

  echo "control dashboard credentials: ${CONTROL_USERNAME}:${CONTROL_PASSWORD}"
  echo "control dashboard password file: $password_file"
fi

exec python3 scripts/training-control-dashboard.py
