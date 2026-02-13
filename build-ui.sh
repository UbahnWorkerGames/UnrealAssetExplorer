#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[error] $1 is not in PATH" >&2
    exit 1
  fi
}

require_cmd npm

echo "[build] Installing frontend dependencies..."
cd "$FRONTEND_DIR"
if [[ -f package-lock.json ]]; then
  npm ci
else
  npm install
fi

echo "[build] Building frontend dist..."
VITE_BASE=/ui/ npm run build

echo "[ok] Built frontend: $FRONTEND_DIR/dist"
