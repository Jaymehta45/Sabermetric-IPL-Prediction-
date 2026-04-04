#!/usr/bin/env bash
# Quick public URL for the local FastAPI app via Cloudflare Try (requires `cloudflared` on PATH).
# Usage: start the app first, e.g.  uvicorn web.dashboard:app --host 127.0.0.1 --port 8000
# Then:  bash scripts/cloudflared_tunnel.sh
# Use 127.0.0.1 (not "localhost") to avoid IPv6 / resolver edge cases.

set -euo pipefail
PORT="${PORT:-8000}"
URL="http://127.0.0.1:${PORT}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared not found. Install: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
  exit 1
fi

if ! curl -sf -o /dev/null --connect-timeout 1 "${URL}/"; then
  echo "Nothing responded at ${URL} — start the web app first, e.g.:"
  echo "  cd $(dirname "$0")/.. && python3 -m uvicorn web.dashboard:app --host 127.0.0.1 --port ${PORT}"
  exit 1
fi

exec cloudflared tunnel --url "${URL}"
