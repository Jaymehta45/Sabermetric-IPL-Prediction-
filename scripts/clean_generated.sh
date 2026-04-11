#!/usr/bin/env bash
# Remove local Python bytecode caches and macOS metadata (not tracked in git).
# Safe to run anytime: bash scripts/clean_generated.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

find . -name '.DS_Store' ! -path './.git/objects/*' -print -delete 2>/dev/null || true
while IFS= read -r -d '' d; do
  rm -rf "$d"
done < <(find . -type d -name '__pycache__' -print0 2>/dev/null)

echo "Removed __pycache__ dirs and .DS_Store under $ROOT (excluding .git/objects)."
