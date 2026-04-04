#!/usr/bin/env bash
# Commit and push prediction_log.csv so the Vercel app (GitHub raw fetch) shows the latest P1/P2/P3 state.
set -euo pipefail
cd "$(dirname "$0")/.."
git add data/processed/prediction_log.csv
if git diff --cached --quiet; then
  echo "No staged changes to data/processed/prediction_log.csv"
  exit 0
fi
git commit -m "chore: sync prediction_log.csv ($(date -u +%Y-%m-%dT%H:%MZ))"
git push
echo "Pushed. Live site refreshes from GitHub (short server cache may delay up to ~30s)."
