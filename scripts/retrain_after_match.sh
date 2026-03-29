#!/usr/bin/env bash
# Rebuild processed data and retrain all models after adding/updating ball-by-ball
# under matches/ (and data/raw/). Run from repo root: bash scripts/retrain_after_match.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> build_unified_dataset.py"
python3 build_unified_dataset.py

echo "==> build_player_match_stats.py"
python3 build_player_match_stats.py

echo "==> build_features.py"
python3 build_features.py

echo "==> build_training_dataset.py"
python3 build_training_dataset.py

echo "==> train_player_model.py"
python3 train_player_model.py

echo "==> train_match_winner_model.py"
python3 train_match_winner_model.py

echo "==> train_team_total_model.py"
python3 train_team_total_model.py

echo "==> scripts/build_player_registry.py"
python3 scripts/build_player_registry.py

echo ""
echo "Done. unified_ball_by_ball rows (incl. header): $(wc -l < data/processed/unified_ball_by_ball.csv)"
echo "Models updated under models/"
