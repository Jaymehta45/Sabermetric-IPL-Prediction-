#!/usr/bin/env bash
# Rebuild processed data and retrain all models after adding/updating ball-by-ball
# under matches/ (and data/raw/). Run from repo root: bash scripts/retrain_after_match.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> iplpred.pipeline.build_unified_dataset"
python3 -m iplpred.pipeline.build_unified_dataset

echo "==> iplpred.pipeline.build_team_franchise_profiles"
python3 -m iplpred.pipeline.build_team_franchise_profiles

echo "==> iplpred.pipeline.build_player_match_stats"
python3 -m iplpred.pipeline.build_player_match_stats

echo "==> iplpred.pipeline.build_features"
python3 -m iplpred.pipeline.build_features

echo "==> iplpred.pipeline.build_training_dataset"
python3 -m iplpred.pipeline.build_training_dataset

echo "==> iplpred.training.train_player_model"
python3 -m iplpred.training.train_player_model

echo "==> iplpred.training.train_match_winner_model"
python3 -m iplpred.training.train_match_winner_model

echo "==> iplpred.training.train_win_prob_ensemble (hybrid ML + MC stack / calibration; requires match winner model)"
python3 -m iplpred.training.train_win_prob_ensemble

echo "==> iplpred.training.train_team_total_model"
python3 -m iplpred.training.train_team_total_model

echo "==> scripts/build_player_registry.py"
python3 scripts/build_player_registry.py

echo ""
echo "Done. unified_ball_by_ball rows (incl. header): $(wc -l < data/processed/unified_ball_by_ball.csv)"
echo "Models updated under models/"
