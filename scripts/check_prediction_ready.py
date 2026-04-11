#!/usr/bin/env python3
"""
Verify model artifacts needed for Process 1 (predict_match_outcomes / export-json).

Exit 0 if all required files exist; exit 1 with a short list of missing paths.
Optional: run after `make build-all`, train_* scripts, and `make train-win-ensemble`.

Usage (repo root):
  python3 scripts/check_prediction_ready.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    ROOT / "data" / "processed" / "player_features.csv",
    ROOT / "models" / "player_runs_model.pkl",
    ROOT / "models" / "player_wickets_model.pkl",
    ROOT / "models" / "match_winner_classifier.pkl",
    ROOT / "models" / "team_total_regressor.pkl",
]

RECOMMENDED = [
    ROOT / "models" / "win_prob_ensemble.pkl",
]


def main() -> None:
    missing = [p for p in REQUIRED if not p.is_file()]
    if missing:
        print("Missing required files for Process 1:", file=sys.stderr)
        for p in missing:
            print(f"  - {p.relative_to(ROOT)}", file=sys.stderr)
        raise SystemExit(1)

    soft = [p for p in RECOMMENDED if not p.is_file()]
    if soft:
        print("Optional (hybrid win prob uses fixed0.6/0.4 blend if absent):")
        for p in soft:
            print(f"  - {p.relative_to(ROOT)}")
        print("  Run: python3 -m iplpred.training.train_win_prob_ensemble")

    print("OK — Process 1 prerequisites present.")


if __name__ == "__main__":
    main()
