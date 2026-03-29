#!/usr/bin/env python3
"""
Compute Brier / log loss on official-style match rows when you have predicted P(team1).

CSV columns expected:
  - winner: team1 | team2 | tie
  - p_team1: optional column with your predicted probability team1 wins

Without p_team1, prints row counts and winner distribution only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iplpred.core.evaluation import brier_score, log_loss_binary, reliability_bins
from iplpred.paths import PROCESSED_DIR

DATA_PATH = PROCESSED_DIR / "match_training_dataset.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate winner predictions vs match_training_dataset.")
    parser.add_argument("--csv", type=str, default=str(DATA_PATH))
    parser.add_argument(
        "--prob-col",
        type=str,
        default="p_team1",
        help="Column with P(team1 wins); if missing, only summary stats are printed",
    )
    args = parser.parse_args()
    path = Path(args.csv)
    if not path.is_file():
        raise SystemExit(f"Missing {path}")

    df = pd.read_csv(path, low_memory=False)
    w = df["winner"].astype(str).str.strip().str.lower()
    y = (w == "team1").astype(float)
    mask = w.isin(["team1", "team2"])
    y = y[mask].values
    print(f"Rows with winner team1/team2: {mask.sum()} / {len(df)}")

    if args.prob_col not in df.columns:
        print(f"No column {args.prob_col!r} — add predictions to score.")
        print("Winner counts:\n", df.loc[mask, "winner"].value_counts())
        return

    p = pd.to_numeric(df.loc[mask, args.prob_col], errors="coerce").values
    valid = ~np.isnan(p)
    y, p = y[valid], p[valid]
    print(f"Brier: {brier_score(y, p):.4f}")
    print(f"Log loss: {log_loss_binary(y, p):.4f}")
    print(reliability_bins(y, p).to_string(index=False))


if __name__ == "__main__":
    main()
