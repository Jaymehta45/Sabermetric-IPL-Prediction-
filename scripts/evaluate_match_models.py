#!/usr/bin/env python3
"""
Temporal holdout metrics for match-winner and team-total models (same splits as training).

Usage:
  python3 scripts/evaluate_match_models.py
  python3 scripts/evaluate_match_models.py --test-frac 0.25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)

from iplpred.match.match_winner_model import (
    WINNER_MODEL_PATH,
    build_winner_feature_matrix,
)
from iplpred.match.team_total_model import MODEL_PATH as TEAM_TOTAL_MODEL_PATH
from iplpred.training.train_match_winner_model import (
    load_match_training_with_dates,
    time_based_split,
)
from iplpred.training.train_team_total_model import load_with_dates, time_based_split as tt_split


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate match-level models on time holdout.")
    ap.add_argument("--test-frac", type=float, default=0.2, help="Fraction most recent for test")
    args = ap.parse_args()

    df = load_match_training_with_dates()
    _train, test = time_based_split(df, test_frac=args.test_frac)
    clf_bundle = joblib.load(WINNER_MODEL_PATH)
    clf = clf_bundle["model"]
    X_test = build_winner_feature_matrix(test)
    y_test = (test["winner"] == "team1").astype(int).values
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("=== Match winner (time holdout) ===")
    print(f"Test rows: {len(test)}")
    print(f"  accuracy:  {accuracy_score(y_test, pred):.4f}")
    try:
        print(f"  roc_auc:   {roc_auc_score(y_test, proba):.4f}")
    except ValueError:
        print("  roc_auc:   n/a")
    try:
        print(f"  log_loss:  {log_loss(y_test, proba):.4f}")
    except ValueError:
        print("  log_loss:  n/a")
    print(f"  brier:     {brier_score_loss(y_test, proba):.4f}")

    if "winner_source" in test.columns:
        for src in sorted(test["winner_source"].dropna().unique()):
            sub = test[test["winner_source"] == src]
            if len(sub) < 5:
                continue
            ys = (sub["winner"] == "team1").astype(int).values
            pr = clf.predict_proba(build_winner_feature_matrix(sub))[:, 1]
            pd_ = (pr >= 0.5).astype(int)
            print(
                f"  [{src}] n={len(sub)}  acc={accuracy_score(ys, pd_):.4f}  "
                f"brier={brier_score_loss(ys, pr):.4f}"
            )

    tdf = load_with_dates()
    _tr, te = tt_split(tdf, test_frac=args.test_frac)
    tt_bundle = joblib.load(TEAM_TOTAL_MODEL_PATH)
    r1, r2 = tt_bundle["model_team1"], tt_bundle["model_team2"]
    X_te = build_winner_feature_matrix(te)
    cap = float(tt_bundle.get("target_clip", 320.0))
    p1 = r1.predict(X_te)
    p2 = r2.predict(X_te)
    if tt_bundle.get("target_transform") == "log1p":
        p1 = np.clip(np.expm1(p1), 0.0, cap)
        p2 = np.clip(np.expm1(p2), 0.0, cap)
    y1 = te["team1_total_runs"].astype(float).values
    y2 = te["team2_total_runs"].astype(float).values

    print()
    print("=== Team innings totals (time holdout, same split) ===")
    print(f"Test rows: {len(te)}")
    print(f"  MAE team1: {mean_absolute_error(y1, p1):.2f}")
    print(f"  MAE team2: {mean_absolute_error(y2, p2):.2f}")


if __name__ == "__main__":
    main()
