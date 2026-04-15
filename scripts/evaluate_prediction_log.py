#!/usr/bin/env python3
"""
Evaluate logged pre-match probabilities vs post-match outcomes (Process 1 →2).

Reads data/processed/prediction_log.csv. For each row with actual_winner and
pred_p_team1_win, reports Brier score, log loss, crude calibration buckets, and
winner pick accuracy (team1 = first innings in the logged row).

Ties / NR: actual_winner matching 'tie' or 'no result' are excluded from binary metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from iplpred.paths import PROCESSED_DIR

LOG_PATH = PROCESSED_DIR / "prediction_log.csv"


def _norm(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def _is_no_result(actual: str) -> bool:
    t = _norm(actual)
    return t in ("tie", "no result", "nr", "abandoned", "match abandoned")


def main() -> None:
    ap = argparse.ArgumentParser(description="Brier / calibration on prediction_log.csv")
    ap.add_argument(
        "--log",
        type=str,
        default=str(LOG_PATH),
        help="Path to prediction_log.csv",
    )
    args = ap.parse_args()
    path = Path(args.log)
    if not path.is_file():
        raise SystemExit(f"Missing {path}")

    df = pd.read_csv(path, low_memory=False)
    df = df[df["actual_winner"].notna() & (df["actual_winner"].astype(str).str.strip() != "")]
    df = df[df["pred_p_team1_win"].notna()]

    t1 = df["team1_name"].astype(str).str.strip()
    t2 = df["team2_name"].astype(str).str.strip()
    aw = df["actual_winner"].astype(str).str.strip()
    mask_nr = aw.map(_is_no_result)
    df = df.loc[~mask_nr].copy()
    t1 = df["team1_name"].astype(str).str.strip()
    t2 = df["team2_name"].astype(str).str.strip()
    aw = df["actual_winner"].astype(str).str.strip()

    y = []
    for a, b, w in zip(t1, t2, aw, strict=True):
        wa = _norm(w)
        if wa == _norm(a):
            y.append(1.0)
        elif wa == _norm(b):
            y.append(0.0)
        else:
            y.append(float("nan"))
    df["y_team1_win"] = y
    df = df[df["y_team1_win"].notna()]
    y = df["y_team1_win"].astype(float).values
    p = pd.to_numeric(df["pred_p_team1_win"], errors="coerce").clip(1e-6, 1.0 - 1e-6).values

    n = len(df)
    if n == 0:
        print("No scored rows (need actual_winner + pred_p_team1_win, non-tie).")
        return

    brier = float(np.mean((p - y) ** 2))
    ll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    acc = float(np.mean((p >= 0.5).astype(int) == y.astype(int)))

    print("=== prediction_log evaluation (binary: team1 = first innings wins) ===")
    print(f"Rows used: {n}")
    print(f"Pick accuracy (p>=0.5 vs team1 win): {acc:.4f}")
    print(f"Brier score:   {brier:.4f}  (lower is better; 0.25 = constant 0.5)")
    print(f"Log loss:      {ll:.4f} (lower is better)")
    print()
    print("Reliability (predicted P(team1) bucket vs fraction team1 won):")
    edges = [0.0, 0.35, 0.45, 0.55, 0.65, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi)
        if not m.any():
            continue
        print(
            f"  [{lo:.2f}, {hi:.2f}): n={int(m.sum())}  "
            f"mean_pred={float(p[m].mean()):.3f}  actual_rate={float(y[m].mean()):.3f}"
        )

    hy_deltas: list[float] = []
    for _, row in df.iterrows():
        raw = row.get("pred_extra_json", "")
        if pd.isna(raw) or not str(raw).strip():
            continue
        try:
            ex = json.loads(str(raw))
        except json.JSONDecodeError:
            continue
        mc = ex.get("sim_win_probability_team1")
        hy = ex.get("ensemble_p_team1")
        if mc is not None and hy is not None:
            hy_deltas.append(float(hy) - float(mc))
    if hy_deltas:
        print()
        print(
            f"Mean |hybrid − MC| (where both in pred_extra): "
            f"{float(np.mean(np.abs(hy_deltas))):.3f}  (n={len(hy_deltas)})"
        )


if __name__ == "__main__":
    main()
