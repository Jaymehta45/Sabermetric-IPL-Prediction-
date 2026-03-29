"""
Calibration-oriented metrics for match-level predictions (winner probability).

Use with held-out official results — see data/processed/match_training_dataset.csv.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from iplpred.paths import REPO_ROOT


def log_loss_binary(y_true: np.ndarray, p_pred: np.ndarray, eps: float = 1e-15) -> float:
    """y_true in {0,1}, p_pred = P(y=1)."""
    p = np.clip(p_pred, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    return float(np.mean((p_pred - y_true) ** 2))


def reliability_bins(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Calibration table: mean predicted prob vs empirical frequency per bin."""
    p_pred = np.asarray(p_pred)
    y_true = np.asarray(y_true)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p_pred, bins[:-1]) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            rows.append({"bin": b, "n": 0, "mean_p": np.nan, "empirical_rate": np.nan})
        else:
            rows.append(
                {
                    "bin": b,
                    "n": int(m.sum()),
                    "mean_p": float(p_pred[m].mean()),
                    "empirical_rate": float(y_true[m].mean()),
                }
            )
    return pd.DataFrame(rows)


def walk_forward_splits(
    dates: pd.Series,
    n_folds: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Return list of (train_idx, test_idx) by sorting on date and chunking tests.
    """
    order = np.argsort(pd.to_datetime(dates, errors="coerce").values)
    n = len(order)
    folds = []
    for k in range(n_folds):
        start = int((k / n_folds) * n)
        end = int(((k + 1) / n_folds) * n)
        test_idx = order[start:end]
        train_idx = np.concatenate([order[:start], order[end:]])
        folds.append((train_idx, test_idx))
    return folds


def load_official_match_results_path() -> Path:
    return REPO_ROOT / "data" / "processed" / "match_training_dataset.csv"


if __name__ == "__main__":
    import pandas as pd

    p = load_official_match_results_path()
    if p.exists():
        df = pd.read_csv(p, nrows=5)
        print("Sample columns:", list(df.columns))
        print("Use scripts/evaluate_official_results.py after adding a p_team1 column with your predictions.")
