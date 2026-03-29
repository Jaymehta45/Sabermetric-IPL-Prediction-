"""
Cross-league recent performance from player_match_stats (last N appearances).

Used for data-confidence weights and optional shrinkage toward team priors.
"""

from __future__ import annotations

from functools import lru_cache
import numpy as np
import pandas as pd

from iplpred.paths import REPO_ROOT

PLAYER_MATCH_STATS_PATH = REPO_ROOT / "data" / "processed" / "player_match_stats.csv"

# Recency weights for last K games (most recent first): sum = 1
DEFAULT_RECENCY_WEIGHTS_10 = np.array(
    [0.22, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.02, 0.01]
)
DEFAULT_RECENCY_WEIGHTS_5 = np.array([0.35, 0.25, 0.18, 0.12, 0.10])


def load_match_stats() -> pd.DataFrame:
    if not PLAYER_MATCH_STATS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(PLAYER_MATCH_STATS_PATH, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values(["player_id", "date", "match_id"])


def build_recent_form_table(
    *,
    last_n: int = 10,
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    One row per player_id: rolling counts and weighted recent averages (all leagues).

    Columns:
      - n_matches_last_5, n_matches_last_10 (distinct match_id in window)
      - weighted_avg_runs_last_5, weighted_avg_wk_last_5
      - weighted_avg_runs_last_10, weighted_avg_wk_last_10
      - data_confidence: min(1, n_distinct_last_10 / 10) — for shrinkage
    """
    df = load_match_stats()
    if df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for pid, g in df.groupby("player_id", sort=False):
        g = g.sort_values(["date", "match_id"])
        # one row per match per player (already match-level)
        last10 = g.tail(min(last_n, len(g)))
        last5 = g.tail(min(5, len(g)))
        n10 = last10["match_id"].nunique() if len(last10) else 0
        n5 = last5["match_id"].nunique() if len(last5) else 0

        runs = pd.to_numeric(last10["runs"], errors="coerce").fillna(0.0).values
        wk = pd.to_numeric(last10["wickets"], errors="coerce").fillna(0.0).values
        w_r10 = DEFAULT_RECENCY_WEIGHTS_10[: len(runs)]
        if len(w_r10) != len(runs):
            w_r10 = np.ones(len(runs)) / max(len(runs), 1)
        else:
            w_r10 = w_r10 / w_r10.sum()

        runs5 = pd.to_numeric(last5["runs"], errors="coerce").fillna(0.0).values
        wk5 = pd.to_numeric(last5["wickets"], errors="coerce").fillna(0.0).values
        w5 = DEFAULT_RECENCY_WEIGHTS_5[: len(runs5)]
        if len(w5) != len(runs5):
            w5 = np.ones(len(runs5)) / max(len(runs5), 1)
        else:
            w5 = w5 / w5.sum()

        conf = min(1.0, float(n10) / 10.0) if last_n >= 10 else min(1.0, float(len(last10)) / 5.0)

        rows.append(
            {
                "player_id": pid,
                "n_matches_last_5": int(n5),
                "n_matches_last_10": int(n10),
                "weighted_avg_runs_last_5": float(np.dot(runs5, w5)),
                "weighted_avg_wk_last_5": float(np.dot(wk5, w5)),
                "weighted_avg_runs_last_10": float(np.dot(runs, w_r10)),
                "weighted_avg_wk_last_10": float(np.dot(wk, w_r10)),
                "data_confidence": float(conf),
            }
        )

    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def get_recent_form_cached() -> pd.DataFrame:
    return build_recent_form_table()


def player_confidence_map(player_ids: list[str]) -> dict[str, float]:
    """Map stats player_id -> data_confidence in [0,1]."""
    t = get_recent_form_cached()
    if t.empty:
        return {str(p).strip(): 0.5 for p in player_ids}
    m = dict(zip(t["player_id"].astype(str).str.strip(), t["data_confidence"]))
    out: dict[str, float] = {}
    for p in player_ids:
        k = str(p).strip()
        out[k] = float(m.get(k, 0.35))
    return out
