"""
Franchise win-rate prior over the last N completed matches (chronological).

Used by the match-winner and team-total heads. Training rows get momentum from
build_training_dataset; at inference we replay history from match_training_dataset.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from iplpred.core.franchise_normalize import canonical_franchise
from iplpred.paths import PROCESSED_DIR

MATCH_TRAINING_PATH = PROCESSED_DIR / "match_training_dataset.csv"
DEFAULT_PRIOR = 0.5
WINDOW = 5


def _rate_from_hist(hist: dict[str, list[float]], team: str) -> float:
    h = hist.get(team, [])
    if not h:
        return DEFAULT_PRIOR
    return float(np.mean(h[-WINDOW:]))


def _append_result(
    hist: dict[str, list[float]],
    t1: str,
    t2: str,
    winner: str,
) -> None:
    if winner == "team1":
        hist.setdefault(t1, []).append(1.0)
        hist.setdefault(t2, []).append(0.0)
    elif winner == "team2":
        hist.setdefault(t1, []).append(0.0)
        hist.setdefault(t2, []).append(1.0)
    else:
        hist.setdefault(t1, []).append(0.5)
        hist.setdefault(t2, []).append(0.5)
    for t in (t1, t2):
        if len(hist.get(t, [])) > 40:
            hist[t] = hist[t][-20:]


def attach_momentum_columns_chronological(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each match row, set team1_momentum / team2_momentum from **prior** results only.
    Expects team1_name, team2_name, winner, match_date; optional match_id.
    """
    out = df.copy()
    if "match_date" not in out.columns:
        out["team1_momentum"] = DEFAULT_PRIOR
        out["team2_momentum"] = DEFAULT_PRIOR
        return out
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce")
    sort_cols = ["match_date"]
    if "match_id" in out.columns:
        sort_cols.append("match_id")
    out = out.sort_values(sort_cols).reset_index(drop=True)

    team_hist: dict[str, list[float]] = {}
    t1m: list[float] = []
    t2m: list[float] = []
    for _, row in out.iterrows():
        a = canonical_franchise(str(row["team1_name"]))
        b = canonical_franchise(str(row["team2_name"]))
        t1m.append(_rate_from_hist(team_hist, a))
        t2m.append(_rate_from_hist(team_hist, b))
        w = str(row.get("winner", "")).strip()
        if w in ("team1", "team2", "tie"):
            _append_result(team_hist, a, b, w)
    out["team1_momentum"] = t1m
    out["team2_momentum"] = t2m
    return out


def momentum_row_from_history(
    team1_name: str,
    team2_name: str,
    match_date: pd.Timestamp | str | None,
    *,
    history: pd.DataFrame | None = None,
) -> tuple[float, float]:
    """
    Win-rate prior (last WINDOW games) for each franchise **before** this fixture.

    ``history`` must have columns: match_date, team1_name, team2_name, winner
    (winner in team1|team2|tie). If None, loads match_training_dataset.csv.
    """
    if history is None:
        if not MATCH_TRAINING_PATH.is_file():
            return DEFAULT_PRIOR, DEFAULT_PRIOR
        history = pd.read_csv(MATCH_TRAINING_PATH, low_memory=False)
    if history.empty or "winner" not in history.columns:
        return DEFAULT_PRIOR, DEFAULT_PRIOR

    hdf = history.copy()
    hdf["match_date"] = pd.to_datetime(hdf["match_date"], errors="coerce")
    sort_cols = ["match_date"]
    if "match_id" in hdf.columns:
        sort_cols.append("match_id")
    hdf = hdf.sort_values(sort_cols)
    hdf = hdf[hdf["winner"].isin(["team1", "team2", "tie"])]

    as_of = pd.to_datetime(match_date, errors="coerce") if match_date is not None else None
    if as_of is not None and pd.notna(as_of):
        hdf = hdf[hdf["match_date"] < as_of]

    team_hist: dict[str, list[float]] = {}
    for _, row in hdf.iterrows():
        t1 = canonical_franchise(str(row["team1_name"]))
        t2 = canonical_franchise(str(row["team2_name"]))
        w = str(row["winner"]).strip()
        _append_result(team_hist, t1, t2, w)

    m1 = _rate_from_hist(team_hist, canonical_franchise(str(team1_name)))
    m2 = _rate_from_hist(team_hist, canonical_franchise(str(team2_name)))
    return m1, m2
