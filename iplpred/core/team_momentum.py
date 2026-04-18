"""
Franchise win-rate prior over the last N completed matches (chronological).

Used by the match-winner and team-total heads. Training rows get momentum from
build_training_dataset; at inference we replay history from match_training_dataset.csv.

Venue momentum uses ``_canonical_venue_key`` so e.g. "Narendra Modi Stadium,
Ahmedabad" and "Ahmedabad" share one bucket (``ipl_ahmedabad``). For Gujarat Titans
at that bucket, thin or contradictory rows are shrunk toward ``GT_AHMEDABAD_VENUE_PRIOR``
(see ``_franchise_venue_momentum_prior``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from iplpred.core.franchise_normalize import canonical_franchise
from iplpred.paths import PROCESSED_DIR

MATCH_TRAINING_PATH = PROCESSED_DIR / "match_training_dataset.csv"
DEFAULT_PRIOR = 0.5
WINDOW = 5

# IPL Ahmedabad: training rows sometimes use "Ahmedabad" only; fixtures use
# "Narendra Modi Stadium, Ahmedabad". Map to one key so venue history matches.
IPL_AHMEDABAD_KEY = "ipl_ahmedabad"
# When sample size at this ground is tiny, shrink GT's rate toward a strong home
# prior (sparse proxy rows should not erase real home dominance).
# IPL home record at Narendra Modi Stadium — shrink sparse/noisy rows here toward this.
GT_AHMEDABAD_VENUE_PRIOR = 0.82
GT_AHMEDABAD_PRIOR_WEIGHT = 8.0


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


def _norm_venue(venue: str) -> str:
    v = " ".join(str(venue or "").strip().lower().split())
    return v[:120] if v else ""


def _canonical_venue_key(venue: str) -> str:
    """Normalize venue labels so the same ground shares one momentum bucket."""
    v = _norm_venue(venue)
    if not v:
        return ""
    if "ahmedabad" in v:
        return IPL_AHMEDABAD_KEY
    return v


def _franchise_venue_momentum_prior(
    team_canon: str, venue_key: str, raw_rate: float, n_games: int
) -> float:
    """Blend empirical venue win-rate with a franchise–ground prior when data are thin."""
    if (
        venue_key == IPL_AHMEDABAD_KEY
        and team_canon == canonical_franchise("Gujarat Titans")
    ):
        if n_games == 0:
            return float(GT_AHMEDABAD_VENUE_PRIOR)
        return float(
            (GT_AHMEDABAD_PRIOR_WEIGHT * GT_AHMEDABAD_VENUE_PRIOR + raw_rate * n_games)
            / (GT_AHMEDABAD_PRIOR_WEIGHT + n_games)
        )
    if n_games == 0:
        return DEFAULT_PRIOR
    return float(raw_rate)


def _venue_rate_adjusted(
    vhist: dict[tuple[str, str], list[float]],
    team: str,
    venue_key: str,
) -> float:
    ct = canonical_franchise(team)
    h = vhist.get((ct, venue_key), [])
    if not h:
        return _franchise_venue_momentum_prior(ct, venue_key, DEFAULT_PRIOR, 0)
    n_games = len(h)
    raw = float(np.mean(h[-WINDOW:]))
    return _franchise_venue_momentum_prior(ct, venue_key, raw, n_games)


def _append_venue_result(
    hist: dict[tuple[str, str], list[float]],
    t1: str,
    t2: str,
    venue_key: str,
    winner: str,
) -> None:
    if not venue_key:
        return
    if winner == "team1":
        hist.setdefault((t1, venue_key), []).append(1.0)
        hist.setdefault((t2, venue_key), []).append(0.0)
    elif winner == "team2":
        hist.setdefault((t1, venue_key), []).append(0.0)
        hist.setdefault((t2, venue_key), []).append(1.0)
    else:
        hist.setdefault((t1, venue_key), []).append(0.5)
        hist.setdefault((t2, venue_key), []).append(0.5)
    for key in ((t1, venue_key), (t2, venue_key)):
        if len(hist.get(key, [])) > 40:
            hist[key] = hist[key][-20:]


def attach_venue_momentum_chronological(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling win rate at this **venue** (last WINDOW games at that ground) prior to the row.
    """
    out = df.copy()
    if "match_date" not in out.columns or "venue" not in out.columns:
        out["team1_venue_momentum"] = DEFAULT_PRIOR
        out["team2_venue_momentum"] = DEFAULT_PRIOR
        return out
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce")
    sort_cols = ["match_date"]
    if "match_id" in out.columns:
        sort_cols.append("match_id")
    out = out.sort_values(sort_cols).reset_index(drop=True)

    vhist: dict[tuple[str, str], list[float]] = {}
    t1v: list[float] = []
    t2v: list[float] = []
    for _, row in out.iterrows():
        a = canonical_franchise(str(row["team1_name"]))
        b = canonical_franchise(str(row["team2_name"]))
        raw_v = str(row.get("venue", ""))
        vk = _canonical_venue_key(raw_v) or _norm_venue(raw_v)
        if not vk:
            vk = "__unknown__"
        t1v.append(_venue_rate_adjusted(vhist, a, vk))
        t2v.append(_venue_rate_adjusted(vhist, b, vk))
        w = str(row.get("winner", "")).strip()
        if w in ("team1", "team2", "tie"):
            _append_venue_result(vhist, a, b, vk, w)
    out["team1_venue_momentum"] = t1v
    out["team2_venue_momentum"] = t2v
    return out


def venue_momentum_row_from_history(
    team1_name: str,
    team2_name: str,
    venue: str | None,
    match_date: pd.Timestamp | str | None,
    *,
    history: pd.DataFrame | None = None,
) -> tuple[float, float]:
    """Venue-specific win-rate prior before this fixture (same WINDOW as global momentum)."""
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

    vk = _canonical_venue_key(str(venue or "")) or _norm_venue(str(venue or ""))
    if not vk:
        return DEFAULT_PRIOR, DEFAULT_PRIOR

    vhist: dict[tuple[str, str], list[float]] = {}
    for _, row in hdf.iterrows():
        t1 = canonical_franchise(str(row["team1_name"]))
        t2 = canonical_franchise(str(row["team2_name"]))
        raw_venue = str(row.get("venue", ""))
        row_vk = _canonical_venue_key(raw_venue) or _norm_venue(raw_venue)
        if not row_vk:
            row_vk = "__unknown__"
        w = str(row["winner"]).strip()
        _append_venue_result(vhist, t1, t2, row_vk, w)

    t1n = canonical_franchise(str(team1_name))
    t2n = canonical_franchise(str(team2_name))
    m1 = _venue_rate_adjusted(vhist, t1n, vk)
    m2 = _venue_rate_adjusted(vhist, t2n, vk)
    return m1, m2
