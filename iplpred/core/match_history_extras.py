"""
Chronological match-history features: H2H, toss mapping, season table, inference replay.

Training uses attach_* on sorted match_training_dataset rows; inference replays from CSV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from iplpred.core.franchise_normalize import canonical_franchise
from iplpred.paths import DATA_DIR, PROCESSED_DIR

MATCH_TRAINING_PATH = PROCESSED_DIR / "match_training_dataset.csv"

DEFAULT_PRIOR = 0.5
H2H_WINDOW = 5


def _pair_key(a: str, b: str) -> frozenset[str]:
    return frozenset({canonical_franchise(a), canonical_franchise(b)})


def attach_h2h_chronological(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prior to each row: mean outcome of last H2H_WINDOW meetings between the same two
    franchises (lex team1 = smaller name). Stores 1 if lex-smaller won, 0 if larger, 0.5 tie.
    """
    out = df.copy()
    if "match_date" not in out.columns or "team1_name" not in out.columns:
        out["h2h_team1_win_prior"] = DEFAULT_PRIOR
        return out
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce")
    sort_cols = ["match_date"]
    if "match_id" in out.columns:
        sort_cols.append("match_id")
    out = out.sort_values(sort_cols).reset_index(drop=True)

    hist: dict[frozenset[str], list[float]] = {}
    priors: list[float] = []
    for _, row in out.iterrows():
        a = canonical_franchise(str(row["team1_name"]))
        b = canonical_franchise(str(row["team2_name"]))
        k = _pair_key(str(row["team1_name"]), str(row["team2_name"]))
        h = hist.get(k, [])
        priors.append(float(np.mean(h[-H2H_WINDOW:])) if h else DEFAULT_PRIOR)
        w = str(row.get("winner", "")).strip()
        if w == "team1":
            hist.setdefault(k, []).append(1.0)
        elif w == "team2":
            hist.setdefault(k, []).append(0.0)
        elif w == "tie":
            hist.setdefault(k, []).append(0.5)
        if len(hist.get(k, [])) > 40:
            hist[k] = hist[k][-20:]

    out["h2h_team1_win_prior"] = priors
    return out


def attach_season_table_chronological(df: pd.DataFrame) -> pd.DataFrame:
    """
    Win-rate and run-margin prior within the same season (or fallback year bucket).
    """
    out = df.copy()
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce")
    if "season" not in out.columns:
        out["season"] = out["match_date"].dt.year.astype(str)
    out["season"] = out["season"].fillna("").astype(str).str.strip()
    out.loc[out["season"].eq(""), "season"] = (
        out.loc[out["season"].eq(""), "match_date"].dt.year.astype(str)
    )

    sort_cols = ["match_date"]
    if "match_id" in out.columns:
        sort_cols.append("match_id")
    out = out.sort_values(sort_cols).reset_index(drop=True)

    # (season, team) -> wins, games, runs_for, runs_against
    winc: dict[tuple[str, str], int] = {}
    games: dict[tuple[str, str], int] = {}
    rf: dict[tuple[str, str], float] = {}
    ra: dict[tuple[str, str], float] = {}

    sw1: list[float] = []
    sw2: list[float] = []
    sm1: list[float] = []
    sm2: list[float] = []

    tr1 = pd.to_numeric(out.get("team1_total_runs"), errors="coerce")
    tr2 = pd.to_numeric(out.get("team2_total_runs"), errors="coerce")

    for i, row in out.iterrows():
        s = str(row["season"])
        t1 = canonical_franchise(str(row["team1_name"]))
        t2 = canonical_franchise(str(row["team2_name"]))
        k1 = (s, t1)
        k2 = (s, t2)
        g1 = max(games.get(k1, 0), 0)
        g2 = max(games.get(k2, 0), 0)
        sw1.append(winc.get(k1, 0) / g1 if g1 > 0 else DEFAULT_PRIOR)
        sw2.append(winc.get(k2, 0) / g2 if g2 > 0 else DEFAULT_PRIOR)
        margin1 = (rf.get(k1, 0.0) - ra.get(k1, 0.0)) / g1 if g1 > 0 else 0.0
        margin2 = (rf.get(k2, 0.0) - ra.get(k2, 0.0)) / g2 if g2 > 0 else 0.0
        sm1.append(margin1)
        sm2.append(margin2)

        w = str(row.get("winner", "")).strip()
        r1 = float(tr1.iloc[i]) if pd.notna(tr1.iloc[i]) else float("nan")
        r2 = float(tr2.iloc[i]) if pd.notna(tr2.iloc[i]) else float("nan")

        games[k1] = games.get(k1, 0) + 1
        games[k2] = games.get(k2, 0) + 1
        if not np.isnan(r1) and not np.isnan(r2):
            rf[k1] = rf.get(k1, 0.0) + r1
            ra[k1] = ra.get(k1, 0.0) + r2
            rf[k2] = rf.get(k2, 0.0) + r2
            ra[k2] = ra.get(k2, 0.0) + r1
        if w == "team1":
            winc[k1] = winc.get(k1, 0) + 1
        elif w == "team2":
            winc[k2] = winc.get(k2, 0) + 1

    out["team1_season_win_pct_prior"] = sw1
    out["team2_season_win_pct_prior"] = sw2
    out["team1_season_run_margin_prior"] = sm1
    out["team2_season_run_margin_prior"] = sm2
    return out


def h2h_prior_from_history(
    team1_name: str,
    team2_name: str,
    match_date: pd.Timestamp | str | None,
    *,
    history: pd.DataFrame | None = None,
) -> float:
    if history is None:
        if not MATCH_TRAINING_PATH.is_file():
            return DEFAULT_PRIOR
        history = pd.read_csv(MATCH_TRAINING_PATH, low_memory=False)
    if history.empty or "winner" not in history.columns:
        return DEFAULT_PRIOR
    hdf = history.copy()
    hdf["match_date"] = pd.to_datetime(hdf["match_date"], errors="coerce")
    sort_cols = ["match_date"]
    if "match_id" in hdf.columns:
        sort_cols.append("match_id")
    hdf = hdf.sort_values(sort_cols)
    as_of = pd.to_datetime(match_date, errors="coerce") if match_date is not None else None
    if as_of is not None and pd.notna(as_of):
        hdf = hdf[hdf["match_date"] < as_of]

    k = _pair_key(team1_name, team2_name)
    hist: list[float] = []
    for _, row in hdf.iterrows():
        rk = _pair_key(str(row["team1_name"]), str(row["team2_name"]))
        if rk != k:
            continue
        w = str(row["winner"]).strip()
        if w == "team1":
            hist.append(1.0)
        elif w == "team2":
            hist.append(0.0)
        elif w == "tie":
            hist.append(0.5)
    return float(np.mean(hist[-H2H_WINDOW:])) if hist else DEFAULT_PRIOR


def season_priors_from_history(
    team1_name: str,
    team2_name: str,
    season: str,
    match_date: pd.Timestamp | str | None,
    *,
    history: pd.DataFrame | None = None,
) -> tuple[float, float, float, float]:
    """Returns team1_win_pct, team2_win_pct, team1_margin, team2_margin prior to date."""
    if history is None:
        if not MATCH_TRAINING_PATH.is_file():
            return DEFAULT_PRIOR, DEFAULT_PRIOR, 0.0, 0.0
        history = pd.read_csv(MATCH_TRAINING_PATH, low_memory=False)
    if history.empty:
        return DEFAULT_PRIOR, DEFAULT_PRIOR, 0.0, 0.0

    hdf = history.copy()
    hdf["match_date"] = pd.to_datetime(hdf["match_date"], errors="coerce")
    if "season" not in hdf.columns:
        hdf["season"] = hdf["match_date"].dt.year.astype(str)
    hdf["season"] = hdf["season"].fillna("").astype(str).str.strip()
    hdf.loc[hdf["season"].eq(""), "season"] = hdf["match_date"].dt.year.astype(str)

    as_of = pd.to_datetime(match_date, errors="coerce") if match_date is not None else None
    if as_of is not None and pd.notna(as_of):
        hdf = hdf[hdf["match_date"] < as_of]

    s = str(season).strip()
    t1 = canonical_franchise(str(team1_name))
    t2 = canonical_franchise(str(team2_name))

    w1 = w2 = g1 = g2 = 0
    rf1 = rf2 = ra1 = ra2 = 0.0
    for _, row in hdf.iterrows():
        if str(row["season"]).strip() != s:
            continue
        a = canonical_franchise(str(row["team1_name"]))
        b = canonical_franchise(str(row["team2_name"]))
        w = str(row["winner"]).strip()
        r1 = pd.to_numeric(row.get("team1_total_runs"), errors="coerce")
        r2 = pd.to_numeric(row.get("team2_total_runs"), errors="coerce")
        if a == t1 and b == t2:
            g1 += 1
            g2 += 1
            if w == "team1":
                w1 += 1
            elif w == "team2":
                w2 += 1
            if pd.notna(r1) and pd.notna(r2):
                rf1 += float(r1)
                ra1 += float(r2)
                rf2 += float(r2)
                ra2 += float(r1)
        elif a == t2 and b == t1:
            g1 += 1
            g2 += 1
            if w == "team1":
                w2 += 1
            elif w == "team2":
                w1 += 1
            if pd.notna(r1) and pd.notna(r2):
                rf2 += float(r1)
                ra2 += float(r2)
                rf1 += float(r2)
                ra1 += float(r1)

    pct1 = w1 / g1 if g1 > 0 else DEFAULT_PRIOR
    pct2 = w2 / g2 if g2 > 0 else DEFAULT_PRIOR
    m1 = (rf1 - ra1) / g1 if g1 > 0 else 0.0
    m2 = (rf2 - ra2) / g2 if g2 > 0 else 0.0
    return float(pct1), float(pct2), float(m1), float(m2)


def load_optional_weather_csv() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "match_weather_optional.csv"
    if not p.is_file():
        p = DATA_DIR / "ipl" / "match_weather_optional.csv"
    if not p.is_file():
        return None
    w = pd.read_csv(p, low_memory=False)
    if "match_id" not in w.columns:
        return None
    w["match_id"] = pd.to_numeric(w["match_id"], errors="coerce")
    return w


def load_optional_injuries_csv() -> pd.DataFrame | None:
    p = PROCESSED_DIR / "match_injuries_optional.csv"
    if not p.is_file():
        p = DATA_DIR / "ipl" / "match_injuries_optional.csv"
    if not p.is_file():
        return None
    inj = pd.read_csv(p, low_memory=False)
    if "match_id" not in inj.columns:
        return None
    inj["match_id"] = pd.to_numeric(inj["match_id"], errors="coerce")
    return inj
