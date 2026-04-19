"""
Recent completed-match scoring and venue affinity for innings total calibration.

Blends model targets toward (a) each team's last few actual innings totals in the
same role (bat first / chase) when toss rows are reliable, (b) same-venue samples,
and (c) roster players who historically score more at this venue than elsewhere.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

import numpy as np
import pandas as pd

from iplpred.core.franchise_normalize import canonical_franchise
from iplpred.core.team_momentum import _canonical_venue_key
from iplpred.paths import PROCESSED_DIR

MATCH_TRAINING_PATH = PROCESSED_DIR / "match_training_dataset.csv"
PLAYER_MATCH_STATS_PATH = PROCESSED_DIR / "player_match_stats.csv"


def _norm_venue_str(venue: str) -> str:
    return " ".join(str(venue or "").strip().lower().replace(".", " ").split())


def venue_same_ground(a: str | None, b: str | None) -> bool:
    """True when two venue labels refer to the same IPL ground (fuzzy)."""
    if not a or not b:
        return False
    ca, cb = _canonical_venue_key(a), _canonical_venue_key(b)
    if ca and cb and ca == cb:
        return True
    va, vb = _norm_venue_str(a), _norm_venue_str(b)
    if not va or not vb:
        return False
    if va == vb:
        return True
    # Chinnaswamy vs short "Bengaluru" labels
    def _is_chinnaswamy(v: str) -> bool:
        return "chinnaswamy" in v or v in ("bengaluru", "bangalore")

    if _is_chinnaswamy(va) and _is_chinnaswamy(vb):
        return True
    if va in vb or vb in va:
        return min(len(va), len(vb)) >= 6
    return False


def _team_runs_in_row(row: pd.Series, team_canon: str) -> float | None:
    t1 = canonical_franchise(str(row["team1_name"]).strip())
    t2 = canonical_franchise(str(row["team2_name"]).strip())
    if team_canon == t1:
        return float(pd.to_numeric(row["team1_total_runs"], errors="coerce"))
    if team_canon == t2:
        return float(pd.to_numeric(row["team2_total_runs"], errors="coerce"))
    return None


def _team_batted_first(
    row: pd.Series, team_canon: str
) -> bool | None:
    sig = float(pd.to_numeric(row.get("team1_bats_first_signal"), errors="coerce"))
    if pd.isna(sig) or abs(sig - 0.5) < 1e-9:
        return None
    t1 = canonical_franchise(str(row["team1_name"]).strip())
    t2 = canonical_franchise(str(row["team2_name"]).strip())
    if team_canon == t1:
        return bool(sig >= 0.5)
    if team_canon == t2:
        return bool(sig < 0.5)
    return None


@lru_cache(maxsize=1)
def _load_match_training() -> pd.DataFrame:
    if not MATCH_TRAINING_PATH.is_file():
        return pd.DataFrame()
    df = pd.read_csv(MATCH_TRAINING_PATH, low_memory=False)
    if df.empty:
        return df
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    sort_cols = ["match_date"]
    if "match_id" in df.columns:
        sort_cols.append("match_id")
    return df.sort_values(sort_cols)


def recent_team_innings_scores(
    team_name: str,
    match_date: pd.Timestamp | str | None,
    *,
    role: Literal["first", "second"],
    venue: str | None = None,
    last_n: int = 3,
    history: pd.DataFrame | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """
    Mean of the franchise's last ``last_n`` **completed** innings scores in ``role``,
    optionally preferring the same venue. Falls back to recent scores when toss/role
    is ambiguous (``team1_bats_first_signal`` == 0.5).
    """
    h = history if history is not None else _load_match_training()
    if h is None or h.empty:
        return None, {"n": 0, "note": "no_history"}
    team_canon = canonical_franchise(str(team_name).strip())
    if not team_canon:
        return None, {"n": 0, "note": "no_team"}

    as_of = pd.to_datetime(match_date, errors="coerce") if match_date is not None else None
    hdf = h.copy()
    if as_of is not None and pd.notna(as_of):
        hdf = hdf[hdf["match_date"] < as_of]

    def _involved(row: pd.Series) -> bool:
        t1 = canonical_franchise(str(row["team1_name"]).strip())
        t2 = canonical_franchise(str(row["team2_name"]).strip())
        return team_canon in (t1, t2)

    sub = hdf[hdf.apply(_involved, axis=1)].sort_values("match_date", ascending=False)

    role_hits: list[float] = []
    venue_role_hits: list[float] = []
    venue_any_hits: list[float] = []
    form_hits: list[float] = []

    for _, row in sub.iterrows():
        r = _team_runs_in_row(row, team_canon)
        if r is None or not np.isfinite(r) or r <= 0:
            continue
        form_hits.append(float(r))
        at_venue = bool(venue) and venue_same_ground(str(row.get("venue", "")), venue)
        if at_venue:
            venue_any_hits.append(float(r))
        bf = _team_batted_first(row, team_canon)
        if bf is None:
            continue
        is_first = bf is True
        want_first = role == "first"
        if is_first == want_first:
            role_hits.append(float(r))
            if at_venue:
                venue_role_hits.append(float(r))

    def _mean_first_n(xs: list[float], n: int) -> float | None:
        if not xs:
            return None
        head = xs[: min(len(xs), n)]
        return float(np.mean(head))

    m_role = _mean_first_n(role_hits, last_n)
    m_venue_role = _mean_first_n(venue_role_hits, last_n)
    m_venue_any = _mean_first_n(venue_any_hits, last_n)
    m_form = _mean_first_n(form_hits, last_n)

    m_venue = m_venue_role if m_venue_role is not None else m_venue_any

    meta: dict[str, Any] = {
        "n_role": len(role_hits),
        "n_venue_role": len(venue_role_hits),
        "n_venue_any": len(venue_any_hits),
        "n_form": len(form_hits),
        "mean_role": m_role,
        "mean_venue": m_venue,
        "mean_form": m_form,
    }

    if m_role is not None and m_venue is not None:
        prior = 0.55 * m_role + 0.45 * m_venue
        meta["blend"] = "role+venue"
        return prior, meta
    if m_role is not None:
        meta["blend"] = "role"
        return m_role, meta
    if m_venue is not None:
        meta["blend"] = "venue"
        return m_venue, meta
    if m_form is not None:
        meta["blend"] = "form_any_innings"
        return m_form, meta
    return None, meta


def blend_innings_targets_with_recent_scores(
    team1_name: str | None,
    team2_name: str | None,
    venue: str | None,
    match_date: pd.Timestamp | str | None,
    tgt1: float,
    tgt2: float,
    *,
    last_n: int = 3,
    weight: float = 0.38,
) -> tuple[float, float, dict[str, Any]]:
    """
    Pull first-innings / chase priors from recent real scores and blend toward ``tgt*``.

    ``weight`` is how much of the **delta** (prior - tgt) is applied (clamped).
    """
    w = float(np.clip(weight, 0.0, 0.65))
    meta: dict[str, Any] = {"weight": w, "last_n": last_n}
    if not team1_name or not team2_name:
        return tgt1, tgt2, meta

    p1, m1 = recent_team_innings_scores(
        team1_name, match_date, role="first", venue=venue, last_n=last_n
    )
    p2, m2 = recent_team_innings_scores(
        team2_name, match_date, role="second", venue=venue, last_n=last_n
    )
    meta["team1_recent"] = m1
    meta["team2_recent"] = m2

    t1, t2 = float(tgt1), float(tgt2)
    if p1 is not None and np.isfinite(p1) and t1 > 1.0:
        delta = w * (float(p1) - t1)
        delta = float(np.clip(delta, -28.0, 28.0))
        t1 = float(np.clip(t1 + delta, 72.0, 260.0))
    if p2 is not None and np.isfinite(p2) and t2 > 1.0:
        delta = w * (float(p2) - t2)
        delta = float(np.clip(delta, -28.0, 28.0))
        t2 = float(np.clip(t2 + delta, 72.0, 260.0))
    meta["prior_first_innings"] = p1
    meta["prior_chase"] = p2
    return t1, t2, meta


@lru_cache(maxsize=1)
def _load_pms() -> pd.DataFrame:
    if not PLAYER_MATCH_STATS_PATH.is_file():
        return pd.DataFrame()
    df = pd.read_csv(PLAYER_MATCH_STATS_PATH, low_memory=False)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def batting_xi_venue_run_lift(
    team_df: pd.DataFrame,
    venue: str | None,
    match_date: pd.Timestamp | str | None,
    *,
    top_k: int = 7,
    max_lift: float = 0.085,
) -> float:
    """
    Multiplier near 1.0 from **batters** who outperform at this venue vs their
    recent overall (last 8 knocks), weighted by predicted runs.
    """
    if team_df is None or len(team_df) == 0 or not venue:
        return 1.0
    as_of = pd.to_datetime(match_date, errors="coerce") if match_date is not None else None
    pms = _load_pms()
    if pms.empty or "player_id" not in pms.columns:
        return 1.0

    sub = team_df.copy()
    if "predicted_runs" not in sub.columns:
        return 1.0
    sub["_pr"] = pd.to_numeric(sub["predicted_runs"], errors="coerce").fillna(0.0)
    sub["_pid"] = sub["player_id"].astype(str).str.strip()
    sub = sub.sort_values("_pr", ascending=False).head(int(top_k))

    weights: list[float] = []
    ratios: list[float] = []
    for _, row in sub.iterrows():
        pid = str(row["_pid"]).strip()
        w = float(row["_pr"])
        if w <= 0.5:
            continue
        pr = pms[pms["player_id"].astype(str).str.strip() == pid].copy()
        if as_of is not None and pd.notna(as_of):
            pr = pr[pr["date"] < as_of]
        if len(pr) < 4:
            continue
        pr = pr.sort_values(["date", "match_id"] if "match_id" in pr.columns else ["date"])
        runs = pd.to_numeric(pr["runs"], errors="coerce").fillna(0.0)
        at_v = pr[pr["venue"].apply(lambda v: venue_same_ground(str(v), venue))]
        if len(at_v) >= 2:
            v_avg = float(pd.to_numeric(at_v["runs"], errors="coerce").tail(6).mean())
        elif len(at_v) == 1:
            v_avg = float(pd.to_numeric(at_v["runs"], errors="coerce").iloc[0])
        else:
            continue
        tail = runs.tail(8)
        o_avg = float(tail.mean())
        base = max(o_avg, 10.0)
        ratio = (v_avg / base) - 1.0
        ratio = float(np.clip(ratio, -0.35, 0.45))
        weights.append(w)
        ratios.append(ratio)

    if not weights:
        return 1.0
    w_arr = np.array(weights, dtype=float)
    r_arr = np.array(ratios, dtype=float)
    comp = float((w_arr * r_arr).sum() / max(w_arr.sum(), 1e-9))
    mult = 1.0 + comp * 0.55
    return float(np.clip(mult, 1.0 - max_lift, 1.0 + max_lift))
