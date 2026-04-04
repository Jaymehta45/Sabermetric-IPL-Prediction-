"""
Inference-time feature selection: venue history, opponent franchise strength, and
head-to-head matchup strike rates — without rebuilding player_features.csv.

Uses the same historical tables as build_features (player_match_stats + prepared
player rows) so RF inputs align with training semantics.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from iplpred.paths import PROCESSED_DIR
from iplpred.pipeline.build_features import (
    build_base_columns,
    build_team_match_strength_past,
    load_match_stats,
)
from iplpred.training.train_player_model import load_and_prepare

PLAYER_MATCH_STATS_PATH = PROCESSED_DIR / "player_match_stats.csv"

# Columns merged from ball-by-ball stats for matchup SR at inference.
_PMS_EXTRA_COLS = [
    "match_id",
    "player_id",
    "venue",
    "balls",
    "balls_bowled",
    "runs_conceded",
]


def _norm_franchise(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def franchise_match(a: str, b: str) -> bool:
    """Loose franchise equality (handles partial / ordering)."""
    a, b = _norm_franchise(a), _norm_franchise(b)
    if not a or not b:
        return False
    if a == b:
        return True
    return a in b or b in a


def venue_matches(cell: object, query: str) -> bool:
    """True if historical venue string matches the upcoming stadium query."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return False
    q = str(query).strip().lower()
    if not q:
        return False
    c = str(cell).strip().lower()
    if q in c:
        return True
    for tok in re.findall(r"[a-zA-Z]{4,}", q):
        if tok in c:
            return True
    return False


@lru_cache(maxsize=1)
def _team_strength_last_by_team() -> pd.DataFrame:
    """Most recent past batting/bowling strength per franchise (training-time definition)."""
    pms = load_match_stats()
    pms = build_base_columns(pms)
    tm = build_team_match_strength_past(pms)
    return tm.sort_values(["team", "date", "match_id"]).groupby("team", as_index=False).last()


def resolve_team_strength_row(team_name: str, tm: pd.DataFrame) -> pd.Series | None:
    if not (team_name or "").strip():
        return None
    want = _norm_franchise(team_name)
    best: tuple[int, pd.Series] | None = None
    for _, r in tm.iterrows():
        t = _norm_franchise(str(r["team"]))
        if t == want:
            return r
        if want in t or t in want:
            score = len(t)
            if best is None or score > best[0]:
                best = (score, r)
    return None if best is None else best[1]


def attach_match_stat_extras(df: pd.DataFrame) -> pd.DataFrame:
    """Add venue + balls from player_match_stats (not always in player_features CSV)."""
    if not PLAYER_MATCH_STATS_PATH.is_file():
        df = df.copy()
        if "venue" not in df.columns:
            df["venue"] = np.nan
        for c in ("balls", "balls_bowled", "runs_conceded"):
            if c not in df.columns:
                df[c] = np.nan
        return df
    pms = pd.read_csv(PLAYER_MATCH_STATS_PATH, low_memory=False)
    keep = [c for c in _PMS_EXTRA_COLS if c in pms.columns]
    pms = pms[keep].drop_duplicates(subset=["match_id", "player_id"])
    out = df.merge(pms, on=["match_id", "player_id"], how="left", suffixes=("", "_pms"))
    if "venue_pms" in out.columns:
        if "venue" in out.columns:
            out["venue"] = out["venue_pms"].combine_first(out["venue"])
        else:
            out["venue"] = out["venue_pms"]
        out = out.drop(columns=["venue_pms"], errors="ignore")
    elif "venue" not in out.columns:
        out["venue"] = np.nan
    return out


def _pick_row_at_venue(sub: pd.DataFrame, venue: str | None) -> pd.Series:
    """Latest row; if venue given, prefer most recent appearance at that ground."""
    if sub.empty:
        raise ValueError("empty player history")
    sub = sub.sort_values(["date", "match_id"])
    if not venue or not str(venue).strip():
        return sub.iloc[-1]
    if "venue" not in sub.columns:
        return sub.iloc[-1]
    mask = sub["venue"].apply(lambda v: venue_matches(v, venue))
    at = sub[mask]
    if len(at) > 0:
        return at.iloc[-1]
    return sub.iloc[-1]


def _matchup_sr_vs_franchise(sub: pd.DataFrame, opponent: str) -> tuple[float | None, float | None]:
    """Mean batting SR and bowling SR conceded when facing this opponent franchise (history)."""
    if "opponent_team" not in sub.columns:
        return None, None
    m = sub["opponent_team"].astype(str).apply(lambda x: franchise_match(x, opponent))
    osub = sub[m].sort_values(["date", "match_id"])
    if len(osub) == 0:
        return None, None
    tail = osub.tail(12)
    bat_sr: float | None = None
    bow_sr: float | None = None
    if "balls" in tail.columns and "runs" in tail.columns:
        balls = pd.to_numeric(tail["balls"], errors="coerce").fillna(0.0)
        runs = pd.to_numeric(tail["runs"], errors="coerce").fillna(0.0)
        valid = balls > 0
        if valid.any():
            bat_sr = float((runs[valid] / balls[valid] * 100.0).mean())
    if bat_sr is None and "batting_matchup_sr" in tail.columns:
        bat_sr = float(pd.to_numeric(tail["batting_matchup_sr"], errors="coerce").mean())
        if np.isnan(bat_sr):
            bat_sr = None
    if "balls_bowled" in tail.columns and "runs_conceded" in tail.columns:
        bb = pd.to_numeric(tail["balls_bowled"], errors="coerce").fillna(0.0)
        rc = pd.to_numeric(tail["runs_conceded"], errors="coerce").fillna(0.0)
        vb = bb > 0
        if vb.any():
            bow_sr = float((rc[vb] / bb[vb] * 100.0).mean())
    if bow_sr is None and "bowling_matchup_sr" in tail.columns:
        bow_sr = float(pd.to_numeric(tail["bowling_matchup_sr"], errors="coerce").mean())
        if np.isnan(bow_sr):
            bow_sr = None
    return bat_sr, bow_sr


def _opponent_strength_scalar(role: str, past_bat: float, past_bowl: float) -> float:
    r = (role or "allrounder").strip().lower()
    if r == "batter":
        return float(past_bowl)
    if r == "bowler":
        return float(past_bat)
    return (float(past_bat) + float(past_bowl)) / 2.0


def build_fixture_player_frame(
    player_ids: list[str],
    *,
    venue: str | None,
    team1_name: str | None,
    team2_name: str | None,
    team1: list[str],
    team2: list[str],
) -> pd.DataFrame:
    """
    One row per player: venue-aware row pick + opponent strength + H2H matchup SRs.

    ``team1`` / ``team2`` are playing XI player_id lists (same order as match_simulator).
    """
    df = load_and_prepare()
    df = attach_match_stat_extras(df)

    opp_by_player: dict[str, str] = {}
    n1 = (team1_name or "").strip()
    n2 = (team2_name or "").strip()
    if n1 and n2:
        for p in team1:
            opp_by_player[str(p).strip()] = n2
        for p in team2:
            opp_by_player[str(p).strip()] = n1

    tm = _team_strength_last_by_team()
    rows: list[pd.Series] = []
    for pid in player_ids:
        key = str(pid).strip()
        sub = df[df["player_id"].astype(str).str.strip() == key].sort_values(["date", "match_id"])
        if sub.empty:
            raise ValueError(f"Player not found in prepared data: {key!r}")
        base = _pick_row_at_venue(sub, venue).copy()
        opp = opp_by_player.get(key)
        if opp:
            tr = resolve_team_strength_row(opp, tm)
            if tr is not None:
                obat = float(tr.get("past_batting_strength", np.nan))
                obowl = float(tr.get("past_bowling_strength", np.nan))
                if not np.isnan(obat):
                    base["opponent_past_batting_strength"] = obat
                if not np.isnan(obowl):
                    base["opponent_past_bowling_strength"] = obowl
                role = str(base.get("role", "allrounder"))
                base["opponent_strength"] = _opponent_strength_scalar(
                    role,
                    float(base.get("opponent_past_batting_strength", obat)),
                    float(base.get("opponent_past_bowling_strength", obowl)),
                )
            bsr, bwsr = _matchup_sr_vs_franchise(sub, opp)
            if bsr is not None and np.isfinite(bsr):
                base["batting_matchup_sr"] = float(np.clip(bsr, 40.0, 220.0))
            if bwsr is not None and np.isfinite(bwsr):
                base["bowling_matchup_sr"] = float(np.clip(bwsr, 40.0, 220.0))

        rows.append(base)

    out = pd.DataFrame(rows)
    # Preserve column order similar to training
    return out


def should_use_fixture_context(
    venue: str | None,
    team1_name: str | None,
    team2_name: str | None,
) -> bool:
    v = (venue or "").strip()
    return bool(v) or (bool((team1_name or "").strip()) and bool((team2_name or "").strip()))
