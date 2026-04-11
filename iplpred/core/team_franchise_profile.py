"""
Load team_franchise_profiles.csv and apply innings-aware tweaks to expected totals.

Also exposes flat feature dict for the match-winner / team-total Ridge heads.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from iplpred.core.franchise_normalize import canonical_franchise
from iplpred.paths import PROCESSED_DIR

PROFILES_PATH = PROCESSED_DIR / "team_franchise_profiles.csv"

_NEUTRAL_TEAM = {
    "bat_first_ratio": 1.0,
    "chase_ratio": 1.0,
    "pp_bat_ratio": 1.0,
    "bowl_first_stingy": 1.0,
    "bowl_second_stingy": 1.0,
    "bat_dom": 0.0,
    "bowl_dom": 0.0,
    "field_dom": 0.0,
}


@lru_cache(maxsize=1)
def _profiles_df() -> pd.DataFrame | None:
    if not PROFILES_PATH.is_file():
        return None
    df = pd.read_csv(PROFILES_PATH, low_memory=False)
    if df.empty or "franchise" not in df.columns:
        return None
    return df


def profile_row_for_franchise(name: str) -> dict[str, float]:
    """One franchise's profile scalars (neutral if unknown)."""
    c = canonical_franchise(str(name or "").strip())
    df = _profiles_df()
    if df is None or not c:
        return dict(_NEUTRAL_TEAM)
    hit = df[df["franchise"].astype(str).str.strip() == c]
    if hit.empty:
        return dict(_NEUTRAL_TEAM)
    r = hit.iloc[0]
    out = {}
    for k in _NEUTRAL_TEAM:
        v = pd.to_numeric(r.get(k, np.nan), errors="coerce")
        try:
            out[k] = _NEUTRAL_TEAM[k] if pd.isna(v) else float(v)
        except (TypeError, ValueError):
            out[k] = _NEUTRAL_TEAM[k]
    return out


def franchise_profile_feature_row(team1_name: str, team2_name: str) -> dict[str, float]:
    """
    Columns consumed by match_winner_model / team_total_model (team1 bats first;
    team2 chases in our prediction convention).
    """
    p1 = profile_row_for_franchise(team1_name)
    p2 = profile_row_for_franchise(team2_name)
    return {
        "team1_bat_first_ratio": p1["bat_first_ratio"],
        "team2_bat_first_ratio": p2["bat_first_ratio"],
        "team1_chase_ratio": p1["chase_ratio"],
        "team2_chase_ratio": p2["chase_ratio"],
        "team1_bowl_first_stingy": p1["bowl_first_stingy"],
        "team2_bowl_first_stingy": p2["bowl_first_stingy"],
        "team1_bowl_second_stingy": p1["bowl_second_stingy"],
        "team2_bowl_second_stingy": p2["bowl_second_stingy"],
        "team1_pp_bat_ratio": p1["pp_bat_ratio"],
        "team2_pp_bat_ratio": p2["pp_bat_ratio"],
        "team1_bat_dom": p1["bat_dom"],
        "team2_bat_dom": p2["bat_dom"],
        "team1_bowl_dom": p1["bowl_dom"],
        "team2_bowl_dom": p2["bowl_dom"],
        "team1_field_dom": p1["field_dom"],
        "team2_field_dom": p2["field_dom"],
    }


def apply_franchise_innings_profile(
    team1_name: str | None,
    team2_name: str | None,
    tgt1: float,
    tgt2: float,
) -> tuple[float, float]:
    """
    team1 = first innings, team2 = chase. Uses historical1st vs 2nd innings
    productivity and opposite bowling stinginess. Clamped multiplicative tweak.
    """
    if not team1_name or not team2_name:
        return tgt1, tgt2
    a = profile_row_for_franchise(team1_name)
    b = profile_row_for_franchise(team2_name)

    # First innings: batting side sets total; opposing bowled first-innings overs = team2 bowls in inn1.
    adj1 = (
        0.20 * (a["bat_first_ratio"] - 1.0)
        + 0.10 * (a["pp_bat_ratio"] - 1.0)
        - 0.12 * (b["bowl_first_stingy"] - 1.0)
        + 0.04 * (a["bat_dom"] - b["bowl_dom"]) / 2.5
    )
    # Chase: team2 bats second; team1 defends with second-innings bowling.
    adj2 = (
        0.20 * (b["chase_ratio"] - 1.0)
        + 0.10 * (b["pp_bat_ratio"] - 1.0)
        - 0.12 * (a["bowl_second_stingy"] - 1.0)
        + 0.04 * (b["bat_dom"] - a["bowl_dom"]) / 2.5
    )

    adj1 = float(np.clip(adj1, -0.14, 0.14))
    adj2 = float(np.clip(adj2, -0.14, 0.14))
    return tgt1 * (1.0 + adj1), tgt2 * (1.0 + adj2)
