"""
Team-level match winner model: feature construction, load, and P(team1 wins).

Trained by ``python -m iplpred.training.train_match_winner_model``; saved to models/match_winner_classifier.pkl.
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd

from iplpred.core.franchise_normalize import canonical_franchise
from iplpred.paths import MODELS_DIR

WINNER_MODEL_PATH = MODELS_DIR / "match_winner_classifier.pkl"

# Order must match training
WINNER_FEATURE_COLS: list[str] = [
    "strength_diff",
    "form_runs_diff",
    "form_runs_ipl_diff",
    "momentum_diff",
    "venue_momentum_diff",
    "strike_rate_diff",
    "economy_diff",
    "bat_dom_diff",
    "bowl_dom_diff",
    "field_dom_diff",
    "death_bat_ratio_diff",
    "death_bowl_stingy_diff",
    "second_innings_dew_prior",
    "team1_strength",
    "team2_strength",
    "team1_form_runs",
    "team2_form_runs",
    "team1_form_runs_ipl",
    "team2_form_runs_ipl",
    "team1_momentum",
    "team2_momentum",
    "team1_venue_momentum",
    "team2_venue_momentum",
    "team1_strike_rate",
    "team2_strike_rate",
    "team1_economy",
    "team2_economy",
    "team1_bat_first_ratio",
    "team2_bat_first_ratio",
    "team1_chase_ratio",
    "team2_chase_ratio",
    "team1_bowl_first_stingy",
    "team2_bowl_first_stingy",
    "team1_bowl_second_stingy",
    "team2_bowl_second_stingy",
    "team1_pp_bat_ratio",
    "team2_pp_bat_ratio",
    "team1_death_bat_ratio",
    "team2_death_bat_ratio",
    "team1_death_bowl_stingy",
    "team2_death_bowl_stingy",
    "team1_bat_dom",
    "team2_bat_dom",
    "team1_bowl_dom",
    "team2_bowl_dom",
    "team1_field_dom",
    "team2_field_dom",
    "h2h_team1_win_prior",
    "toss_team1_won",
    "team1_bats_first_signal",
    "season_win_pct_diff",
    "season_run_margin_diff",
    "lineup_bowler_share_diff",
    "pp_sr_diff",
    "mid_sr_diff",
    "death_sr_diff",
    "match_humidity_prior",
    "match_rain_risk",
    "injury_availability_diff",
]

MOMENTUM_NEUTRAL = 0.5


def _default_profile_columns(out: pd.DataFrame) -> None:
    ratio_cols = [
        "team1_bat_first_ratio",
        "team2_bat_first_ratio",
        "team1_chase_ratio",
        "team2_chase_ratio",
        "team1_bowl_first_stingy",
        "team2_bowl_first_stingy",
        "team1_bowl_second_stingy",
        "team2_bowl_second_stingy",
        "team1_pp_bat_ratio",
        "team2_pp_bat_ratio",
        "team1_death_bat_ratio",
        "team2_death_bat_ratio",
        "team1_death_bowl_stingy",
        "team2_death_bowl_stingy",
    ]
    dom_cols = [
        "team1_bat_dom",
        "team2_bat_dom",
        "team1_bowl_dom",
        "team2_bowl_dom",
        "team1_field_dom",
        "team2_field_dom",
    ]
    for c in ratio_cols:
        if c not in out.columns:
            out[c] = 1.0
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(1.0)
    for c in dom_cols:
        if c not in out.columns:
            out[c] = 0.0
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)


def add_winner_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add diff columns; expects team1_* / team2_* base columns."""
    out = df.copy()
    _default_profile_columns(out)
    for c in ("team1_form_runs_ipl", "team2_form_runs_ipl"):
        if c not in out.columns:
            alt = "team1_form_runs" if c.startswith("team1") else "team2_form_runs"
            out[c] = out[alt]
    for c in ("team1_momentum", "team2_momentum"):
        if c not in out.columns:
            out[c] = MOMENTUM_NEUTRAL
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(MOMENTUM_NEUTRAL)
    for c in ("team1_venue_momentum", "team2_venue_momentum"):
        if c not in out.columns:
            out[c] = MOMENTUM_NEUTRAL
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(MOMENTUM_NEUTRAL)
    if "second_innings_dew_prior" not in out.columns:
        out["second_innings_dew_prior"] = 0.0
    else:
        out["second_innings_dew_prior"] = pd.to_numeric(
            out["second_innings_dew_prior"], errors="coerce"
        ).fillna(0.0)
    for c, dflt in [
        ("h2h_team1_win_prior", MOMENTUM_NEUTRAL),
        ("toss_team1_won", 0.5),
        ("team1_bats_first_signal", 0.5),
        ("team1_season_win_pct_prior", MOMENTUM_NEUTRAL),
        ("team2_season_win_pct_prior", MOMENTUM_NEUTRAL),
        ("team1_season_run_margin_prior", 0.0),
        ("team2_season_run_margin_prior", 0.0),
        ("team1_lineup_bowler_share", 0.35),
        ("team2_lineup_bowler_share", 0.35),
        ("team1_avg_form_pp_sr", 120.0),
        ("team2_avg_form_pp_sr", 120.0),
        ("team1_avg_form_mid_sr", 120.0),
        ("team2_avg_form_mid_sr", 120.0),
        ("team1_avg_form_death_sr", 120.0),
        ("team2_avg_form_death_sr", 120.0),
        ("match_humidity_prior", 0.0),
        ("match_rain_risk", 0.0),
        ("team1_injury_availability", 1.0),
        ("team2_injury_availability", 1.0),
    ]:
        if c not in out.columns:
            out[c] = dflt
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(dflt)
    out["season_win_pct_diff"] = (
        out["team1_season_win_pct_prior"] - out["team2_season_win_pct_prior"]
    )
    out["season_run_margin_diff"] = (
        out["team1_season_run_margin_prior"] - out["team2_season_run_margin_prior"]
    )
    out["lineup_bowler_share_diff"] = (
        out["team1_lineup_bowler_share"] - out["team2_lineup_bowler_share"]
    )
    out["pp_sr_diff"] = out["team1_avg_form_pp_sr"] - out["team2_avg_form_pp_sr"]
    out["mid_sr_diff"] = out["team1_avg_form_mid_sr"] - out["team2_avg_form_mid_sr"]
    out["death_sr_diff"] = out["team1_avg_form_death_sr"] - out["team2_avg_form_death_sr"]
    out["injury_availability_diff"] = (
        out["team1_injury_availability"] - out["team2_injury_availability"]
    )
    out["strength_diff"] = out["team1_strength"] - out["team2_strength"]
    out["form_runs_diff"] = out["team1_form_runs"] - out["team2_form_runs"]
    out["form_runs_ipl_diff"] = out["team1_form_runs_ipl"] - out["team2_form_runs_ipl"]
    out["momentum_diff"] = out["team1_momentum"] - out["team2_momentum"]
    out["venue_momentum_diff"] = out["team1_venue_momentum"] - out["team2_venue_momentum"]
    out["strike_rate_diff"] = out["team1_strike_rate"] - out["team2_strike_rate"]
    out["economy_diff"] = out["team1_economy"] - out["team2_economy"]
    out["bat_dom_diff"] = out["team1_bat_dom"] - out["team2_bat_dom"]
    out["bowl_dom_diff"] = out["team1_bowl_dom"] - out["team2_bowl_dom"]
    out["field_dom_diff"] = out["team1_field_dom"] - out["team2_field_dom"]
    out["death_bat_ratio_diff"] = out["team1_death_bat_ratio"] - out["team2_death_bat_ratio"]
    out["death_bowl_stingy_diff"] = (
        out["team1_death_bowl_stingy"] - out["team2_death_bowl_stingy"]
    )
    return out


def build_winner_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    d = add_winner_feature_columns(df)
    for c in WINNER_FEATURE_COLS:
        if c not in d.columns:
            raise ValueError(f"Missing column for winner model: {c}")
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[WINNER_FEATURE_COLS].values.astype(np.float64)


def load_winner_model() -> dict[str, Any]:
    if not WINNER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Winner model not found at {WINNER_MODEL_PATH}. Run: python -m iplpred.training.train_match_winner_model"
        )
    bundle = joblib.load(WINNER_MODEL_PATH)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Invalid winner model bundle")
    return bundle


def predict_team1_win_proba(df: pd.DataFrame) -> np.ndarray:
    """
    Probability that team1 wins (one score per row).
    `df` must contain team1_strength … team2_economy (diffs are optional; recomputed).
    """
    bundle = load_winner_model()
    clf = bundle["model"]
    X = build_winner_feature_matrix(df)
    return clf.predict_proba(X)[:, 1]


def team_pre_match_metrics_from_latest(latest_df: pd.DataFrame) -> dict[str, float]:
    """
    Aggregate pre-match team signals from one row per player (same formula as
    build_training_dataset.team_aggregates / team_strength_score).
    """
    if latest_df.empty:
        return {
            "team_strength": 0.0,
            "team_avg_form_runs": 0.0,
            "team_avg_form_wickets": 0.0,
            "team_avg_form_runs_ipl": 0.0,
            "team_avg_form_wickets_ipl": 0.0,
            "team_avg_strike_rate": 0.0,
            "team_avg_economy": 0.0,
        }
    fr = pd.to_numeric(latest_df["form_runs"], errors="coerce").fillna(0.0).mean()
    fw = pd.to_numeric(latest_df["form_wickets"], errors="coerce").fillna(0.0).mean()
    if "form_runs_ipl" in latest_df.columns:
        fri = pd.to_numeric(latest_df["form_runs_ipl"], errors="coerce").fillna(fr).mean()
    else:
        fri = fr
    if "form_wickets_ipl" in latest_df.columns:
        fwi = pd.to_numeric(latest_df["form_wickets_ipl"], errors="coerce").fillna(fw).mean()
    else:
        fwi = fw
    sr = pd.to_numeric(latest_df["strike_rate"], errors="coerce").fillna(0.0).mean()
    econ = pd.to_numeric(latest_df["economy"], errors="coerce").fillna(0.0).mean()
    strength = 0.4 * fri + 0.3 * sr + 0.3 * fwi
    return {
        "team_strength": float(strength),
        "team_avg_form_runs": float(fr),
        "team_avg_form_wickets": float(fw),
        "team_avg_form_runs_ipl": float(fri),
        "team_avg_form_wickets_ipl": float(fwi),
        "team_avg_strike_rate": float(sr),
        "team_avg_economy": float(econ),
    }


def lineup_bowler_share_from_latest(latest: pd.DataFrame, team_ids: set[str]) -> float:
    """Fraction of roster rows treated as bowlers (role column or bowling-heavy heuristic)."""
    key = latest["player_id"].astype(str).str.strip()
    sub = latest[key.isin(team_ids)]
    if sub.empty:
        return 0.35
    if "role" in sub.columns:
        r = sub["role"].fillna("").astype(str).str.strip().str.lower()
        return float((r == "bowler").mean()) if len(r) else 0.35
    bb = pd.to_numeric(sub.get("balls_bowled"), errors="coerce").fillna(0.0)
    b = pd.to_numeric(sub.get("balls"), errors="coerce").fillna(0.0)
    return float((bb > b * 0.4).mean()) if len(sub) else 0.35


def learned_team1_win_proba_from_rosters(
    latest: pd.DataFrame,
    team1: list[str],
    team2: list[str],
    *,
    team1_name: str | None = None,
    team2_name: str | None = None,
    match_date: str | None = None,
    venue: str | None = None,
    second_innings_dew_prior: float = 0.0,
    toss_team1_won: float | None = None,
    team1_bats_first_signal: float | None = None,
    match_humidity_prior: float = 0.0,
    match_rain_risk: float = 0.0,
    team1_injury_availability: float = 1.0,
    team2_injury_availability: float = 1.0,
    season: str | None = None,
) -> float | None:
    """
    P(team1 wins) from the trained team-level model, using latest pre-match
    features averaged per roster. Returns None if the model file is missing.
    """
    from iplpred.core.match_history_extras import h2h_prior_from_history, season_priors_from_history
    from iplpred.core.team_momentum import momentum_row_from_history, venue_momentum_row_from_history

    t1 = {str(p).strip() for p in team1}
    t2 = {str(p).strip() for p in team2}
    key = latest["player_id"].astype(str).str.strip()
    sub1 = latest[key.isin(t1)]
    sub2 = latest[key.isin(t2)]
    m1 = team_pre_match_metrics_from_latest(sub1)
    m2 = team_pre_match_metrics_from_latest(sub2)
    mo1, mo2 = MOMENTUM_NEUTRAL, MOMENTUM_NEUTRAL
    vo1, vo2 = MOMENTUM_NEUTRAL, MOMENTUM_NEUTRAL
    h2h = MOMENTUM_NEUTRAL
    s1wp = s2wp = MOMENTUM_NEUTRAL
    s1m = s2m = 0.0
    l1 = l2 = 0.35
    pp1 = pp2 = mid1 = mid2 = d1 = d2 = 120.0
    if team1_name and team2_name:
        mo1, mo2 = momentum_row_from_history(
            str(team1_name).strip(),
            str(team2_name).strip(),
            match_date,
        )
        vo1, vo2 = venue_momentum_row_from_history(
            str(team1_name).strip(),
            str(team2_name).strip(),
            venue,
            match_date,
        )
        h2h = h2h_prior_from_history(
            str(team1_name).strip(),
            str(team2_name).strip(),
            match_date,
        )
        md = match_date or ""
        sy = season
        if not sy and match_date:
            sy = str(pd.to_datetime(match_date, errors="coerce").year)
        if not sy:
            sy = "2024"
        s1wp, s2wp, s1m, s2m = season_priors_from_history(
            str(team1_name).strip(),
            str(team2_name).strip(),
            sy,
            match_date,
        )
        l1 = lineup_bowler_share_from_latest(latest, t1)
        l2 = lineup_bowler_share_from_latest(latest, t2)
        sub1 = latest[latest["player_id"].astype(str).str.strip().isin(t1)]
        sub2 = latest[latest["player_id"].astype(str).str.strip().isin(t2)]
        if "form_pp_sr" in sub1.columns:
            pp1 = float(pd.to_numeric(sub1["form_pp_sr"], errors="coerce").fillna(120.0).mean())
            pp2 = float(pd.to_numeric(sub2["form_pp_sr"], errors="coerce").fillna(120.0).mean())
        if "form_mid_sr" in sub1.columns:
            mid1 = float(pd.to_numeric(sub1["form_mid_sr"], errors="coerce").fillna(120.0).mean())
            mid2 = float(pd.to_numeric(sub2["form_mid_sr"], errors="coerce").fillna(120.0).mean())
        if "form_death_sr" in sub1.columns:
            d1 = float(pd.to_numeric(sub1["form_death_sr"], errors="coerce").fillna(120.0).mean())
            d2 = float(pd.to_numeric(sub2["form_death_sr"], errors="coerce").fillna(120.0).mean())
    tw = 0.5 if toss_team1_won is None else float(toss_team1_won)
    bf = 0.5 if team1_bats_first_signal is None else float(team1_bats_first_signal)
    try:
        p = predict_team1_win_proba_single(
            team1_strength=m1["team_strength"],
            team2_strength=m2["team_strength"],
            team1_form_runs=m1["team_avg_form_runs"],
            team2_form_runs=m2["team_avg_form_runs"],
            team1_form_runs_ipl=m1["team_avg_form_runs_ipl"],
            team2_form_runs_ipl=m2["team_avg_form_runs_ipl"],
            team1_strike_rate=m1["team_avg_strike_rate"],
            team2_strike_rate=m2["team_avg_strike_rate"],
            team1_economy=m1["team_avg_economy"],
            team2_economy=m2["team_avg_economy"],
            team1_momentum=mo1,
            team2_momentum=mo2,
            team1_venue_momentum=vo1,
            team2_venue_momentum=vo2,
            second_innings_dew_prior=float(second_innings_dew_prior or 0.0),
            team1_name=team1_name or "",
            team2_name=team2_name or "",
            h2h_team1_win_prior=h2h,
            toss_team1_won=tw,
            team1_bats_first_signal=bf,
            team1_season_win_pct_prior=s1wp,
            team2_season_win_pct_prior=s2wp,
            team1_season_run_margin_prior=s1m,
            team2_season_run_margin_prior=s2m,
            team1_lineup_bowler_share=l1,
            team2_lineup_bowler_share=l2,
            team1_avg_form_pp_sr=pp1,
            team2_avg_form_pp_sr=pp2,
            team1_avg_form_mid_sr=mid1,
            team2_avg_form_mid_sr=mid2,
            team1_avg_form_death_sr=d1,
            team2_avg_form_death_sr=d2,
            match_humidity_prior=float(match_humidity_prior),
            match_rain_risk=float(match_rain_risk),
            team1_injury_availability=float(team1_injury_availability),
            team2_injury_availability=float(team2_injury_availability),
        )
        if (
            p is not None
            and venue
            and team1_name
            and team2_name
            and os.environ.get("IPLPRED_NO_GT_AHMEDABAD_ROSTER_NUDGE", "").strip() != "1"
        ):
            vlow = str(venue).lower()
            t1c = canonical_franchise(str(team1_name).strip())
            t2c = canonical_franchise(str(team2_name).strip())
            gtc = canonical_franchise("Gujarat Titans")
            if "ahmedabad" in vlow and t2c == gtc and t1c != gtc:
                p = float(np.clip(p - 0.024, 1e-6, 1.0 - 1e-6))
        return p
    except FileNotFoundError:
        return None


def predict_team1_win_proba_single(
    *,
    team1_strength: float,
    team2_strength: float,
    team1_form_runs: float,
    team2_form_runs: float,
    team1_form_runs_ipl: float | None = None,
    team2_form_runs_ipl: float | None = None,
    team1_strike_rate: float,
    team2_strike_rate: float,
    team1_economy: float,
    team2_economy: float,
    team1_momentum: float | None = None,
    team2_momentum: float | None = None,
    team1_venue_momentum: float | None = None,
    team2_venue_momentum: float | None = None,
    second_innings_dew_prior: float = 0.0,
    team1_name: str = "",
    team2_name: str = "",
    h2h_team1_win_prior: float | None = None,
    toss_team1_won: float = 0.5,
    team1_bats_first_signal: float = 0.5,
    team1_season_win_pct_prior: float | None = None,
    team2_season_win_pct_prior: float | None = None,
    team1_season_run_margin_prior: float | None = None,
    team2_season_run_margin_prior: float | None = None,
    team1_lineup_bowler_share: float | None = None,
    team2_lineup_bowler_share: float | None = None,
    team1_avg_form_pp_sr: float | None = None,
    team2_avg_form_pp_sr: float | None = None,
    team1_avg_form_mid_sr: float | None = None,
    team2_avg_form_mid_sr: float | None = None,
    team1_avg_form_death_sr: float | None = None,
    team2_avg_form_death_sr: float | None = None,
    match_humidity_prior: float = 0.0,
    match_rain_risk: float = 0.0,
    team1_injury_availability: float = 1.0,
    team2_injury_availability: float = 1.0,
) -> float:
    """Convenience for one match."""
    from iplpred.core.team_franchise_profile import franchise_profile_feature_row

    t1i = team1_form_runs if team1_form_runs_ipl is None else team1_form_runs_ipl
    t2i = team2_form_runs if team2_form_runs_ipl is None else team2_form_runs_ipl
    m1 = MOMENTUM_NEUTRAL if team1_momentum is None else float(team1_momentum)
    m2 = MOMENTUM_NEUTRAL if team2_momentum is None else float(team2_momentum)
    v1 = MOMENTUM_NEUTRAL if team1_venue_momentum is None else float(team1_venue_momentum)
    v2 = MOMENTUM_NEUTRAL if team2_venue_momentum is None else float(team2_venue_momentum)
    h2h = MOMENTUM_NEUTRAL if h2h_team1_win_prior is None else float(h2h_team1_win_prior)
    s1p = MOMENTUM_NEUTRAL if team1_season_win_pct_prior is None else float(team1_season_win_pct_prior)
    s2p = MOMENTUM_NEUTRAL if team2_season_win_pct_prior is None else float(team2_season_win_pct_prior)
    s1rm = 0.0 if team1_season_run_margin_prior is None else float(team1_season_run_margin_prior)
    s2rm = 0.0 if team2_season_run_margin_prior is None else float(team2_season_run_margin_prior)
    lb1 = 0.35 if team1_lineup_bowler_share is None else float(team1_lineup_bowler_share)
    lb2 = 0.35 if team2_lineup_bowler_share is None else float(team2_lineup_bowler_share)
    pp1 = 120.0 if team1_avg_form_pp_sr is None else float(team1_avg_form_pp_sr)
    pp2 = 120.0 if team2_avg_form_pp_sr is None else float(team2_avg_form_pp_sr)
    mid1 = 120.0 if team1_avg_form_mid_sr is None else float(team1_avg_form_mid_sr)
    mid2 = 120.0 if team2_avg_form_mid_sr is None else float(team2_avg_form_mid_sr)
    dd1 = 120.0 if team1_avg_form_death_sr is None else float(team1_avg_form_death_sr)
    dd2 = 120.0 if team2_avg_form_death_sr is None else float(team2_avg_form_death_sr)
    base = {
        "team1_strength": team1_strength,
        "team2_strength": team2_strength,
        "team1_form_runs": team1_form_runs,
        "team2_form_runs": team2_form_runs,
        "team1_form_runs_ipl": t1i,
        "team2_form_runs_ipl": t2i,
        "team1_momentum": m1,
        "team2_momentum": m2,
        "team1_venue_momentum": v1,
        "team2_venue_momentum": v2,
        "second_innings_dew_prior": float(second_innings_dew_prior or 0.0),
        "team1_strike_rate": team1_strike_rate,
        "team2_strike_rate": team2_strike_rate,
        "team1_economy": team1_economy,
        "team2_economy": team2_economy,
        "h2h_team1_win_prior": h2h,
        "toss_team1_won": float(toss_team1_won),
        "team1_bats_first_signal": float(team1_bats_first_signal),
        "team1_season_win_pct_prior": s1p,
        "team2_season_win_pct_prior": s2p,
        "team1_season_run_margin_prior": s1rm,
        "team2_season_run_margin_prior": s2rm,
        "team1_lineup_bowler_share": lb1,
        "team2_lineup_bowler_share": lb2,
        "team1_avg_form_pp_sr": pp1,
        "team2_avg_form_pp_sr": pp2,
        "team1_avg_form_mid_sr": mid1,
        "team2_avg_form_mid_sr": mid2,
        "team1_avg_form_death_sr": dd1,
        "team2_avg_form_death_sr": dd2,
        "match_humidity_prior": float(match_humidity_prior),
        "match_rain_risk": float(match_rain_risk),
        "team1_injury_availability": float(team1_injury_availability),
        "team2_injury_availability": float(team2_injury_availability),
    }
    base.update(franchise_profile_feature_row(str(team1_name).strip(), str(team2_name).strip()))
    row = pd.DataFrame([base])
    return float(predict_team1_win_proba(row)[0])
