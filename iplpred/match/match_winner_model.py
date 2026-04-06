"""
Team-level match winner model: feature construction, load, and P(team1 wins).

Trained by train_match_winner_model.py; saved to models/match_winner_classifier.pkl.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from iplpred.paths import MODELS_DIR

WINNER_MODEL_PATH = MODELS_DIR / "match_winner_classifier.pkl"

# Order must match training
WINNER_FEATURE_COLS: list[str] = [
    "strength_diff",
    "form_runs_diff",
    "form_runs_ipl_diff",
    "momentum_diff",
    "strike_rate_diff",
    "economy_diff",
    "team1_strength",
    "team2_strength",
    "team1_form_runs",
    "team2_form_runs",
    "team1_form_runs_ipl",
    "team2_form_runs_ipl",
    "team1_momentum",
    "team2_momentum",
    "team1_strike_rate",
    "team2_strike_rate",
    "team1_economy",
    "team2_economy",
]

MOMENTUM_NEUTRAL = 0.5


def add_winner_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add diff columns; expects team1_* / team2_* base columns."""
    out = df.copy()
    for c in ("team1_form_runs_ipl", "team2_form_runs_ipl"):
        if c not in out.columns:
            alt = "team1_form_runs" if c.startswith("team1") else "team2_form_runs"
            out[c] = out[alt]
    for c in ("team1_momentum", "team2_momentum"):
        if c not in out.columns:
            out[c] = MOMENTUM_NEUTRAL
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(MOMENTUM_NEUTRAL)
    out["strength_diff"] = out["team1_strength"] - out["team2_strength"]
    out["form_runs_diff"] = out["team1_form_runs"] - out["team2_form_runs"]
    out["form_runs_ipl_diff"] = out["team1_form_runs_ipl"] - out["team2_form_runs_ipl"]
    out["momentum_diff"] = out["team1_momentum"] - out["team2_momentum"]
    out["strike_rate_diff"] = out["team1_strike_rate"] - out["team2_strike_rate"]
    out["economy_diff"] = out["team1_economy"] - out["team2_economy"]
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
            f"Winner model not found at {WINNER_MODEL_PATH}. Run train_match_winner_model.py."
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


def learned_team1_win_proba_from_rosters(
    latest: pd.DataFrame,
    team1: list[str],
    team2: list[str],
    *,
    team1_name: str | None = None,
    team2_name: str | None = None,
    match_date: str | None = None,
) -> float | None:
    """
    P(team1 wins) from the trained team-level model, using latest pre-match
    features averaged per roster. Returns None if the model file is missing.
    """
    from iplpred.core.team_momentum import momentum_row_from_history

    t1 = {str(p).strip() for p in team1}
    t2 = {str(p).strip() for p in team2}
    key = latest["player_id"].astype(str).str.strip()
    sub1 = latest[key.isin(t1)]
    sub2 = latest[key.isin(t2)]
    m1 = team_pre_match_metrics_from_latest(sub1)
    m2 = team_pre_match_metrics_from_latest(sub2)
    mo1, mo2 = MOMENTUM_NEUTRAL, MOMENTUM_NEUTRAL
    if team1_name and team2_name:
        mo1, mo2 = momentum_row_from_history(
            str(team1_name).strip(),
            str(team2_name).strip(),
            match_date,
        )
    try:
        return predict_team1_win_proba_single(
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
        )
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
) -> float:
    """Convenience for one match."""
    t1i = team1_form_runs if team1_form_runs_ipl is None else team1_form_runs_ipl
    t2i = team2_form_runs if team2_form_runs_ipl is None else team2_form_runs_ipl
    m1 = MOMENTUM_NEUTRAL if team1_momentum is None else float(team1_momentum)
    m2 = MOMENTUM_NEUTRAL if team2_momentum is None else float(team2_momentum)
    row = pd.DataFrame(
        [
            {
                "team1_strength": team1_strength,
                "team2_strength": team2_strength,
                "team1_form_runs": team1_form_runs,
                "team2_form_runs": team2_form_runs,
                "team1_form_runs_ipl": t1i,
                "team2_form_runs_ipl": t2i,
                "team1_momentum": m1,
                "team2_momentum": m2,
                "team1_strike_rate": team1_strike_rate,
                "team2_strike_rate": team2_strike_rate,
                "team1_economy": team1_economy,
                "team2_economy": team2_economy,
            }
        ]
    )
    return float(predict_team1_win_proba(row)[0])
