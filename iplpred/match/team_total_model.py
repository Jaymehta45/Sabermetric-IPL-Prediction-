"""
Optional Ridge models for expected team innings totals (match-level features).
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from iplpred.match.match_winner_model import (
    build_winner_feature_matrix,
    team_pre_match_metrics_from_latest,
)
from iplpred.paths import MODELS_DIR

MODEL_PATH = MODELS_DIR / "team_total_regressor.pkl"


def load_team_total_bundle() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    b = joblib.load(MODEL_PATH)
    if not isinstance(b, dict) or "model_team1" not in b:
        return None
    return b


def predict_team_totals_from_rosters(
    latest: pd.DataFrame,
    team1: list[str],
    team2: list[str],
    *,
    team1_name: str | None = None,
    team2_name: str | None = None,
    match_date: str | None = None,
) -> tuple[float, float] | None:
    """Expected team totals from trained Ridge models, or None if missing."""
    from iplpred.core.team_momentum import momentum_row_from_history
    from iplpred.match.match_winner_model import MOMENTUM_NEUTRAL

    bundle = load_team_total_bundle()
    if bundle is None:
        return None
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
    row = pd.DataFrame(
        [
            {
                "team1_strength": m1["team_strength"],
                "team2_strength": m2["team_strength"],
                "team1_form_runs": m1["team_avg_form_runs"],
                "team2_form_runs": m2["team_avg_form_runs"],
                "team1_form_runs_ipl": m1["team_avg_form_runs_ipl"],
                "team2_form_runs_ipl": m2["team_avg_form_runs_ipl"],
                "team1_momentum": mo1,
                "team2_momentum": mo2,
                "team1_strike_rate": m1["team_avg_strike_rate"],
                "team2_strike_rate": m2["team_avg_strike_rate"],
                "team1_economy": m1["team_avg_economy"],
                "team2_economy": m2["team_avg_economy"],
            }
        ]
    )
    X = build_winner_feature_matrix(row)
    r1 = float(bundle["model_team1"].predict(X)[0])
    r2 = float(bundle["model_team2"].predict(X)[0])
    if bundle.get("target_transform") == "log1p":
        cap = float(bundle.get("target_clip", 320.0))
        r1 = float(np.clip(np.expm1(r1), 0.0, cap))
        r2 = float(np.clip(np.expm1(r2), 0.0, cap))
    return r1, r2
