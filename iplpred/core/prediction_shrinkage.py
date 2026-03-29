"""
Shrink raw model outputs toward team-level priors when per-player data confidence is low.
"""

from __future__ import annotations

import pandas as pd


def apply_confidence_shrinkage_raw(
    sub: pd.DataFrame,
    confidence: dict[str, float],
    *,
    team_mean_weight: float = 0.45,
) -> pd.DataFrame:
    """
    Blend pred_runs_raw / pred_wk_raw with team means: w*pred + (1-w)*weight*mean.

    `confidence` in [0,1] — higher means trust the model row more (recent games).
    """
    out = sub.copy()
    keys = out["player_id"].astype(str).str.strip()
    w = keys.map(lambda k: float(confidence.get(k, 0.4)))
    w = w.clip(0.0, 1.0)

    mean_r = float(pd.to_numeric(out["pred_runs_raw"], errors="coerce").fillna(0.0).mean())
    mean_w = float(pd.to_numeric(out["pred_wk_raw"], errors="coerce").fillna(0.0).mean())

    pr = pd.to_numeric(out["pred_runs_raw"], errors="coerce").fillna(0.0)
    pw = pd.to_numeric(out["pred_wk_raw"], errors="coerce").fillna(0.0)
    out["pred_runs_raw"] = w * pr + (1.0 - w) * (team_mean_weight * mean_r)
    out["pred_wk_raw"] = w * pw + (1.0 - w) * (team_mean_weight * mean_w)
    return out
