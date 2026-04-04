"""
Learned blend of ML vs simulation win probability + optional isotonic calibration.

Trained by train_win_prob_ensemble.py; saved to models/win_prob_ensemble.pkl.
If missing, falls back to fixed 0.6/0.4 blend with no extra calibration.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from scipy.special import expit, logit

from iplpred.paths import MODELS_DIR

ENSEMBLE_BUNDLE_PATH = MODELS_DIR / "win_prob_ensemble.pkl"

_DEFAULT_ML = 0.6
_DEFAULT_SIM = 0.4

# Reported win probabilities are floored/ceiled so isotonic tails never read as 0%/100%.
_REPORT_PROB_MIN = 0.02
_REPORT_PROB_MAX = 0.98


def _clip_p(p: float, eps: float = 1e-6) -> float:
    return float(np.clip(p, eps, 1.0 - eps))


def _clip_report_p(p: float) -> float:
    """Clamp final *reported* ensemble probability to a sensible band."""
    return float(np.clip(p, _REPORT_PROB_MIN, _REPORT_PROB_MAX))


def load_ensemble_bundle() -> dict[str, Any] | None:
    if not ENSEMBLE_BUNDLE_PATH.is_file():
        return None
    b = joblib.load(ENSEMBLE_BUNDLE_PATH)
    return b if isinstance(b, dict) else None


def apply_ensemble_and_calibrate(
    ml_p: float | None,
    sim_p: float | None,
) -> float | None:
    """
    Blend ML and MC sim win prob, then apply optional regime isotonic calibration.
    """
    b = load_ensemble_bundle()
    w_ml = float(b.get("ml_weight", _DEFAULT_ML)) if b else _DEFAULT_ML
    w_sim = float(b.get("sim_weight", _DEFAULT_SIM)) if b else _DEFAULT_SIM
    s = w_ml + w_sim
    if s > 0:
        w_ml, w_sim = w_ml / s, w_sim / s

    if ml_p is None and sim_p is None:
        return None
    if ml_p is None:
        raw = float(sim_p)
    elif sim_p is None:
        raw = float(ml_p)
    else:
        raw = w_ml * float(ml_p) + w_sim * float(sim_p)

    raw = _clip_p(raw)

    if not b:

        return _clip_report_p(raw)

    iso_f = b.get("isotonic_favorite")
    iso_u = b.get("isotonic_underdog")
    use_split = iso_f is not None and iso_u is not None

    if use_split:
        fav = float(ml_p) >= 0.5 if ml_p is not None else raw >= 0.5
        iso = iso_f if fav else iso_u
        try:
            lo = float(logit(raw))
            out = float(iso.predict([lo])[0])
        except Exception:
            out = float(iso.predict([raw])[0])
        return _clip_report_p(_clip_p(out))

    iso = b.get("isotonic")
    if iso is None:
        return _clip_report_p(raw)
    try:
        lo = float(logit(raw))
        out = float(iso.predict([lo])[0])
    except Exception:
        out = float(iso.predict([raw])[0])
    return _clip_report_p(_clip_p(out))


def sim_proxy_from_run_diff(
    team1_runs: float,
    team2_runs: float,
    scale: float = 22.0,
) -> float:
    """Logistic mapping of run differential to [0,1] (team1 perspective)."""
    d = float(team1_runs) - float(team2_runs)
    return float(expit(d / max(scale, 1e-6)))


def get_mc_noise_params() -> dict[str, float]:
    """Shared MC log-variance shock + heteroskedastic noise scale for player preds."""
    b = load_ensemble_bundle()
    if not b:
        return {"mc_shared_log_sigma": 0.025, "het_noise_gamma": 0.45}
    return {
        "mc_shared_log_sigma": float(b.get("mc_shared_log_sigma", 0.025)),
        "het_noise_gamma": float(b.get("het_noise_gamma", 0.45)),
    }
