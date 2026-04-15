"""
Learned blend of ML vs simulation win probability, optional logit stacking of both,
and optional isotonic calibration.

Trained by iplpred/training/train_win_prob_ensemble.py → models/win_prob_ensemble.pkl.
If missing, falls back to fixed 0.6/0.4 blend with no extra calibration.

Reported hybrid probabilities are not artificially capped away from 0%/100%
(beyond a tiny epsilon clip so values stay in (0, 1) for numerics).
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


def _clip_p(p: float, eps: float = 1e-6) -> float:
    return float(np.clip(p, eps, 1.0 - eps))


def _finalize_report_p(p: float) -> float:
    """Return the blended/calibrated probability for display — only epsilon clip for logit stability."""
    return _clip_p(float(p))


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
    Blend ML and MC sim win prob (or learned logit stack of both), then apply optional isotonic.
    """
    b = load_ensemble_bundle()
    w_ml = float(b.get("ml_weight", _DEFAULT_ML)) if b else _DEFAULT_ML
    w_sim = float(b.get("sim_weight", _DEFAULT_SIM)) if b else _DEFAULT_SIM
    s = w_ml + w_sim
    if s > 0:
        w_ml, w_sim = w_ml / s, w_sim / s

    if ml_p is None and sim_p is None:
        return None

    stack = b.get("stack_logit") if b else None
    if (
        isinstance(stack, dict)
        and ml_p is not None
        and sim_p is not None
        and stack.get("coef") is not None
    ):
        coef = np.asarray(stack["coef"], dtype=float).ravel()
        b0 = float(stack.get("intercept", 0.0))
        if len(coef) >= 2:
            lo_m = float(logit(_clip_p(float(ml_p))))
            lo_s = float(logit(_clip_p(float(sim_p))))
            raw = float(expit(b0 + float(coef[0]) * lo_m + float(coef[1]) * lo_s))
            raw = _clip_p(raw)
        else:
            raw = _clip_p(w_ml * float(ml_p) + w_sim * float(sim_p))
    elif ml_p is None:
        raw = float(sim_p)
    elif sim_p is None:
        raw = float(ml_p)
    else:
        w_m, w_s = w_ml, w_sim
        if b and (stack is None or stack.get("coef") is None):
            gap = abs(float(ml_p) - float(sim_p))
            th = float(b.get("divergence_thresh", 0.18))
            boost = float(b.get("divergence_sim_boost", 0.35))
            if gap >= th:
                w_s = float(np.clip(w_s + boost * gap, 0.15, 0.85))
                w_m = 1.0 - w_s
        raw = w_m * float(ml_p) + w_s * float(sim_p)
        raw = _clip_p(raw)

    raw = _clip_p(raw)

    if not b:
        return _finalize_report_p(raw)

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
        return _finalize_report_p(out)

    iso = b.get("isotonic")
    if iso is None:
        return _finalize_report_p(raw)
    try:
        lo = float(logit(raw))
        out = float(iso.predict([lo])[0])
    except Exception:
        out = float(iso.predict([raw])[0])
    return _finalize_report_p(out)


def sim_proxy_from_run_diff(
    team1_runs: float,
    team2_runs: float,
    scale: float = 22.0,
) -> float:
    """Logistic mapping of run differential to [0,1] (team1 perspective)."""
    d = float(team1_runs) - float(team2_runs)
    return float(expit(d / max(scale, 1e-6)))


def get_mc_noise_params() -> dict[str, float]:
    """
    Shared MC log-variance shock + heteroskedastic noise for player preds.

    MC loops use ``mc_runs_noise_scale`` / ``mc_wk_noise_scale`` (per-player Gaussian
    on RF outputs). The headline (six outcomes) path uses ``headline_*`` scales —
    slightly wider by default to reflect execution / human error on top of model risk.
    """
    defaults: dict[str, float] = {
        "mc_shared_log_sigma": 0.025,
        "het_noise_gamma": 0.45,
        "mc_runs_noise_scale": 0.15,
        "mc_wk_noise_scale": 0.15,
        "headline_runs_noise_scale": 0.18,
        "headline_wk_noise_scale": 0.20,
    }
    b = load_ensemble_bundle()
    if not b:
        return defaults
    out = {**defaults}
    out["mc_shared_log_sigma"] = float(b.get("mc_shared_log_sigma", 0.025))
    out["het_noise_gamma"] = float(b.get("het_noise_gamma", 0.45))
    out["mc_runs_noise_scale"] = float(
        b.get("mc_runs_noise_scale", b.get("player_runs_noise_scale", 0.15))
    )
    out["mc_wk_noise_scale"] = float(
        b.get("mc_wk_noise_scale", b.get("player_wk_noise_scale", 0.15))
    )
    out["headline_runs_noise_scale"] = float(
        b.get("headline_runs_noise_scale", out["headline_runs_noise_scale"])
    )
    out["headline_wk_noise_scale"] = float(
        b.get("headline_wk_noise_scale", out["headline_wk_noise_scale"])
    )
    return out
