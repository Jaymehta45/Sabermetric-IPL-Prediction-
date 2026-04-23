"""
Learned blend of ML vs simulation win probability, optional logit stacking of both,
and optional isotonic calibration.

Trained by iplpred/training/train_win_prob_ensemble.py → models/win_prob_ensemble.pkl.
If missing, falls back to fixed 0.6/0.4 blend with no extra calibration.

Isotonic calibration can push favorites to 99.9%+ while Monte Carlo often sits ~55–80%.
For **reported** headline probabilities we apply **upper-tail-only** moderation: the
sub‑93% band is left intact so matchups stay distinguishable; only mass above that is
soft-compressed so isotonic tails do not all print as the same ~94% (legacy cap bug).
Disable with ``IPLPRED_WIN_P_NO_MODERATE=1``. Set ``IPLPRED_WIN_P_SKIP_ISOTONIC=1`` to skip learned
isotonic and keep only stack/linear blend plus moderation (useful when ML vs MC diverge
and isotonic inflates the headline). Set ``IPLPRED_WIN_P_ML_SHARE`` to a float in ``[0,1]``
to **replace** the blended probability with ``share * ml_p + (1-share) * sim_p`` (skips
logit stack for that run; applied before isotonic/moderation).
With ``IPLPRED_WIN_P_SKIP_ISOTONIC=1``, if ``|ml_p - sim_p| >= 0.25`` and
``IPLPRED_WIN_P_HIGH_GAP_ML_SHARE`` is not ``0``, the stack output is replaced by a
high ML-weight blend (default share **0.97**) so bat-first Monte Carlo does not
single-handedly swamp a toss-up roster model. Optional bundle keys (also written by retraining):
``report_win_p_upper_thr``, ``report_win_p_upper_scale``, ``report_win_p_upper_cap``.
"""

from __future__ import annotations

import os
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


def _moderate_display_win_p_upper(p: float, b: dict[str, Any] | None) -> float:
    """
    Compress only the extreme upper tail (default: above ~93%).

    Probabilities at or below ``report_win_p_upper_thr`` pass through unchanged so
    plausible favorites (e.g. 58% vs 72%) stay distinct. A too-low cap (~0.94) with
    aggressive scale previously **flattened** almost every strong favorite to the same headline.
    """
    if os.environ.get("IPLPRED_WIN_P_NO_MODERATE", "").strip() == "1":
        return float(p)
    # Defaults: high thr + gentle scale + high cap so isotonic output keeps variance
    # (old defaults thr=0.90 cap=0.94 made ~every big favorite print as 94/6).
    thr = float(b.get("report_win_p_upper_thr", 0.93)) if b else 0.93
    scale = float(b.get("report_win_p_upper_scale", 0.78)) if b else 0.78
    cap = float(b.get("report_win_p_upper_cap", 0.992)) if b else 0.992
    thr = float(np.clip(thr, 0.55, 0.96))
    scale = float(np.clip(scale, 0.15, 1.0))
    cap = float(np.clip(cap, thr + 0.02, 0.995))
    p = float(p)
    if p <= thr:
        return p
    q = thr + (p - thr) * scale
    return float(min(q, cap))


def load_ensemble_bundle() -> dict[str, Any] | None:
    if not ENSEMBLE_BUNDLE_PATH.is_file():
        return None
    b = joblib.load(ENSEMBLE_BUNDLE_PATH)
    if not isinstance(b, dict):
        return None
    # Legacy bundles capped almost every strong favorite at 0.94 (flat 94/6 headlines).
    # Upgrade display-moderation keys unless the bundle already opts into newer values.
    cap = float(b.get("report_win_p_upper_cap", 1.0))
    if cap <= 0.95:
        b = {
            **b,
            "report_win_p_upper_thr": 0.93,
            "report_win_p_upper_scale": 0.78,
            "report_win_p_upper_cap": 0.992,
        }
    return b


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

    ml_share_override = os.environ.get("IPLPRED_WIN_P_ML_SHARE", "").strip()
    if ml_share_override and ml_p is not None and sim_p is not None:
        sh = float(np.clip(float(ml_share_override), 0.0, 1.0))
        raw = _clip_p(sh * float(ml_p) + (1.0 - sh) * float(sim_p))
        raw = _clip_p(raw)
        if os.environ.get("IPLPRED_WIN_P_SKIP_ISOTONIC", "").strip() == "1":
            out = _moderate_display_win_p_upper(raw, b)
            return _finalize_report_p(out)
        if not b:
            out = _moderate_display_win_p_upper(raw, None)
            return _finalize_report_p(out)
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
            out = _moderate_display_win_p_upper(out, b)
            return _finalize_report_p(out)
        iso = b.get("isotonic")
        if iso is None:
            out = _moderate_display_win_p_upper(raw, b)
            return _finalize_report_p(out)
        try:
            lo = float(logit(raw))
            out = float(iso.predict([lo])[0])
        except Exception:
            out = float(iso.predict([raw])[0])
        out = _moderate_display_win_p_upper(out, b)
        return _finalize_report_p(out)

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

    if (
        os.environ.get("IPLPRED_WIN_P_SKIP_ISOTONIC", "").strip() == "1"
        and ml_p is not None
        and sim_p is not None
    ):
        gap = abs(float(ml_p) - float(sim_p))
        if gap >= 0.25:
            hg = os.environ.get("IPLPRED_WIN_P_HIGH_GAP_ML_SHARE", "").strip()
            if hg != "0":
                sh = float(hg) if hg else 0.97
                sh = float(np.clip(sh, 0.5, 0.995))
                raw = _clip_p(sh * float(ml_p) + (1.0 - sh) * float(sim_p))

    if os.environ.get("IPLPRED_WIN_P_SKIP_ISOTONIC", "").strip() == "1":
        out = _moderate_display_win_p_upper(raw, b)
        return _finalize_report_p(out)

    if not b:
        out = _moderate_display_win_p_upper(raw, None)
        return _finalize_report_p(out)

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
        out = _moderate_display_win_p_upper(out, b)
        return _finalize_report_p(out)

    iso = b.get("isotonic")
    if iso is None:
        out = _moderate_display_win_p_upper(raw, b)
        return _finalize_report_p(out)
    try:
        lo = float(logit(raw))
        out = float(iso.predict([lo])[0])
    except Exception:
        out = float(iso.predict([raw])[0])
    out = _moderate_display_win_p_upper(out, b)
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
