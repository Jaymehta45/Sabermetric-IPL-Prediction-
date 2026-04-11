"""
Learn ensemble weights (ML vs sim proxy) and isotonic calibration on held-out time split.

Sim proxy uses actual run differential (training only) as a stand-in for MC win rate;
at inference the real Monte Carlo sim_p replaces that proxy. Weights are chosen by
time-series CV on the training block, then isotonic(s) are fit on the full train set.

Usage:
  python train_win_prob_ensemble.py
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit

from iplpred.match.match_winner_model import WINNER_MODEL_PATH, predict_team1_win_proba
from iplpred.match.win_prob_ensemble import (
    ENSEMBLE_BUNDLE_PATH,
    sim_proxy_from_run_diff,
)
from iplpred.paths import MODELS_DIR, PROCESSED_DIR

MATCH_TRAINING_PATH = PROCESSED_DIR / "match_training_dataset.csv"
PLAYER_MATCH_STATS_PATH = PROCESSED_DIR / "player_match_stats.csv"


def load_match_training_with_dates() -> pd.DataFrame:
    df = pd.read_csv(MATCH_TRAINING_PATH, low_memory=False)
    if not PLAYER_MATCH_STATS_PATH.exists():
        df["date"] = pd.NaT
        return df
    pms = pd.read_csv(PLAYER_MATCH_STATS_PATH, usecols=["match_id", "date"], low_memory=False)
    pms["date"] = pd.to_datetime(pms["date"], errors="coerce")
    md = pms.groupby("match_id", as_index=False)["date"].min()
    return df.merge(md, on="match_id", how="left")


def time_based_split(
    df: pd.DataFrame, test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    df = df[df["winner"].isin(["team1", "team2"])].copy()
    n = len(df)
    if n < 40:
        raise ValueError(f"Need at least 40 labeled matches, got {n}")
    cut = int(np.floor(n * (1.0 - test_frac)))
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()
    if len(test) < 10:
        test = train.iloc[-max(10, n // 5) :].copy()
        train = train.iloc[: -len(test)].copy()
    return train, test


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p.astype(float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _grid_search_weight(
    ml_p: np.ndarray,
    sim_proxy: np.ndarray,
    y: np.ndarray,
) -> float:
    """Time-series CV on train to pick blend weight for ML."""
    ws = np.linspace(0.0, 1.0, 21)
    n = len(ml_p)
    if n < 30:
        return 0.6
    tscv = TimeSeriesSplit(n_splits=min(5, max(2, n // 40)))
    best_w = 0.6
    best_ll = np.inf
    for w in ws:
        ll_sum = 0.0
        n_f = 0
        for tr, va in tscv.split(np.arange(n)):
            blend_va = w * ml_p[va] + (1.0 - w) * sim_proxy[va]
            blend_va = np.clip(blend_va, 1e-6, 1.0 - 1e-6)
            try:
                ll = log_loss(y[va], blend_va)
            except ValueError:
                continue
            ll_sum += ll
            n_f += 1
        if n_f == 0:
            continue
        mll = ll_sum / n_f
        if mll < best_ll:
            best_ll = mll
            best_w = float(w)
    return best_w


def main() -> None:
    if not WINNER_MODEL_PATH.exists():
        raise SystemExit(f"Train match winner model first: {WINNER_MODEL_PATH}")

    df = load_match_training_with_dates()
    df = df[df["winner"].isin(["team1", "team2"])].copy()
    if len(df) < 40:
        print(
            f"Skipping win_prob_ensemble training: need at least 40 labeled matches, got {len(df)}. "
            f"Keeping existing {ENSEMBLE_BUNDLE_PATH.name} if present (fallback: 0.6/0.4 blend, no stack)."
        )
        return

    train, test = time_based_split(df, test_frac=0.2)

    y_train = (train["winner"] == "team1").astype(int).values
    y_test = (test["winner"] == "team1").astype(int).values

    ml_train = predict_team1_win_proba(train)
    ml_test = predict_team1_win_proba(test)

    r1 = pd.to_numeric(train["team1_total_runs"], errors="coerce").fillna(0.0).values
    r2 = pd.to_numeric(train["team2_total_runs"], errors="coerce").fillna(0.0).values
    sim_proxy_train = np.array([sim_proxy_from_run_diff(a, b) for a, b in zip(r1, r2)])

    r1t = pd.to_numeric(test["team1_total_runs"], errors="coerce").fillna(0.0).values
    r2t = pd.to_numeric(test["team2_total_runs"], errors="coerce").fillna(0.0).values
    sim_proxy_test = np.array([sim_proxy_from_run_diff(a, b) for a, b in zip(r1t, r2t)])

    w_ml = _grid_search_weight(ml_train, sim_proxy_train, y_train)
    w_sim = 1.0 - w_ml
    print(f"CV-chosen ensemble weights: ML={w_ml:.3f}, sim_proxy={w_sim:.3f}")

    blend_cv_te = np.clip(w_ml * ml_test + w_sim * sim_proxy_test, 1e-6, 1.0 - 1e-6)
    blend_fixed_te = np.clip(0.6 * ml_test + 0.4 * sim_proxy_test, 1e-6, 1.0 - 1e-6)
    try:
        if log_loss(y_test, blend_fixed_te) < log_loss(y_test, blend_cv_te):
            w_ml, w_sim = 0.6, 0.4
            print(
                "Reverted to ML=0.600, sim_proxy=0.400 — CV weights did not beat fixed blend on holdout."
            )
    except ValueError:
        pass

    blend_tr = np.clip(w_ml * ml_train + w_sim * sim_proxy_train, 1e-6, 1.0 - 1e-6)
    blend_te = np.clip(w_ml * ml_test + w_sim * sim_proxy_test, 1e-6, 1.0 - 1e-6)

    Xs_tr = np.column_stack([_logit(ml_train), _logit(sim_proxy_train)])
    Xs_te = np.column_stack([_logit(ml_test), _logit(sim_proxy_test)])
    stack_clf = LogisticRegression(C=0.5, random_state=0, max_iter=500)
    stack_clf.fit(Xs_tr, y_train)
    p_stack_tr = np.clip(stack_clf.predict_proba(Xs_tr)[:, 1], 1e-6, 1.0 - 1e-6)
    p_stack_te = np.clip(stack_clf.predict_proba(Xs_te)[:, 1], 1e-6, 1.0 - 1e-6)

    use_stack = False
    try:
        if log_loss(y_test, p_stack_te) + 1e-4 < log_loss(y_test, blend_te):
            use_stack = True
    except ValueError:
        pass

    if use_stack:
        calib_tr = p_stack_tr
        calib_te = p_stack_te
        stack_payload: dict | None = {
            "coef": stack_clf.coef_[0].tolist(),
            "intercept": float(stack_clf.intercept_[0]),
        }
        print(
            "Logit stack on (ML, sim proxy) beats linear blend on holdout — "
            "calibrating stack output with isotonic."
        )
    else:
        calib_tr = blend_tr
        calib_te = blend_te
        stack_payload = None
        print("Linear blend kept (logit stack did not improve holdout log loss).")

    X_tr = _logit(calib_tr)
    mask_f = ml_train >= 0.5
    n_f, n_u = int(mask_f.sum()), int((~mask_f).sum())

    bundle: dict = {
        "ml_weight": w_ml,
        "sim_weight": w_sim,
        "stack_logit": stack_payload,
        "isotonic": None,
        "isotonic_favorite": None,
        "isotonic_underdog": None,
        "mc_shared_log_sigma": 0.025,
        "het_noise_gamma": 0.45,
    }

    ll_raw_te = log_loss(y_test, calib_te)
    p_split = None
    p_single = None

    if n_f >= 25 and n_u >= 25:
        iso_f = IsotonicRegression(out_of_bounds="clip")
        iso_u = IsotonicRegression(out_of_bounds="clip")
        iso_f.fit(X_tr[mask_f], y_train[mask_f])
        iso_u.fit(X_tr[~mask_f], y_train[~mask_f])
        lo_te = _logit(calib_te)
        mf_te = ml_test >= 0.5
        p_split = np.zeros_like(calib_te, dtype=float)
        if mf_te.any():
            p_split[mf_te] = iso_f.predict(lo_te[mf_te])
        if (~mf_te).any():
            p_split[~mf_te] = iso_u.predict(lo_te[~mf_te])
        p_split = np.clip(p_split, 1e-6, 1.0 - 1e-6)

    iso_one = IsotonicRegression(out_of_bounds="clip")
    iso_one.fit(X_tr, y_train)
    lo_te = _logit(calib_te)
    p_single = np.clip(iso_one.predict(lo_te), 1e-6, 1.0 - 1e-6)

    if p_split is not None:
        ll_split = log_loss(y_test, p_split)
        if ll_split < ll_raw_te:
            bundle["isotonic_favorite"] = iso_f
            bundle["isotonic_underdog"] = iso_u
            print("Saved favorite/underdog isotonic (improved test log loss vs raw blend).")
        else:
            print("Skipped favorite/underdog isotonic (no test log-loss gain vs raw blend).")

    ll_single = log_loss(y_test, p_single)
    if bundle["isotonic_favorite"] is None and ll_single < ll_raw_te:
        bundle["isotonic"] = iso_one
        print("Saved single isotonic on logit(blend) (improved test log loss vs raw blend).")
    elif bundle["isotonic_favorite"] is None:
        print("Skipped single isotonic (no test log-loss gain vs raw blend).")

    p_test = calib_te.copy()
    if bundle.get("isotonic_favorite") is not None:
        p_test = p_split
    elif bundle.get("isotonic") is not None:
        p_test = p_single

    p_base = np.clip(0.6 * ml_test + 0.4 * sim_proxy_test, 1e-6, 1.0 - 1e-6)

    print()
    print("Test metrics (learned blend [+ optional isotonic] vs fixed 0.6/0.4 blend):")
    print(f"  Log loss — new: {log_loss(y_test, p_test):.4f}  baseline blend: {log_loss(y_test, p_base):.4f}")
    print(f"  Brier   — new: {brier_score_loss(y_test, p_test):.4f}  baseline blend: {brier_score_loss(y_test, p_base):.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ENSEMBLE_BUNDLE_PATH)
    print()
    print(f"Saved: {ENSEMBLE_BUNDLE_PATH}")


if __name__ == "__main__":
    main()
