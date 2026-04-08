"""
Simulate match outcomes using trained player runs/wickets models.

Per-player run predictions are **scaled** after pitch/venue so each team's sum
matches the Ridge team-innings model when ``team_total_regressor.pkl`` exists,
otherwise a T20 anchor while preserving relative team strength. Monte Carlo
win draws use the same scales. This replaces the old 80–250 clip that often
floored both innings at 80.
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from iplpred.core.match_context import PitchMultipliers
from iplpred.core.venue_bowling_adjust import apply_venue_bowling_style_to_team_df
from iplpred.match.win_prob_ensemble import (
    apply_ensemble_and_calibrate,
    get_mc_noise_params,
    load_ensemble_bundle,
)
from iplpred.core.prediction_shrinkage import apply_confidence_shrinkage_raw
from iplpred.core.recent_form import player_confidence_map
from iplpred.core.squad_utils import validate_team_playing_xi
from iplpred.match.match_winner_model import learned_team1_win_proba_from_rosters
from iplpred.match.team_total_model import predict_team_totals_from_rosters
from iplpred.paths import MODELS_DIR
from iplpred.training.train_player_model import FEATURE_COLS, load_and_prepare

RUNS_MODEL_PATH = MODELS_DIR / "player_runs_model.pkl"
WICKETS_MODEL_PATH = MODELS_DIR / "player_wickets_model.pkl"

MAX_SQUAD = 12
# Legacy clip for one-off tools; headline innings totals use calibrated_team_runs instead.
MIN_TEAM_RUNS = 80.0
MAX_TEAM_RUNS = 250.0

# T20 IPL-ish innings anchor: sum of per-player RF means is often ~40–90 — scale to this band.
T20_INNINGS_ANCHOR = 165.0
MIN_INNINGS_DISPLAY = 110.0
MAX_INNINGS_DISPLAY = 255.0
MAX_PLAYER_RUN_SCALE = 6.0

MATCHUP_NEUTRAL_SR = 100.0
MATCHUP_SCALE_MIN = 0.82
MATCHUP_SCALE_MAX = 1.18


def load_models():
    if not RUNS_MODEL_PATH.exists() or not WICKETS_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Need {RUNS_MODEL_PATH} and {WICKETS_MODEL_PATH}. Run train_player_model.py first."
        )
    runs_model = joblib.load(RUNS_MODEL_PATH)
    wk_model = joblib.load(WICKETS_MODEL_PATH)
    return runs_model, wk_model


def get_latest_player_rows(
    player_ids: list[str],
    *,
    venue: str | None = None,
    team1_name: str | None = None,
    team2_name: str | None = None,
    team1: list[str] | None = None,
    team2: list[str] | None = None,
) -> pd.DataFrame:
    """
    Latest feature row per player (same prep as training).

    When ``venue`` and/or both franchise names are set, uses **fixture-aware**
    rows: prefer history at the upcoming ground, recompute opponent strength from
    rolling team top-5 metrics, and refresh batting/bowling matchup SR vs this
    opponent from past matches (see ``inference_feature_rows``).
    """
    from iplpred.match.inference_feature_rows import (
        build_fixture_player_frame,
        should_use_fixture_context,
    )

    ids = [str(p).strip() for p in player_ids]
    id_set = set(ids)
    if (
        should_use_fixture_context(venue, team1_name, team2_name)
        and team1 is not None
        and team2 is not None
    ):
        return build_fixture_player_frame(
            ids,
            venue=venue,
            team1_name=team1_name,
            team2_name=team2_name,
            team1=team1,
            team2=team2,
        )

    df = load_and_prepare()
    df = df[df["player_id"].astype(str).str.strip().isin(id_set)]
    found = set(df["player_id"].astype(str).str.strip().unique())
    missing = id_set - found
    if missing:
        raise ValueError(f"Players not found in prepared data: {sorted(missing)}")
    latest = (
        df.sort_values(["player_id", "date", "match_id"])
        .groupby("player_id", as_index=False)
        .last()
    )
    return latest


def playing_probs_from_history(df_full: pd.DataFrame, player_ids: list[str]) -> dict[str, float]:
    """
    playing_prob = min(1.0, matches_played_last_5 / 5), where matches_played_last_5
    is the number of distinct matches in that player's last five appearances (by date).
    """
    out: dict[str, float] = {}
    for p in player_ids:
        key = str(p).strip()
        sub = df_full[df_full["player_id"].astype(str).str.strip() == key]
        if len(sub) == 0:
            out[key] = 0.0
            continue
        sub = sub.sort_values(["date", "match_id"])
        if "match_id" in sub.columns:
            last5 = sub.tail(5)
            played = int(last5["match_id"].nunique())
        else:
            played = min(5, len(sub))
        out[key] = min(1.0, played / 5.0)
    return out


def _order_team_rows(sub: pd.DataFrame, team: list[str]) -> pd.DataFrame:
    order = {str(p).strip(): i for i, p in enumerate(team)}
    sub = sub.copy()
    key = sub["player_id"].astype(str).str.strip()
    sub["_ord"] = key.map(order)
    return sub.sort_values("_ord").drop(columns=["_ord"])


def apply_matchup_scaling(sub: pd.DataFrame) -> pd.DataFrame:
    """
    **Deprecated (unused):** ``batting_matchup_sr`` / ``bowling_matchup_sr`` are already in
    ``FEATURE_COLS`` for the RandomForest. Multiplying predictions by the same SRs again
    **double-counted** matchup effects and skewed player outputs. Kept for reference only.
    """
    sub = sub.copy()
    if "batting_matchup_sr" not in sub.columns or "bowling_matchup_sr" not in sub.columns:
        return sub
    n = MATCHUP_NEUTRAL_SR
    bt = pd.to_numeric(sub["batting_matchup_sr"], errors="coerce").fillna(n)
    bw = pd.to_numeric(sub["bowling_matchup_sr"], errors="coerce").fillna(n)
    r_mult = (bt / n).clip(MATCHUP_SCALE_MIN, MATCHUP_SCALE_MAX)
    bw_safe = bw.clip(40.0, 220.0)
    w_mult = (n / bw_safe).clip(MATCHUP_SCALE_MIN, MATCHUP_SCALE_MAX)
    sub["pred_runs_raw"] = sub["pred_runs_raw"].astype(float) * r_mult
    sub["pred_wk_raw"] = sub["pred_wk_raw"].astype(float) * w_mult
    return sub


def _feature_matrix_for_rf(sub: pd.DataFrame) -> np.ndarray:
    """Strictly aligned, finite feature matrix for the trained RF (same order as training)."""
    mat = sub[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    mat = mat.replace([np.inf, -np.inf], np.nan)
    if mat.isna().any().any():
        mat = mat.fillna(0.0)
    return mat.values.astype(np.float64)


def predict_team_raw(
    latest: pd.DataFrame,
    team: list[str],
    runs_model,
    wk_model,
) -> pd.DataFrame:
    """RF runs/wickets from ``FEATURE_COLS`` (matchup SRs are inputs only — not re-applied to outputs)."""
    team = [str(p).strip() for p in team]
    if len(team) != len(set(team)):
        raise ValueError("Duplicate player in team list.")
    if len(team) > MAX_SQUAD:
        raise ValueError(f"Each team may have at most {MAX_SQUAD} players.")
    sub = latest[latest["player_id"].astype(str).str.strip().isin(team)].copy()
    if len(sub) != len(team):
        raise ValueError("Missing player in latest features vs team list.")
    sub = _order_team_rows(sub, team)
    X = _feature_matrix_for_rf(sub)
    sub["pred_runs_raw"] = runs_model.predict(X)
    sub["pred_wk_raw"] = wk_model.predict(X)
    return sub


def clip_team_total(total: float) -> float:
    return float(np.clip(total, MIN_TEAM_RUNS, MAX_TEAM_RUNS))


def _lift_low_totals_for_batting_venue(
    t1: float,
    t2: float,
    venue: str | None,
) -> tuple[float, float]:
    """
    Ridge ML heads can sit ~95-115; after pitch mult + clip both innings may floor at
    ``MIN_INNINGS_DISPLAY`` (110). ``_blend_high_scoring_pitch_totals`` only runs when
    pitch run mults average ≥1.14, so *modest* belter reports never fire.

    At batting-friendly venues (reuse spin-harshness as small-ground proxy), scale both
    targets toward a T20 par so Chinnaswamy-style grounds do not show ~110 vs ~110.

    **Second branch:** ML/Ridge often sit ~160–195 on belters while realised par is 200+;
    we still lift when the innings mean is in [152, 205) so pre-match totals are not
    stuck ~170 at high-scoring grounds.
    """
    from iplpred.core.venue_bowling_adjust import venue_spin_harshness

    h = venue_spin_harshness(venue)
    if h < 0.35:
        return t1, t2
    mid = (float(t1) + float(t2)) / 2.0
    if mid < 152.0:
        # ~158–194+ depending on harshness (Chinnaswamy tier ~0.9+)
        par = 158.0 + h * 28.0 + min(14.0, max(0.0, h - 0.35) * 20.0)
        scale = float(par / max(mid, 72.0))
        scale = float(np.clip(scale, 1.0, 1.55))
        t1n = float(np.clip(t1 * scale, MIN_INNINGS_DISPLAY, MAX_INNINGS_DISPLAY))
        t2n = float(np.clip(t2 * scale, MIN_INNINGS_DISPLAY, MAX_INNINGS_DISPLAY))
        return t1n, t2n
    if h >= 0.48 and 152.0 <= mid < 205.0:
        par_mid = 160.0 + h * 48.0
        if mid < par_mid - 6.0:
            scale = float(np.clip(par_mid / max(mid, 80.0), 1.0, 1.30))
            t1n = float(np.clip(t1 * scale, MIN_INNINGS_DISPLAY, MAX_INNINGS_DISPLAY))
            t2n = float(np.clip(t2 * scale, MIN_INNINGS_DISPLAY, MAX_INNINGS_DISPLAY))
            return t1n, t2n
    return t1, t2


def _blend_high_scoring_pitch_totals(
    t1: float,
    t2: float,
    pm: PitchMultipliers,
) -> tuple[float, float]:
    """
    When pitch multipliers reflect a belter / high-par read (strong run mults),
    scale team innings targets toward ~185–205 so Ridge heads (~110–130) are not
    stuck far below broadcast par. Bounded so neutral pitch (mult ~1) is unchanged.
    """
    hi = (pm.first_innings_runs + pm.second_innings_runs) / 2.0
    if hi < 1.14:
        return t1, t2
    m = (t1 + t2) / 2.0
    # Stronger pitch reads (hi≈1.35+) already imply ~195–215 par; do not stop blending at 178.
    m_done = 185.0 if hi >= 1.35 else 178.0
    if m >= m_done:
        return t1, t2
    excess = hi - 1.14
    base_a = 196.0 if hi >= 1.35 else 188.0
    aspirational = base_a + min(18.0 if hi >= 1.35 else 14.0, excess * 35.0)
    blend_cap = 0.82 if hi >= 1.35 else 0.72
    blend = min(blend_cap, 0.28 + excess * 2.5)
    scale = 1.0 + blend * (aspirational / max(m, 72.0) - 1.0)
    scale_cap = 1.65 if hi >= 1.35 else 1.55
    scale = float(np.clip(scale, 1.0, scale_cap))
    t1n = float(np.clip(t1 * scale, MIN_INNINGS_DISPLAY, MAX_INNINGS_DISPLAY))
    t2n = float(np.clip(t2 * scale, MIN_INNINGS_DISPLAY, MAX_INNINGS_DISPLAY))
    return t1n, t2n


def _calibrated_innings_targets(
    raw1: float,
    raw2: float,
    pm: PitchMultipliers,
    ml_team_totals: tuple[float, float] | None,
    venue: str | None = None,
) -> tuple[float, float, str]:
    """
    Target expected innings totals before per-player scaling.

    Prefer Ridge team-total heads (× innings pitch) when the bundle exists; otherwise
    lift raw sums of player predictions toward ~T20_INNINGS_ANCHOR while preserving
    relative strength between teams.

    ``venue`` is used to lift low Ridge targets at small / batting-friendly grounds
    (see ``_lift_low_totals_for_batting_venue``).
    """
    if ml_team_totals is not None:
        m1, m2 = float(ml_team_totals[0]), float(ml_team_totals[1])
        t1 = float(np.clip(m1 * pm.first_innings_runs, MIN_INNINGS_DISPLAY, MAX_INNINGS_DISPLAY))
        t2 = float(np.clip(m2 * pm.second_innings_runs, MIN_INNINGS_DISPLAY, MAX_INNINGS_DISPLAY))
        t1, t2 = _blend_high_scoring_pitch_totals(t1, t2, pm)
        t1, t2 = _lift_low_totals_for_batting_venue(t1, t2, venue)
        return t1, t2, "ml_team_total"
    mid = (float(raw1) + float(raw2)) / 2.0
    k = T20_INNINGS_ANCHOR / max(mid, 45.0)
    t1 = float(np.clip(k * float(raw1), MIN_INNINGS_DISPLAY, MAX_INNINGS_DISPLAY))
    t2 = float(np.clip(k * float(raw2), MIN_INNINGS_DISPLAY, MAX_INNINGS_DISPLAY))
    t1, t2 = _blend_high_scoring_pitch_totals(t1, t2, pm)
    t1, t2 = _lift_low_totals_for_batting_venue(t1, t2, venue)
    return t1, t2, "raw_sum_anchor"


def _team_calibration_scales(
    raw1: float,
    raw2: float,
    tgt1: float,
    tgt2: float,
    *,
    cap_large_scales: bool = True,
) -> tuple[float, float]:
    """When ``cap_large_scales`` is False (ML team-total path), do not cap — raw sums can be tiny."""
    s1 = float(tgt1) / max(float(raw1), 1e-9)
    s2 = float(tgt2) / max(float(raw2), 1e-9)
    if cap_large_scales:
        s1 = min(s1, MAX_PLAYER_RUN_SCALE)
        s2 = min(s2, MAX_PLAYER_RUN_SCALE)
    return (float(max(s1, 0.0)), float(max(s2, 0.0)))


def _scale_team_run_df(active: pd.DataFrame, scale: float) -> pd.DataFrame:
    """Multiply per-player run columns so team sum matches calibrated innings total."""
    out = active.copy()
    sc = float(scale)
    for col in ("predicted_runs", "pred_runs_raw"):
        if col in out.columns:
            out[col] = out[col].astype(float) * sc
    re = out["role_encoded"] if "role_encoded" in out.columns else None
    if "impact_base" in out.columns:
        out["impact_base"] = _impact_base(out["predicted_runs"], out["predicted_wickets"], re)
    return out


def _noise_on_preds(
    runs: np.ndarray,
    wickets: np.ndarray,
    rng: np.random.Generator,
    std_runs_arr: np.ndarray | None = None,
    std_wk_arr: np.ndarray | None = None,
    *,
    base_scale_r: float = 0.15,
    base_scale_w: float = 0.15,
    het_gamma: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Heteroskedastic noise: scale ∝ predicted mean × (1 + γ·tanh(std/scale)).
    Uses rolling std_runs/std_wickets from features when present.
    """
    r = np.maximum(runs.astype(float), 0.0)
    w = np.maximum(wickets.astype(float), 0.0)
    med_r = (
        float(np.nanmedian(std_runs_arr))
        if std_runs_arr is not None and np.size(std_runs_arr)
        else 15.0
    )
    med_w = (
        float(np.nanmedian(std_wk_arr))
        if std_wk_arr is not None and np.size(std_wk_arr)
        else 0.8
    )
    if std_runs_arr is not None and len(std_runs_arr) == len(r):
        sr = np.nan_to_num(std_runs_arr.astype(float), nan=med_r)
        sr = np.maximum(sr, 0.0)
        mult_r = 1.0 + het_gamma * np.tanh(sr / max(med_r, 1e-6))
    else:
        mult_r = np.ones_like(r)
    if std_wk_arr is not None and len(std_wk_arr) == len(w):
        sw = np.nan_to_num(std_wk_arr.astype(float), nan=med_w)
        sw = np.maximum(sw, 0.0)
        mult_w = 1.0 + het_gamma * np.tanh(sw / max(med_w, 1e-6))
    else:
        mult_w = np.ones_like(w)
    std_r = base_scale_r * np.maximum(r, 1e-9) * mult_r
    std_w = base_scale_w * np.maximum(w, 1e-9) * mult_w
    runs_n = r + rng.normal(0.0, std_r, size=r.shape)
    wk_n = w + rng.normal(0.0, std_w, size=w.shape)
    return np.maximum(runs_n, 0.0), np.maximum(wk_n, 0.0)


def _impact_base(
    runs: pd.Series,
    wk: pd.Series,
    role_enc: pd.Series | None = None,
) -> pd.Series:
    """
    IPL impact substitute: who to trim from the 12-player pool.

    Bowlers (role_encoded == 1) get extra weight on predicted wickets so pure
    bowlers are not dropped solely because predicted batting runs are near zero.
    """
    base = runs + 25.0 * wk
    if role_enc is not None and len(role_enc) == len(runs):
        re = pd.to_numeric(role_enc, errors="coerce").fillna(2.0)
        is_bowl = re.eq(1.0)
        base = base + np.where(is_bowl, 18.0 * wk.astype(float), 0.0)
    return pd.Series(base, index=runs.index)


def apply_playing_prob_and_drop(
    sub: pd.DataFrame,
    playing_prob: dict[str, float],
    drop_lowest: bool = True,
) -> tuple[pd.DataFrame, str | None]:
    """predicted_runs/wickets *= playing_prob; optionally drop lowest impact (runs + 25*wickets)."""
    sub = sub.copy()
    keys = sub["player_id"].astype(str).str.strip()
    sub["playing_prob"] = keys.map(lambda k: playing_prob.get(k, 1.0))
    sub["predicted_runs"] = sub["pred_runs_raw"] * sub["playing_prob"]
    sub["predicted_wickets"] = sub["pred_wk_raw"] * sub["playing_prob"]
    re = sub["role_encoded"] if "role_encoded" in sub.columns else None
    sub["impact_base"] = _impact_base(sub["predicted_runs"], sub["predicted_wickets"], re)

    if len(sub) <= 1:
        sub["dropped"] = False
        return sub, None
    if not drop_lowest:
        sub["dropped"] = False
        return sub, None
    # IPL-style: only trim one player when the pool is 12 (XI + impact substitute).
    if len(sub) < MAX_SQUAD:
        sub["dropped"] = False
        return sub, None

    drop_idx = sub["impact_base"].idxmin()
    dropped_id = str(sub.loc[drop_idx, "player_id"]).strip()
    sub["dropped"] = False
    sub.loc[drop_idx, "dropped"] = True
    active = sub[~sub["dropped"]].copy()
    return active, dropped_id


def pom_score_row(
    runs: float,
    wk: float,
    role_enc: float | None = None,
) -> float:
    """
    Composite POTM-style score (not official award). Uses ``role_encoded`` when present:
    batters weight runs more, bowlers weight wickets more, allrounders stay near the legacy mix.
    """
    re = 2.0
    if role_enc is not None:
        try:
            re = float(role_enc)
            if np.isnan(re):
                re = 2.0
        except (TypeError, ValueError):
            re = 2.0
    if re < 0.75:
        wr, ww = 1.1, 18.0
    elif re > 1.25:
        wr, ww = 0.45, 28.0
    else:
        wr, ww = 1.0, 25.0
    s = wr * float(runs) + ww * float(wk)
    if runs > 50:
        s += 10.0
    if wk >= 3:
        s += 15.0
    return float(s)


def _role_encoded_series(combined: pd.DataFrame) -> pd.Series:
    if "role_encoded" not in combined.columns:
        return pd.Series(2.0, index=combined.index)
    return pd.to_numeric(combined["role_encoded"], errors="coerce").fillna(2.0)


def _batting_eligible_mask(combined: pd.DataFrame) -> pd.Series:
    """Exclude pure bowlers (1) from batting leaderboards when roles exist."""
    re = _role_encoded_series(combined)
    return re != 1.0


def _bowling_eligible_mask(combined: pd.DataFrame) -> pd.Series:
    """Prefer bowlers + allrounders for wicket leaderboards."""
    re = _role_encoded_series(combined)
    m = re.isin([1.0, 2.0])
    if bool(m.any()):
        return m
    return pd.Series(True, index=combined.index)


def role_aware_top5_batters(combined: pd.DataFrame, cols5: list[str]) -> pd.DataFrame:
    pool = combined[_batting_eligible_mask(combined)] if "role_encoded" in combined.columns else combined
    if len(pool) >= 5:
        return pool.nlargest(5, "predicted_runs")[cols5]
    return combined.nlargest(5, "predicted_runs")[cols5]


def role_aware_top3_bowlers(combined: pd.DataFrame, cols3: list[str]) -> pd.DataFrame:
    pool = combined[_bowling_eligible_mask(combined)] if "role_encoded" in combined.columns else combined
    if len(pool) >= 3:
        return pool.nlargest(3, "predicted_wickets")[cols3]
    return combined.nlargest(3, "predicted_wickets")[cols3]


def pipeline_from_raw(
    sub: pd.DataFrame,
    playing_prob: dict[str, float],
    rng: np.random.Generator | None,
    add_noise: bool,
    drop_lowest: bool = True,
) -> tuple[pd.DataFrame, str | None]:
    """Raw preds -> optional MC noise on preds -> * playing_prob -> optional drop lowest impact."""
    sub = sub.copy()
    pr = sub["pred_runs_raw"].values.astype(float)
    pw = sub["pred_wk_raw"].values.astype(float)
    if add_noise and rng is not None:
        np_het = get_mc_noise_params()
        sr = sub["std_runs"].values if "std_runs" in sub.columns else None
        sw = sub["std_wickets"].values if "std_wickets" in sub.columns else None
        pr, pw = _noise_on_preds(
            pr,
            pw,
            rng,
            std_runs_arr=sr,
            std_wk_arr=sw,
            het_gamma=float(np_het.get("het_noise_gamma", 0.45)),
        )
    sub["pred_runs_raw"] = pr
    sub["pred_wk_raw"] = pw
    return apply_playing_prob_and_drop(sub, playing_prob, drop_lowest=drop_lowest)


def team_total_runs(active: pd.DataFrame) -> float:
    return float(active["predicted_runs"].sum())


def apply_pitch_to_team_df(
    active: pd.DataFrame,
    runs_mult: float,
    wk_mult: float,
) -> pd.DataFrame:
    """Scale predicted runs/wickets (and raw preds for consistency)."""
    out = active.copy()
    for col in ("predicted_runs", "pred_runs_raw"):
        if col in out.columns:
            out[col] = out[col].astype(float) * runs_mult
    for col in ("predicted_wickets", "pred_wk_raw"):
        if col in out.columns:
            out[col] = out[col].astype(float) * wk_mult
    if "impact_base" in out.columns:
        re = out["role_encoded"] if "role_encoded" in out.columns else None
        out["impact_base"] = _impact_base(out["predicted_runs"], out["predicted_wickets"], re)
    return out


def _maybe_shrink_raw_predictions(
    sub: pd.DataFrame,
    team_player_ids: list[str],
    confidence: dict[str, float],
    *,
    enabled: bool,
    team_mean_weight: float,
) -> pd.DataFrame:
    if not enabled:
        return sub
    c_map = {str(k).strip(): float(confidence.get(str(k).strip(), 0.4)) for k in team_player_ids}
    return apply_confidence_shrinkage_raw(sub, c_map, team_mean_weight=team_mean_weight)


def pick_winner(raw1: float, raw2: float) -> str:
    """Winner from team run totals (calibrated sums of per-player predictions)."""
    if raw1 > raw2:
        return "team1"
    if raw2 > raw1:
        return "team2"
    return "tie"


def run_simulation(
    team1: list[str],
    team2: list[str],
    n_monte_carlo: int = 100,
    rng: np.random.Generator | None = None,
    team1_name: str | None = None,
    team2_name: str | None = None,
    validate_squad: bool = True,
    drop_lowest_impact: bool = True,
    pitch: PitchMultipliers | None = None,
    use_recent_form_shrinkage: bool = True,
    shrink_team_prior_weight: float = 0.45,
    venue: str | None = None,
    match_date: str | None = None,
) -> dict:
    if validate_squad and team1_name and team2_name:
        validate_team_playing_xi(team1_name, team1)
        validate_team_playing_xi(team2_name, team2)

    runs_model, wk_model = load_models()
    all_ids = list(dict.fromkeys([str(p).strip() for p in team1 + team2]))
    df_full = load_and_prepare()
    latest = get_latest_player_rows(
        all_ids,
        venue=venue,
        team1_name=team1_name,
        team2_name=team2_name,
        team1=team1,
        team2=team2,
    )
    playing_prob = playing_probs_from_history(df_full, all_ids)
    conf_map = player_confidence_map(all_ids)

    t1_raw = predict_team_raw(latest, team1, runs_model, wk_model)
    t2_raw = predict_team_raw(latest, team2, runs_model, wk_model)
    t1_raw = _maybe_shrink_raw_predictions(
        t1_raw,
        team1,
        conf_map,
        enabled=use_recent_form_shrinkage,
        team_mean_weight=shrink_team_prior_weight,
    )
    t2_raw = _maybe_shrink_raw_predictions(
        t2_raw,
        team2,
        conf_map,
        enabled=use_recent_form_shrinkage,
        team_mean_weight=shrink_team_prior_weight,
    )

    t1_act, impact_drop_t1 = apply_playing_prob_and_drop(
        t1_raw, playing_prob, drop_lowest=drop_lowest_impact
    )
    t2_act, impact_drop_t2 = apply_playing_prob_and_drop(
        t2_raw, playing_prob, drop_lowest=drop_lowest_impact
    )

    pm = pitch or PitchMultipliers()
    t1_act = apply_pitch_to_team_df(
        t1_act, pm.first_innings_runs, pm.first_innings_wickets
    )
    t2_act = apply_pitch_to_team_df(
        t2_act, pm.second_innings_runs, pm.second_innings_wickets
    )
    t1_act, venue_harsh = apply_venue_bowling_style_to_team_df(t1_act, venue)
    t2_act, _ = apply_venue_bowling_style_to_team_df(t2_act, venue)

    raw1 = team_total_runs(t1_act)
    raw2 = team_total_runs(t2_act)

    eff_t1 = t1_act["player_id"].astype(str).str.strip().tolist()
    eff_t2 = t2_act["player_id"].astype(str).str.strip().tolist()
    ml_team_totals = predict_team_totals_from_rosters(
        latest,
        eff_t1,
        eff_t2,
        team1_name=team1_name,
        team2_name=team2_name,
        match_date=match_date,
    )
    tgt1, tgt2, calib_method = _calibrated_innings_targets(
        raw1, raw2, pm, ml_team_totals, venue
    )
    s1, s2 = _team_calibration_scales(
        raw1, raw2, tgt1, tgt2, cap_large_scales=(ml_team_totals is None)
    )
    t1_act = _scale_team_run_df(t1_act, s1)
    t2_act = _scale_team_run_df(t2_act, s2)
    total1 = team_total_runs(t1_act)
    total2 = team_total_runs(t2_act)

    winner = pick_winner(total1, total2)

    combined = pd.concat(
        [
            t1_act.assign(side="team1"),
            t2_act.assign(side="team2"),
        ],
        ignore_index=True,
    )
    pr = combined["predicted_runs"].astype(float).values
    pw = combined["predicted_wickets"].astype(float).values
    rr = _role_encoded_series(combined).values
    combined["pom_score"] = [pom_score_row(pr[i], pw[i], rr[i]) for i in range(len(combined))]

    cols5 = ["player_id", "side", "predicted_runs"]
    if "strike_rate" in combined.columns:
        cols5.append("strike_rate")
    top5_bat = role_aware_top5_batters(combined, cols5)
    cols3 = ["player_id", "side", "predicted_wickets"]
    if "economy" in combined.columns:
        cols3.append("economy")
    top3_bowl = role_aware_top3_bowlers(combined, cols3)
    pom = combined.loc[combined["pom_score"].idxmax()]

    win_prob_team1 = None
    if n_monte_carlo > 0:
        if rng is None:
            rng = np.random.default_rng(42)
        wins1 = 0
        for _ in range(n_monte_carlo):
            p1 = predict_team_raw(latest, team1, runs_model, wk_model)
            p2 = predict_team_raw(latest, team2, runs_model, wk_model)
            p1 = _maybe_shrink_raw_predictions(
                p1,
                team1,
                conf_map,
                enabled=use_recent_form_shrinkage,
                team_mean_weight=shrink_team_prior_weight,
            )
            p2 = _maybe_shrink_raw_predictions(
                p2,
                team2,
                conf_map,
                enabled=use_recent_form_shrinkage,
                team_mean_weight=shrink_team_prior_weight,
            )
            t1_mc, _ = pipeline_from_raw(
                p1,
                playing_prob,
                rng,
                add_noise=True,
                drop_lowest=drop_lowest_impact,
            )
            t2_mc, _ = pipeline_from_raw(
                p2,
                playing_prob,
                rng,
                add_noise=True,
                drop_lowest=drop_lowest_impact,
            )
            t1_mc = apply_pitch_to_team_df(
                t1_mc, pm.first_innings_runs, pm.first_innings_wickets
            )
            t2_mc = apply_pitch_to_team_df(
                t2_mc, pm.second_innings_runs, pm.second_innings_wickets
            )
            t1_mc, _ = apply_venue_bowling_style_to_team_df(t1_mc, venue)
            t2_mc, _ = apply_venue_bowling_style_to_team_df(t2_mc, venue)
            r1 = team_total_runs(t1_mc) * s1
            r2 = team_total_runs(t2_mc) * s2
            sig = float(get_mc_noise_params().get("mc_shared_log_sigma", 0.025))
            if sig > 0:
                z = rng.normal(0.0, sig)
                r1 = float(r1 * np.exp(z))
                r2 = float(r2 * np.exp(z))
            if pick_winner(r1, r2) == "team1":
                wins1 += 1
        win_prob_team1 = wins1 / n_monte_carlo

    leader_p_team1 = learned_team1_win_proba_from_rosters(
        latest,
        eff_t1,
        eff_t2,
        team1_name=team1_name,
        team2_name=team2_name,
        match_date=match_date,
    )
    ensemble_p_team1 = apply_ensemble_and_calibrate(leader_p_team1, win_prob_team1)

    # Expected wickets taken by the fielding XI per innings (sum of player predicted_wickets).
    # Innings 1: team2 bowls to team1 → use team2's bowling rows. Innings 2: team1 bowls.
    # Capped at 10 (max dismissals in an innings). Raw sums are kept for transparency.
    w_first_raw = float(t2_act["predicted_wickets"].sum())
    w_second_raw = float(t1_act["predicted_wickets"].sum())

    return {
        "team1_total_runs": total1,
        "team2_total_runs": total2,
        "team1_raw_runs": raw1,
        "team2_raw_runs": raw2,
        "winner": winner,
        "top5_batters": top5_bat,
        "top3_bowlers": top3_bowl,
        "player_of_match": pom,
        "win_probability_team1": win_prob_team1,
        "n_monte_carlo": n_monte_carlo,
        "t1_df": t1_act,
        "t2_df": t2_act,
        "squad_validated": bool(
            validate_squad and team1_name and team2_name
        ),
        "leader_model_p_team1": leader_p_team1,
        "ensemble_p_team1": ensemble_p_team1,
        "drop_lowest_impact": drop_lowest_impact,
        "impact_dropped_team1": impact_drop_t1,
        "impact_dropped_team2": impact_drop_t2,
        "pitch_multipliers": pm,
        "venue_spin_harshness": float(venue_harsh),
        "use_recent_form_shrinkage": use_recent_form_shrinkage,
        "player_confidence": conf_map,
        "ml_team1_total_runs": ml_team_totals[0] if ml_team_totals else None,
        "ml_team2_total_runs": ml_team_totals[1] if ml_team_totals else None,
        "team_run_calibration_method": calib_method,
        "team1_calibration_scale": s1,
        "team2_calibration_scale": s2,
        "team1_calibration_target": tgt1,
        "team2_calibration_target": tgt2,
        "pred_wickets_first_innings": min(10.0, w_first_raw),
        "pred_wickets_second_innings": min(10.0, w_second_raw),
        "pred_wickets_first_innings_raw_sum": w_first_raw,
        "pred_wickets_second_innings_raw_sum": w_second_raw,
    }


def six_match_outcomes(
    sim_out: dict,
    team1_name: str | None = None,
    team2_name: str | None = None,
) -> dict:
    """
    Six pre-match style predictions from a simulation dict:
    winning team, best batsman, best bowler, top 5 batsmen, top 3 bowlers, player of the match.

    **Winner** is aligned with **Monte Carlo win rate** when ``n_monte_carlo > 0`` and
    ``win_probability_team1`` is set — otherwise falls back to **deterministic** totals
    (``winner`` / ``pick_winner`` on **calibrated** team runs).
    """
    p1 = sim_out.get("win_probability_team1")
    n_mc = int(sim_out.get("n_monte_carlo") or 0)
    win_basis = "deterministic_totals"

    if p1 is not None and n_mc > 0:
        p1 = float(p1)
        win_basis = "monte_carlo"
        if p1 > 0.5:
            winning_team = team1_name or "team1"
        elif p1 < 0.5:
            winning_team = team2_name or "team2"
        else:
            winning_team = "tie"
    else:
        w = sim_out["winner"]
        if w == "team1":
            winning_team = team1_name or "team1"
        elif w == "team2":
            winning_team = team2_name or "team2"
        else:
            winning_team = "tie"
        p1 = None

    p_winner: float | None = None
    if p1 is not None:
        t1n = (team1_name or "").strip()
        t2n = (team2_name or "").strip()
        if winning_team == "tie":
            p_winner = None
        elif winning_team == t1n or winning_team == "team1":
            p_winner = p1
        elif winning_team == t2n or winning_team == "team2":
            p_winner = 1.0 - p1
        else:
            p_winner = max(p1, 1.0 - p1)

    top5 = sim_out["top5_batters"]
    top3 = sim_out["top3_bowlers"]
    best_bat = top5.iloc[0] if len(top5) else None
    best_bowl = top3.iloc[0] if len(top3) else None

    out = {
        "winning_team": winning_team,
        "best_batsman": best_bat,
        "best_bowler": best_bowl,
        "top5_batsmen": top5,
        "top3_bowlers": top3,
        "player_of_the_match": sim_out["player_of_match"],
        "p_team1_win": p1,
        "p_team2_win": (1.0 - p1) if p1 is not None else None,
        "p_predicted_winner": p_winner,
        "win_probability_basis": win_basis,
    }
    return out


def confidence_level(favorite_prob: float) -> str:
    """favorite_prob = max(p, 1-p) for team1 win probability p."""
    if favorite_prob > 0.65:
        return "HIGH"
    if favorite_prob >= 0.55:
        return "MEDIUM"
    return "LOW"


def print_report(out: dict) -> None:
    print(
        f"Predicted team scores (normalized to [{MIN_TEAM_RUNS:.0f}, {MAX_TEAM_RUNS:.0f}]): "
        f"team1={out['team1_total_runs']:.2f}, team2={out['team2_total_runs']:.2f}"
    )
    print(
        f"Winner: {out['winner']} (unclipped team runs: team1={out['team1_raw_runs']:.2f}, "
        f"team2={out['team2_raw_runs']:.2f})"
    )
    vh = out.get("venue_spin_harshness")
    if vh is not None and float(vh) > 0:
        print(
            f"Venue spin harshness: {float(vh):.2f} "
            "(spinners' predicted wickets scaled down; pace slightly up)"
        )
    lm = out.get("leader_model_p_team1")
    if lm is not None:
        print(
            f"Learned team model P(team1 wins): {lm:.2%} "
            f"(trained on official IPL results when available; see train_match_winner_model.py)"
        )
    wp = out.get("win_probability_team1")
    if wp is not None:
        print(f"Monte Carlo P(team1 wins): {wp:.2%} (headline; use this with predicted winner above)")
    ens = out.get("ensemble_p_team1")
    if ens is not None:
        b = load_ensemble_bundle()
        w_ml = float(b.get("ml_weight", 0.6)) if b else 0.6
        w_sim = float(b.get("sim_weight", 0.4)) if b else 0.4
        s = w_ml + w_sim
        if s > 0:
            w_ml, w_sim = w_ml / s, w_sim / s
        cal = " + isotonic calibration" if b and (b.get("isotonic") or b.get("isotonic_favorite")) else ""
        print(
            f"Ensemble P(team1 wins) [secondary]: {ens:.2%} "
            f"({w_ml:.0%} ML + {w_sim:.0%} simulation{cal})"
        )
    t1m = out.get("ml_team1_total_runs")
    t2m = out.get("ml_team2_total_runs")
    if t1m is not None and t2m is not None:
        print(
            f"Ridge team-total head (from match features): "
            f"team1={t1m:.1f}, team2={t2m:.1f}"
        )
    print()
    print("Top 5 batters (predicted runs):")
    print(out["top5_batters"].to_string(index=False))
    print()
    print("Top 3 bowlers (predicted wickets):")
    print(out["top3_bowlers"].to_string(index=False))
    print()
    pom = out["player_of_match"]
    r, w = float(pom["predicted_runs"]), float(pom["predicted_wickets"])
    print(
        f"Player of match: {pom['player_id']} (score={float(pom['pom_score']):.2f} — "
        f"role-weighted composite; {r:.2f} runs, {w:.2f} wkts)"
    )
    if wp is not None:
        n = out.get("n_monte_carlo", 100)
        print()
        print(
            f"Simulation win probability: team1 {wp:.2%}, team2 {(1.0 - wp):.2%} "
            f"({n} Monte Carlo sims)"
        )
    if ens is not None:
        fav = max(ens, 1.0 - ens)
        print(
            f"Confidence level (ensemble): {confidence_level(fav)} "
            f"(favorite {fav:.2%}; HIGH >65%, MEDIUM 55–65%, LOW <55%)"
        )
    elif wp is not None:
        fav = max(wp, 1.0 - wp)
        print(
            f"Confidence level (simulation): {confidence_level(fav)} "
            f"(favorite {fav:.2%}; HIGH >65%, MEDIUM 55–65%, LOW <55%)"
        )


def main():
    parser = argparse.ArgumentParser(description="Simulate a match between two teams.")
    parser.add_argument(
        "--team1",
        type=str,
        default="",
        help="Comma-separated player names for team1 (must match player_id in data)",
    )
    parser.add_argument(
        "--team2",
        type=str,
        default="",
        help="Comma-separated player names for team2",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=100,
        help="Monte Carlo simulations for win probability (0 to skip)",
    )
    parser.add_argument(
        "--team1-name",
        type=str,
        default="",
        help="Franchise name for IPL 2026 squad validation (team1)",
    )
    parser.add_argument(
        "--team2-name",
        type=str,
        default="",
        help="Franchise name for IPL 2026 squad validation (team2)",
    )
    parser.add_argument(
        "--no-squad-check",
        action="store_true",
        help="Skip IPL 2026 squad and overseas validation",
    )
    parser.add_argument(
        "--venue",
        type=str,
        default="",
        help="Stadium/city for spin-vs-pace wicket nudges (e.g. Chinnaswamy, Bengaluru)",
    )
    parser.add_argument(
        "--match-date",
        type=str,
        default="",
        help="ISO date YYYY-MM-DD for franchise momentum in ML heads (with --team1-name/--team2-name)",
    )
    args = parser.parse_args()

    if args.team1.strip() and args.team2.strip():
        team1 = [p.strip() for p in args.team1.split(",") if p.strip()]
        team2 = [p.strip() for p in args.team2.split(",") if p.strip()]
        t1n = args.team1_name.strip() or None
        t2n = args.team2_name.strip() or None
    else:
        team1 = ["RG Sharma", "Q de Kock", "SA Yadav"]
        team2 = ["RD Gaikwad", "MS Dhoni", "S Dube"]
        t1n = "Mumbai Indians"
        t2n = "Chennai Super Kings"

    validate_squad = not args.no_squad_check
    ven = (args.venue or "").strip() or None
    md = (args.match_date or "").strip() or None
    out = run_simulation(
        team1,
        team2,
        n_monte_carlo=args.sims,
        team1_name=t1n,
        team2_name=t2n,
        validate_squad=validate_squad,
        venue=ven,
        match_date=md,
    )
    print_report(out)
    print("Match simulation complete")
    if out.get("squad_validated"):
        print("IPL 2026 squad integration complete")


if __name__ == "__main__":
    main()
