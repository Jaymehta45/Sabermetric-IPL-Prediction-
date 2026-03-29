"""
Simulate match outcomes using trained player runs/wickets models.
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from iplpred.core.match_context import PitchMultipliers
from iplpred.core.venue_bowling_adjust import apply_venue_bowling_style_to_team_df
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
MIN_TEAM_RUNS = 80.0
MAX_TEAM_RUNS = 250.0

# Final upgrade: blend learned team model with Monte Carlo simulation
ENSEMBLE_ML_WEIGHT = 0.6
ENSEMBLE_SIM_WEIGHT = 0.4

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


def get_latest_player_rows(player_ids: list[str]) -> pd.DataFrame:
    """Latest row per player after same feature prep as training."""
    df = load_and_prepare()
    ids = set(str(p).strip() for p in player_ids)
    df = df[df["player_id"].astype(str).str.strip().isin(ids)]
    found = set(df["player_id"].astype(str).str.strip().unique())
    missing = ids - found
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
    Adjust raw predictions using pre-match matchup strike rates from the latest row.
    Batting: scale runs when batting_matchup_sr differs from neutral.
    Bowling: scale wickets inversely with bowling_matchup_sr (higher SR conceded → fewer wickets).
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


def predict_team_raw(
    latest: pd.DataFrame,
    team: list[str],
    runs_model,
    wk_model,
) -> pd.DataFrame:
    """Model outputs + matchup scaling (uses batting_matchup_sr / bowling_matchup_sr on latest row)."""
    team = [str(p).strip() for p in team]
    if len(team) != len(set(team)):
        raise ValueError("Duplicate player in team list.")
    if len(team) > MAX_SQUAD:
        raise ValueError(f"Each team may have at most {MAX_SQUAD} players.")
    sub = latest[latest["player_id"].astype(str).str.strip().isin(team)].copy()
    if len(sub) != len(team):
        raise ValueError("Missing player in latest features vs team list.")
    sub = _order_team_rows(sub, team)
    X = sub[FEATURE_COLS].values.astype(float)
    sub["pred_runs_raw"] = runs_model.predict(X)
    sub["pred_wk_raw"] = wk_model.predict(X)
    return apply_matchup_scaling(sub)


def ensemble_win_probability(
    ml_p: float | None,
    sim_p: float | None,
) -> float | None:
    """0.6 * ML + 0.4 * simulation when both exist."""
    if ml_p is None and sim_p is None:
        return None
    if ml_p is None:
        return float(sim_p)
    if sim_p is None:
        return float(ml_p)
    return float(ENSEMBLE_ML_WEIGHT * ml_p + ENSEMBLE_SIM_WEIGHT * sim_p)


def clip_team_total(total: float) -> float:
    return float(np.clip(total, MIN_TEAM_RUNS, MAX_TEAM_RUNS))


def _noise_on_preds(
    runs: np.ndarray,
    wickets: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """predicted_runs += N(0, 0.15 * predicted_runs); same for wickets (floor scale at 0)."""
    r = np.maximum(runs.astype(float), 0.0)
    w = np.maximum(wickets.astype(float), 0.0)
    std_r = 0.15 * np.maximum(r, 1e-9)
    std_w = 0.15 * np.maximum(w, 1e-9)
    runs_n = r + rng.normal(0.0, std_r, size=r.shape)
    wk_n = w + rng.normal(0.0, std_w, size=w.shape)
    return np.maximum(runs_n, 0.0), np.maximum(wk_n, 0.0)


def _impact_base(runs: pd.Series, wk: pd.Series) -> pd.Series:
    return runs + 25.0 * wk


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
    sub["impact_base"] = _impact_base(sub["predicted_runs"], sub["predicted_wickets"])

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


def pom_score_row(runs: float, wk: float) -> float:
    s = runs + 25.0 * wk
    if runs > 50:
        s += 10.0
    if wk >= 3:
        s += 15.0
    return s


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
        pr, pw = _noise_on_preds(pr, pw, rng)
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
        out["impact_base"] = _impact_base(out["predicted_runs"], out["predicted_wickets"])
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
    """Winner from unclipped team run totals (clipping is display-only)."""
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
) -> dict:
    if validate_squad and team1_name and team2_name:
        validate_team_playing_xi(team1_name, team1)
        validate_team_playing_xi(team2_name, team2)

    runs_model, wk_model = load_models()
    all_ids = list(dict.fromkeys([str(p).strip() for p in team1 + team2]))
    df_full = load_and_prepare()
    latest = get_latest_player_rows(all_ids)
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
    total1 = clip_team_total(raw1)
    total2 = clip_team_total(raw2)

    winner = pick_winner(raw1, raw2)

    combined = pd.concat(
        [
            t1_act.assign(side="team1"),
            t2_act.assign(side="team2"),
        ],
        ignore_index=True,
    )
    combined["pom_score"] = [
        pom_score_row(r, w) for r, w in zip(combined["predicted_runs"], combined["predicted_wickets"])
    ]

    cols5 = ["player_id", "side", "predicted_runs"]
    if "strike_rate" in combined.columns:
        cols5.append("strike_rate")
    top5_bat = combined.nlargest(5, "predicted_runs")[cols5]
    cols3 = ["player_id", "side", "predicted_wickets"]
    if "economy" in combined.columns:
        cols3.append("economy")
    top3_bowl = combined.nlargest(3, "predicted_wickets")[cols3]
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
            r1 = team_total_runs(t1_mc)
            r2 = team_total_runs(t2_mc)
            if pick_winner(r1, r2) == "team1":
                wins1 += 1
        win_prob_team1 = wins1 / n_monte_carlo

    eff_t1 = t1_act["player_id"].astype(str).str.strip().tolist()
    eff_t2 = t2_act["player_id"].astype(str).str.strip().tolist()
    leader_p_team1 = learned_team1_win_proba_from_rosters(latest, eff_t1, eff_t2)
    ensemble_p_team1 = ensemble_win_probability(leader_p_team1, win_prob_team1)
    ml_team_totals = predict_team_totals_from_rosters(latest, eff_t1, eff_t2)

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
    }


def six_match_outcomes(
    sim_out: dict,
    team1_name: str | None = None,
    team2_name: str | None = None,
) -> dict:
    """
    Six pre-match style predictions from a simulation dict:
    winning team, best batsman, best bowler, top 5 batsmen, top 3 bowlers, player of the match.
    """
    w = sim_out["winner"]
    if w == "team1":
        winning_team = team1_name or "team1"
    elif w == "team2":
        winning_team = team2_name or "team2"
    else:
        winning_team = "tie"

    top5 = sim_out["top5_batters"]
    top3 = sim_out["top3_bowlers"]
    best_bat = top5.iloc[0] if len(top5) else None
    best_bowl = top3.iloc[0] if len(top3) else None

    return {
        "winning_team": winning_team,
        "best_batsman": best_bat,
        "best_bowler": best_bowl,
        "top5_batsmen": top5,
        "top3_bowlers": top3,
        "player_of_the_match": sim_out["player_of_match"],
    }


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
    ens = out.get("ensemble_p_team1")
    if ens is not None:
        print(
            f"Ensemble P(team1 wins): {ens:.2%} "
            f"({ENSEMBLE_ML_WEIGHT:.0%} ML + {ENSEMBLE_SIM_WEIGHT:.0%} simulation)"
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
    base = r + 25.0 * w
    bonus = pom["pom_score"] - base
    print(
        f"Player of match: {pom['player_id']} (score={pom['pom_score']:.2f} = "
        f"{r:.2f} + 25×{w:.2f} + {bonus:.2f} bonus)"
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
    out = run_simulation(
        team1,
        team2,
        n_monte_carlo=args.sims,
        team1_name=t1n,
        team2_name=t2n,
        validate_squad=validate_squad,
        venue=ven,
    )
    print_report(out)
    print("Match simulation complete")
    if out.get("squad_validated"):
        print("IPL 2026 squad integration complete")


if __name__ == "__main__":
    main()
