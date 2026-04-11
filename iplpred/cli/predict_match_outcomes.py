"""
Pre-match prediction for six outcomes when both playing XIs are known (11 vs 11):

1. Winning team
2. Best batsman (highest predicted runs)
3. Best bowler (highest predicted wickets)
4. Top 5 batsmen
5. Top 3 bowlers
6. Player of the match (role-weighted composite score from predicted runs/wickets; not official POTM)

Uses the same models and rules as match_simulator. For a plain 11 vs 11, all 11
count. With ``--team1-impact`` / ``--team2-impact`` (12 per side), the default
is to trim the lowest predicted-impact row from each 12 for team totals; use
``--impact-add-only`` to count all 12.

Optional match context (toss, pitch) via ``MatchContext``: after reordering,
**team1 is always the side batting first** (first innings). Supply
``batting_first="team2"`` if your ``--xi2`` bats first, or use ``--toss-winner``
and ``--elected-to``.

**Process 1A vs 1B (impact substitutes only):**
  * **1A** (default): XI + impact sub = 12 names per side; **lowest predicted-impact**
    row is **dropped from team totals** (``--process-variant 1a``; same as omitting).
  * **1B** (experimental): **all 12** count toward team totals (``--process-variant 1b``
    or ``--impact-add-only``). Same RF models and features; only scoring changes.

Example:
  python predict_match_outcomes.py \\
    --team1-name "Mumbai Indians" --team2-name "Chennai Super Kings" \\
    --xi1 "RG Sharma,Q de Kock,SA Yadav,..." \\
    --xi2 "RD Gaikwad,MS Dhoni,..." \\
    --batting-first team1 \\
    --pitch-text "Good batting surface, short boundaries, prefer to chase"

Export JSON for prediction_log (log_prediction.py pre):
  --export-json predictions/match_pre.json --match-date 2026-03-28 \\
    [--match-key IPL2026-M01] [--venue "M Chinnaswamy Stadium, Bengaluru"]

**Innings team scores:** After the six outcomes, the CLI prints **calibrated**
mean team runs for innings 1 and 2: per-player RF outputs are scaled so the
team sum matches the Ridge **team-total** model when available, otherwise a
T20 anchor (~165) while preserving relative strength. Wickets fallen use the
fielding XI ``predicted_wickets`` sum (capped at 10). Logged in ``pred_extra`` as
``pred_first_innings_team_runs`` / ``pred_second_innings_team_runs`` and wicket
keys; ``team_run_calibration`` records scales and pre-scale sums.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from iplpred.core.match_context import (
    MatchContext,
    PitchMultipliers,
    parse_batting_first,
    parse_toss_elected,
    resolve_pitch_multipliers,
)
from iplpred.core.player_registry import resolve_playing_xi, resolve_player_name
from iplpred.match.match_payload import load_match_json, payload_to_match_context
from iplpred.match.match_simulator import run_simulation, six_match_outcomes
from iplpred.paths import REPO_ROOT

PLAYING_XI_SIZE = 11


def normalize_teams_for_batting_first(
    team1_xi: list[str],
    team2_xi: list[str],
    team1_name: str | None,
    team2_name: str | None,
    batting_first: str | None,
) -> tuple[list[str], list[str], str | None, str | None, bool]:
    """
    If batting_first is ``team2``, swap so internal team1 = first innings.
    Returns (t1_xi, t2_xi, t1_name, t2_name, swapped).
    """
    if batting_first is None or batting_first == "team1":
        return team1_xi, team2_xi, team1_name, team2_name, False
    if batting_first == "team2":
        return team2_xi, team1_xi, team2_name, team1_name, True
    raise ValueError(f"batting_first must be 'team1' or 'team2', got {batting_first!r}")


def parse_player_list(s: str) -> list[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def parse_player_file(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    out: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "," in line:
            out.extend(parse_player_list(line))
        else:
            out.append(line)
    return out


def validate_xi(name: str, players: list[str], *, n: int = PLAYING_XI_SIZE) -> None:
    if len(players) != n:
        raise ValueError(
            f"{name} must have exactly {n} players (got {len(players)}). "
            "Use player_id strings as in player_features.csv."
        )
    if len(players) != len(set(players)):
        raise ValueError(f"{name} has duplicate player_ids.")


def predict_match_outcomes(
    team1_xi: list[str],
    team2_xi: list[str],
    *,
    team1_name: str | None = None,
    team2_name: str | None = None,
    n_monte_carlo: int = 100,
    validate_squad: bool = True,
    match_context: MatchContext | None = None,
    use_player_registry: bool = False,
    use_recent_form_shrinkage: bool = True,
    shrink_team_prior_weight: float = 0.45,
    team1_impact: str | None = None,
    team2_impact: str | None = None,
    impact_trim_lowest: bool = True,
    match_date: str | None = None,
) -> dict:
    """
    Run simulation and return
    ``{"six": ..., "simulation": ..., "batting_order_swapped": bool}``.

    If ``team1_impact`` / ``team2_impact`` is set, that side's XI must be **11**
    names; the impact sub is **appended** (12th player). By default
    (``impact_trim_lowest=True``) one row per 12-player side is **dropped** from
    team-total scoring (lowest ``runs + 25×wickets`` heuristic). Set
    ``impact_trim_lowest=False`` so **all 12** count (XI + sub, no trim).

    If ``match_context.batting_first`` is set, teams are reordered so **team1
    always bats first** in the model (first innings multipliers apply to team1).

    With ``use_player_registry=True``, names are resolved via ``player_aliases.csv``
    to stats ``player_id`` strings before validation.
    """
    if use_player_registry:
        team1_xi, res1 = resolve_playing_xi(team1_xi, allow_fuzzy=True)
        team2_xi, res2 = resolve_playing_xi(team2_xi, allow_fuzzy=True)
        name_resolutions: dict | None = {"team1": res1, "team2": res2}
    else:
        name_resolutions = None

    if team1_impact:
        imp_r = resolve_player_name(team1_impact.strip(), allow_fuzzy=True)
        imp_id = str(imp_r.stats_player_id).strip()
        validate_xi("team1_xi", team1_xi, n=PLAYING_XI_SIZE)
        if imp_id in {str(p).strip() for p in team1_xi}:
            raise ValueError("team1_impact player is already in team1 XI")
        team1_xi = [*team1_xi, imp_id]
        if name_resolutions is None:
            name_resolutions = {}
        name_resolutions["team1_impact"] = [imp_r]

    if team2_impact:
        imp_r = resolve_player_name(team2_impact.strip(), allow_fuzzy=True)
        imp_id = str(imp_r.stats_player_id).strip()
        validate_xi("team2_xi", team2_xi, n=PLAYING_XI_SIZE)
        if imp_id in {str(p).strip() for p in team2_xi}:
            raise ValueError("team2_impact player is already in team2 XI")
        team2_xi = [*team2_xi, imp_id]
        if name_resolutions is None:
            name_resolutions = {}
        name_resolutions["team2_impact"] = [imp_r]

    validate_xi("team1_xi", team1_xi, n=len(team1_xi))
    validate_xi("team2_xi", team2_xi, n=len(team2_xi))

    has_12 = len(team1_xi) == PLAYING_XI_SIZE + 1 or len(team2_xi) == PLAYING_XI_SIZE + 1
    use_drop = has_12 and impact_trim_lowest

    bf = match_context.batting_first if match_context else None
    t1_xi, t2_xi, n1, n2, swapped = normalize_teams_for_batting_first(
        team1_xi, team2_xi, team1_name, team2_name, bf
    )

    pitch: PitchMultipliers = (
        resolve_pitch_multipliers(match_context) if match_context else PitchMultipliers()
    )

    venue = None
    if match_context is not None:
        v = getattr(match_context, "venue", None)
        venue = str(v).strip() if v else None

    sim = run_simulation(
        t1_xi,
        t2_xi,
        n_monte_carlo=n_monte_carlo,
        team1_name=n1,
        team2_name=n2,
        validate_squad=validate_squad,
        drop_lowest_impact=use_drop,
        pitch=pitch,
        use_recent_form_shrinkage=use_recent_form_shrinkage,
        shrink_team_prior_weight=shrink_team_prior_weight,
        venue=venue,
        match_date=match_date,
    )
    six = six_match_outcomes(sim, team1_name=n1, team2_name=n2)
    return {
        "six": six,
        "simulation": sim,
        "batting_order_swapped": swapped,
        "match_context": match_context,
        "name_resolutions": name_resolutions,
    }


def print_six(six: dict, meta: dict | None = None) -> None:
    print("=" * 60)
    print("PRE-MATCH PREDICTIONS (6 outcomes)")
    print("=" * 60)
    print()
    if meta:
        n1 = meta.get("team1_name")
        n2 = meta.get("team2_name")
        if n1 and n2:
            print(f"(team1 = first innings: {n1} | team2 = chase: {n2})")
            print()
        pm = meta.get("pitch_multipliers")
        if pm is not None:
            print(
                f"Pitch multipliers — 1st inn runs×{pm.first_innings_runs:.3f} "
                f"wk×{pm.first_innings_wickets:.3f} | "
                f"2nd inn runs×{pm.second_innings_runs:.3f} "
                f"wk×{pm.second_innings_wickets:.3f}"
            )
            print()
        vh = meta.get("venue_spin_harshness")
        if vh is not None and float(vh) > 0:
            print(
                f"Venue spin harshness (0–1): {float(vh):.2f} — spinners' predicted wickets "
                "scaled down at batting-friendly / small grounds; pace slightly up."
            )
            print()
    p1 = six.get("p_team1_win")
    basis = six.get("win_probability_basis") or ""
    n1 = meta.get("team1_name") if meta else None
    n2 = meta.get("team2_name") if meta else None
    if basis == "hybrid_ensemble":
        basis_lbl = "Hybrid ML+sim+calibration"
    elif basis == "monte_carlo":
        basis_lbl = "Monte Carlo"
    else:
        basis_lbl = basis or "deterministic totals"
    if p1 is not None and n1 and n2 and six.get("winning_team") == "tie":
        print(
            f"1. Essentially tied ({basis_lbl}) — P({n1})={float(p1):.1%}, "
            f"P({n2})={1.0 - float(p1):.1%}"
        )
    elif p1 is not None and n1 and n2:
        ppw = six.get("p_predicted_winner")
        pp_s = f"{float(ppw):.1%}" if ppw is not None else "—"
        print(
            f"1. Predicted winner: {six['winning_team']} — "
            f"P({six['winning_team']} wins) = {pp_s} "
            f"({basis_lbl}; P({n1})={float(p1):.1%}, P({n2})={1.0 - float(p1):.1%})"
        )
    elif p1 is not None:
        print(
            f"1. Predicted winner: {six['winning_team']} — "
            f"P(win) = {float(six.get('p_predicted_winner') or 0):.1%} ({basis_lbl})"
        )
    else:
        print(f"1. Predicted winner: {six['winning_team']} (mean team totals; use --sims > 0 for win %)")
    print()
    bb = six["best_batsman"]
    if bb is not None:
        print(
            f"2. Best batsman: {bb['player_id']} "
            f"({bb['side']}, predicted runs={float(bb['predicted_runs']):.2f})"
        )
    else:
        print("2. Best batsman: —")
    print()
    bw = six["best_bowler"]
    if bw is not None:
        print(
            f"3. Best bowler: {bw['player_id']} "
            f"({bw['side']}, predicted wickets={float(bw['predicted_wickets']):.2f})"
        )
    else:
        print("3. Best bowler: —")
    print()
    print("4. Top 5 batsmen (predicted runs):")
    print(six["top5_batsmen"].to_string(index=False))
    print()
    print("5. Top 3 bowlers (predicted wickets):")
    print(six["top3_bowlers"].to_string(index=False))
    print()
    pom = six["player_of_the_match"]
    r, w = float(pom["predicted_runs"]), float(pom["predicted_wickets"])
    print(
        f"6. Player of the match: {pom['player_id']} ({pom['side']}) "
        f"— score={float(pom['pom_score']):.2f} "
        f"(role-weighted composite; {r:.2f} runs, {w:.2f} wkts)"
    )
    print()
    print("=" * 60)


def _player_id_from_row(row: object) -> str:
    if row is None:
        return ""
    if isinstance(row, pd.Series):
        return str(row.get("player_id", "") or "")
    if isinstance(row, dict):
        return str(row.get("player_id", "") or "")
    return str(row)


def _slug_key(*parts: str) -> str:
    raw = "-".join(p for p in parts if p and str(p).strip())
    s = re.sub(r"[^a-zA-Z0-9]+", "-", raw.strip().lower()).strip("-")
    return s or "match"


def build_prediction_log_payload(
    six: dict,
    sim: dict,
    *,
    team1_first_innings_name: str,
    team2_chase_name: str,
    match_key: str,
    match_date: str,
    venue: str,
    n_sims: int,
    batting_order_swapped: bool,
    used_impact_subs: bool = False,
) -> dict:
    """
    Schema compatible with ``scripts/log_prediction.py pre`` (see REQUIRED_PRE there).
    team1_* = first innings (after toss reordering); pred_p_team1_win = P(team1 wins).
    """
    top5 = six.get("top5_batsmen")
    top3 = six.get("top3_bowlers")
    if isinstance(top5, pd.DataFrame) and len(top5):
        ids5 = top5["player_id"].astype(str).tolist()[:5]
    else:
        ids5 = []
    while len(ids5) < 5:
        ids5.append("")
    if isinstance(top3, pd.DataFrame) and len(top3):
        ids3 = top3["player_id"].astype(str).tolist()[:3]
    else:
        ids3 = []
    while len(ids3) < 3:
        ids3.append("")

    # Logged P(team1) matches headline six outcomes: hybrid ensemble when available, else MC, else 0.5
    p1 = six.get("p_team1_win")
    if p1 is None:
        p1 = sim.get("ensemble_p_team1")
    if p1 is None:
        p1 = sim.get("win_probability_team1")
    if p1 is None:
        p1 = 0.5
    p1 = float(p1)

    pm = sim.get("pitch_multipliers")
    pred_extra: dict = {
        "sims": int(n_sims),
        "win_probability_basis": six.get("win_probability_basis"),
        "ensemble_p_team1": sim.get("ensemble_p_team1"),
        "sim_win_probability_team1": sim.get("win_probability_team1"),
        "leader_model_p_team1": sim.get("leader_model_p_team1"),
        "batting_order_swapped": batting_order_swapped,
        "impact_trim_lowest": bool(sim.get("drop_lowest_impact")),
        "impact_dropped_team1": sim.get("impact_dropped_team1"),
        "impact_dropped_team2": sim.get("impact_dropped_team2"),
    }
    if pm is not None:
        pred_extra["pitch_multipliers"] = {
            "first_innings_runs": getattr(pm, "first_innings_runs", None),
            "second_innings_runs": getattr(pm, "second_innings_runs", None),
            "first_innings_wickets": getattr(pm, "first_innings_wickets", None),
            "second_innings_wickets": getattr(pm, "second_innings_wickets", None),
        }
    vh = sim.get("venue_spin_harshness")
    if vh is not None:
        pred_extra["venue_spin_harshness"] = float(vh)

    drop_imp = bool(sim.get("drop_lowest_impact"))
    if used_impact_subs:
        pred_extra["process_variant"] = "1A" if drop_imp else "1B"
        pred_extra["full_playing_twelve"] = not drop_imp
    else:
        pred_extra["process_variant"] = "1A"
        pred_extra["full_playing_twelve"] = False

    # Mean predicted team totals (RF player pipeline + pitch/venue; not ball-by-ball MC)
    t1r = sim.get("team1_total_runs")
    t2r = sim.get("team2_total_runs")
    if t1r is not None:
        pred_extra["pred_first_innings_team_runs"] = float(t1r)
    if t2r is not None:
        pred_extra["pred_second_innings_team_runs"] = float(t2r)
    w1 = sim.get("pred_wickets_first_innings")
    w2 = sim.get("pred_wickets_second_innings")
    if w1 is not None:
        pred_extra["pred_first_innings_team_wickets"] = float(w1)
    if w2 is not None:
        pred_extra["pred_second_innings_team_wickets"] = float(w2)
    w1r = sim.get("pred_wickets_first_innings_raw_sum")
    w2r = sim.get("pred_wickets_second_innings_raw_sum")
    if w1r is not None:
        pred_extra["pred_first_innings_team_wickets_raw_sum"] = float(w1r)
    if w2r is not None:
        pred_extra["pred_second_innings_team_wickets_raw_sum"] = float(w2r)
    ml1 = sim.get("ml_team1_total_runs")
    ml2 = sim.get("ml_team2_total_runs")
    if ml1 is not None:
        pred_extra["ml_team_total_runs_team1"] = float(ml1)
    if ml2 is not None:
        pred_extra["ml_team_total_runs_team2"] = float(ml2)

    cm = sim.get("team_run_calibration_method")
    if cm is not None:
        pred_extra["team_run_calibration"] = {
            "method": str(cm),
            "scale_team1": sim.get("team1_calibration_scale"),
            "scale_team2": sim.get("team2_calibration_scale"),
            "sum_player_runs_before_team1": sim.get("team1_raw_runs"),
            "sum_player_runs_before_team2": sim.get("team2_raw_runs"),
        }
    db1 = sim.get("team1_dynamic_bat_mult")
    db2 = sim.get("team2_dynamic_bat_mult")
    if db1 is not None and db2 is not None:
        pred_extra["dynamic_batting_par_mult"] = {
            "team1": float(db1),
            "team2": float(db2),
        }

    pred_extra["player_process1"] = {
        "heuristic": "role_aware_v1",
        "summary": (
            "Best bat / top 5 runs exclude pure bowlers when enough batters exist; "
            "best bowl / top 3 wickets prefer bowlers and allrounders; "
            "POTM uses a role-weighted composite (not the official match award)."
        ),
    }

    return {
        "match_key": match_key,
        "match_date": match_date,
        "venue": venue,
        "team1_name": team1_first_innings_name,
        "team2_name": team2_chase_name,
        "team1_bats_first": True,
        "pred_winner": str(six.get("winning_team") or ""),
        "pred_p_team1_win": p1,
        "pred_best_bat": _player_id_from_row(six.get("best_batsman")),
        "pred_best_bowl": _player_id_from_row(six.get("best_bowler")),
        "pred_top5_bats": ids5,
        "pred_top3_bowls": ids3,
        "pred_potm": _player_id_from_row(six.get("player_of_the_match")),
        "pred_extra": pred_extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict 6 outcomes for an 11v11 match (playing XIs required)."
    )
    parser.add_argument(
        "--team1-name",
        type=str,
        default="",
        help="Franchise name (for winner label + optional squad check)",
    )
    parser.add_argument("--team2-name", type=str, default="")
    parser.add_argument(
        "--xi1",
        type=str,
        default="",
        help="Comma-separated 11 player_ids for team1",
    )
    parser.add_argument("--xi2", type=str, default="", help="Comma-separated 11 for team2")
    parser.add_argument(
        "--xi1-file",
        type=str,
        default="",
        help="File with one player_id per line (11 lines)",
    )
    parser.add_argument("--xi2-file", type=str, default="")
    parser.add_argument("--sims", type=int, default=100, help="Monte Carlo sims (0 to skip)")
    parser.add_argument(
        "--no-squad-check",
        action="store_true",
        help="Do not validate against IPL 2026 squads",
    )
    parser.add_argument(
        "--batting-first",
        type=str,
        default="",
        help="Which input side bats first: team1 or team2 (after --xi1/--xi2 order)",
    )
    parser.add_argument(
        "--toss-winner",
        type=str,
        default="",
        help="Franchise name or substring of toss winner (needs --elected-to and both team names)",
    )
    parser.add_argument(
        "--elected-to",
        type=str,
        default="",
        help="bat | field — what the toss winner chose",
    )
    parser.add_argument(
        "--pitch-text",
        type=str,
        default="",
        help="Free-text pitch report (keyword-based multipliers)",
    )
    parser.add_argument(
        "--pitch-file",
        type=str,
        default="",
        help="File containing pitch report text",
    )
    parser.add_argument(
        "--match-json",
        type=str,
        default="",
        help="Path to JSON with team names, XIs, toss, pitch (see examples/match_payload.example.json)",
    )
    parser.add_argument(
        "--use-registry",
        action="store_true",
        help="Resolve display names via data/processed/player_aliases.csv",
    )
    parser.add_argument(
        "--no-shrinkage",
        action="store_true",
        help="Disable recent-form confidence shrinkage toward team priors",
    )
    parser.add_argument(
        "--team1-impact",
        type=str,
        default="",
        help="IPL impact substitute appended to team1 XI (--xi1 must be exactly 11 names)",
    )
    parser.add_argument(
        "--team2-impact",
        type=str,
        default="",
        help="IPL impact substitute appended to team2 XI (--xi2 must be exactly 11 names)",
    )
    parser.add_argument(
        "--impact-add-only",
        action="store_true",
        help=(
            "Same as --process-variant 1b: when using impact subs, count all 12 "
            "players toward team totals (no lowest-row drop)."
        ),
    )
    parser.add_argument(
        "--process-variant",
        type=str,
        choices=("1a", "1b"),
        default="1a",
        help=(
            "Process 1A (default): impact subs present → trim lowest predicted-impact "
            "from each 12 for team totals (main pipeline). "
            "Process 1B (experimental): full playing 12 — all XI+impact count. "
            "Same models; only team-total scoring differs. "
            "Overridden by --impact-add-only (equivalent to 1b)."
        ),
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default="",
        help="Write predictions JSON for scripts/log_prediction.py pre (see --match-date)",
    )
    parser.add_argument(
        "--match-date",
        type=str,
        default="",
        help="ISO date YYYY-MM-DD — export-json, and ML momentum (with team names)",
    )
    parser.add_argument(
        "--match-key",
        type=str,
        default="",
        help="Stable key for prediction log export (default: slug from teams + date)",
    )
    parser.add_argument(
        "--venue",
        type=str,
        default="",
        help="Venue string for export-json (or use venue in --match-json)",
    )
    parser.add_argument(
        "--show-ensemble-details",
        action="store_true",
        help="Print ML/ensemble win probability (secondary to Monte Carlo headline)",
    )
    args = parser.parse_args()

    payload = None
    load_path = None
    if args.match_json:
        mj = Path(args.match_json)
        load_path = mj if mj.is_file() else (REPO_ROOT / mj).resolve()
        if not load_path.is_file():
            raise SystemExit(f"--match-json file not found: {load_path}")
        payload = load_match_json(load_path)
        team1 = payload.xi1
        team2 = payload.xi2
        t1n = payload.team1_name or None
        t2n = payload.team2_name or None
        mc = payload_to_match_context(payload)
        if len(team1) != PLAYING_XI_SIZE or len(team2) != PLAYING_XI_SIZE:
            raise SystemExit(
                f"match-json: need {PLAYING_XI_SIZE} players per side; got {len(team1)}, {len(team2)}"
            )
    else:
        if args.xi1_file:
            team1 = parse_player_file(REPO_ROOT / args.xi1_file)
        else:
            team1 = parse_player_list(args.xi1)
        if args.xi2_file:
            team2 = parse_player_file(REPO_ROOT / args.xi2_file)
        else:
            team2 = parse_player_list(args.xi2)

        if len(team1) != PLAYING_XI_SIZE or len(team2) != PLAYING_XI_SIZE:
            raise SystemExit(
                f"Need exactly {PLAYING_XI_SIZE} players per side. "
                f"Got team1={len(team1)}, team2={len(team2)}. "
                "Use --xi1/--xi2 or --xi1-file/--xi2-file."
            )

        t1n = args.team1_name.strip() or None
        t2n = args.team2_name.strip() or None

        pitch_text = (args.pitch_text or "").strip()
        if args.pitch_file:
            pitch_text = (REPO_ROOT / args.pitch_file).read_text(encoding="utf-8").strip()

        batting_first = parse_batting_first(args.batting_first) if args.batting_first else None
        if args.toss_winner and args.elected_to:
            if not t1n or not t2n:
                raise SystemExit("--toss-winner/--elected-to require --team1-name and --team2-name")
            batting_first = parse_toss_elected(
                args.toss_winner, args.elected_to, t1n, t2n
            )
            if batting_first is None:
                raise SystemExit(
                    "Could not resolve toss winner to a team; check --toss-winner vs --team1-name/--team2-name"
                )

        venue_cli = (args.venue or "").strip() or None
        mc = None
        if batting_first is not None or pitch_text or venue_cli:
            mc = MatchContext(
                batting_first=batting_first,
                pitch_text=pitch_text or None,
                venue=venue_cli,
            )

    t1_imp = args.team1_impact.strip() or None
    t2_imp = args.team2_impact.strip() or None

    impact_trim_lowest = not args.impact_add_only
    if args.process_variant == "1b":
        impact_trim_lowest = False

    match_date_for_sim = (args.match_date or "").strip() or None
    if not match_date_for_sim and payload is not None:
        rd = payload.raw.get("match_date") or payload.raw.get("date")
        if rd is not None:
            match_date_for_sim = str(rd).strip() or None

    out = predict_match_outcomes(
        team1,
        team2,
        team1_name=t1n,
        team2_name=t2n,
        n_monte_carlo=args.sims,
        validate_squad=not args.no_squad_check,
        match_context=mc,
        use_player_registry=args.use_registry,
        use_recent_form_shrinkage=not args.no_shrinkage,
        team1_impact=t1_imp,
        team2_impact=t2_imp,
        impact_trim_lowest=impact_trim_lowest,
        match_date=match_date_for_sim,
    )
    sim = out["simulation"]
    t1_name_eff, t2_name_eff = t1n, t2n
    if out.get("batting_order_swapped"):
        t1_name_eff, t2_name_eff = t2n, t1n
    if t1_imp or t2_imp:
        pv = "1B" if not impact_trim_lowest else "1A"
        print(
            f"Process {pv} — impact subs: "
            + (
                "full playing 12 (all count toward team totals)"
                if pv == "1B"
                else "trim lowest predicted-impact from each 12 for team totals"
            )
        )
        print()
    print_six(
        out["six"],
        meta={
            "team1_name": t1_name_eff,
            "team2_name": t2_name_eff,
            "pitch_multipliers": sim.get("pitch_multipliers"),
            "venue_spin_harshness": sim.get("venue_spin_harshness"),
        },
    )
    t1r = sim.get("team1_total_runs")
    t2r = sim.get("team2_total_runs")
    w1i = sim.get("pred_wickets_first_innings")
    w2i = sim.get("pred_wickets_second_innings")
    if t1_name_eff and t2_name_eff and t1r is not None and t2r is not None:
        print()
        print(
            f"Predicted innings totals (calibrated team sums — ML team-total or T20 anchor): "
            f"{t1_name_eff} (1st) ≈ {float(t1r):.1f}  |  "
            f"{t2_name_eff} (2nd) ≈ {float(t2r):.1f}"
        )
        if w1i is not None and w2i is not None:
            print(
                f"Predicted wickets fallen (fielding XI wicket sums, capped at 10): "
                f"1st innings ≈ {float(w1i):.1f}  |  2nd innings ≈ {float(w2i):.1f}"
            )
        ml1, ml2 = sim.get("ml_team1_total_runs"), sim.get("ml_team2_total_runs")
        if ml1 is not None and ml2 is not None:
            print(
                f"  (optional) dedicated team-total model: "
                f"{t1_name_eff} ≈ {float(ml1):.1f}, {t2_name_eff} ≈ {float(ml2):.1f}"
            )
    nr = out.get("name_resolutions")
    if nr and args.use_registry:
        print()
        print("Name resolution (registry):")
        for side, lst in nr.items():
            if side.endswith("_impact"):
                continue
            for r in lst:
                if r.method != "exact_registry" or r.input_raw != r.stats_player_id:
                    print(f"  {r.input_raw!r} -> {r.stats_player_id} ({r.method}, score={r.score:.1f})")
    if nr:
        imp_lines = []
        for key in ("team1_impact", "team2_impact"):
            for r in nr.get(key) or []:
                imp_lines.append(f"  {key}: {r.input_raw!r} -> {r.stats_player_id} ({r.method})")
        if imp_lines:
            print()
            print("Impact substitute (resolved):")
            for line in imp_lines:
                print(line)
    id1 = sim.get("impact_dropped_team1")
    id2 = sim.get("impact_dropped_team2")
    if (t1_imp or t2_imp) and not impact_trim_lowest:
        print()
        print(
            "Impact: all 12 players count toward team totals "
            "(Process 1B / --process-variant 1b / --impact-add-only)."
        )
    elif id1 or id2:
        print()
        print(
            "Impact pool: excluded from team total (lowest runs + 25×wickets among 12) — "
            f"team1={id1!s}, team2={id2!s}"
        )
    if args.show_ensemble_details:
        wp = sim.get("win_probability_team1")
        if wp is not None:
            print(f"Monte Carlo P(team1 wins): {wp:.2%} (component; headline uses hybrid when set)")
        lm = sim.get("leader_model_p_team1")
        if lm is not None:
            print(f"Leader (roster) model P(team1 wins): {lm:.2%}")
        if sim.get("ensemble_p_team1") is not None:
            print(f"Ensemble (ML + sim + calibration) P(team1 wins): {sim['ensemble_p_team1']:.2%}")

    export_json = (args.export_json or "").strip()
    if export_json:
        export_path = Path(export_json)
        if not export_path.is_absolute():
            export_path = (REPO_ROOT / export_path).resolve()

        match_date = (args.match_date or "").strip()
        if not match_date and payload is not None:
            rawd = payload.raw.get("match_date") or payload.raw.get("date")
            if rawd is not None:
                match_date = str(rawd).strip()
        if not match_date:
            raise SystemExit(
                "--export-json requires --match-date YYYY-MM-DD (or match_date / date in --match-json)"
            )

        venue = (args.venue or "").strip()
        if not venue and payload is not None and payload.venue:
            venue = payload.venue.strip()

        if not t1_name_eff or not t2_name_eff:
            raise SystemExit(
                "--export-json requires --team1-name and --team2-name (or match JSON with both names)"
            )

        match_key = (args.match_key or "").strip()
        if not match_key and payload is not None and payload.match_id:
            match_key = str(payload.match_id).strip()
        if not match_key:
            match_key = _slug_key(t1_name_eff, t2_name_eff, match_date)

        log_payload = build_prediction_log_payload(
            out["six"],
            sim,
            team1_first_innings_name=t1_name_eff,
            team2_chase_name=t2_name_eff,
            match_key=match_key,
            match_date=match_date,
            venue=venue,
            n_sims=int(args.sims),
            batting_order_swapped=bool(out.get("batting_order_swapped")),
            used_impact_subs=bool(t1_imp or t2_imp),
        )
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")
        print()
        print(f"Wrote prediction log JSON -> {export_path}")
        print("  (feed to: python3 scripts/log_prediction.py pre <file>)")

    print("Prediction complete.")


if __name__ == "__main__":
    main()
