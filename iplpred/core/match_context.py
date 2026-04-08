"""
Pre-match context: toss (who bats first), pitch report text, optional numeric overrides.

Used to (1) order teams so team1 = first innings / team2 = second, and
(2) apply mild multipliers to predicted runs/wickets from pitch signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BattingFirst = Literal["team1", "team2"]


@dataclass(frozen=True)
class PitchMultipliers:
    """Applied to all predicted runs / wickets after playing-prob (full XI mode)."""

    first_innings_runs: float = 1.0
    second_innings_runs: float = 1.0
    first_innings_wickets: float = 1.0
    second_innings_wickets: float = 1.0


@dataclass(frozen=True)
class MatchContext:
    """
    - batting_first: which side bats first (team1 or team2 **before** any reordering).
      If set, predict_match_outcomes will swap XIs so internal team1 always bats first.
    - pitch_text: free-text commentary; use parse_pitch_report() or pass explicit mults.
    - Explicit mults override parsed pitch when not None.
    - venue: optional stadium/city string; used for spin-vs-pace wicket nudges (e.g. small grounds).
    """

    batting_first: BattingFirst | None = None
    pitch_text: str | None = None
    pitch: PitchMultipliers | None = None
    venue: str | None = None


def _is_high_scoring_par_track(t: str) -> bool:
    """
    True when commentary explicitly describes a 180–220 style T20 belter / cited par total.
    On those tracks, lines like 'good luck to the bowlers' are rhetorical — do not pull mults down.
    """
    if "belter" in t:
        return True
    if "199" in t and "average" in t:
        return True
    if "216" in t and "average" in t:
        return True
    if "190" in t and "average" in t and "score" in t:
        return True
    if "200-run" in t or "200 run ground" in t or ("200 run" in t and "ground" in t):
        return True
    if "more than 200" in t:
        return True
    if "ballpark" in t and ("200" in t or "199" in t):
        return True
    if "batting paradise" in t or "batters paradise" in t:
        return True
    if "bat dominating" in t or "batting dominating" in t:
        return True
    if "250" in t and ("average" in t or "par " in t or "score" in t):
        return True
    # Cited 250+ with explicit "runs galore" / very high scoring — still a belter read
    if "250" in t and ("galore" in t or "runs galore" in t or "very high scoring" in t):
        return True
    if "high scoring" in t and ("chase" in t or "run chase" in t):
        return True
    if "runs on the board" in t and ("average" in t or "216" in t or "200" in t):
        return True
    return False


def _high_par_target_lift(t: str) -> float:
    """
    Extra multiplier on innings run *targets* when broadcast cites par scores (~199–200) or belters.
    Applied on top of keyword bat_score. Caps keep win-prob sim stable.
    """
    lift = 1.0
    if "belter" in t:
        lift += 0.20
    if "199" in t and "average" in t:
        lift += 0.24
    # Same idea as 199 — many reports cite ~198 first-innings par at Kotla-style venues
    if "198" in t and "average" in t:
        lift += 0.22
    if "216" in t and "average" in t:
        lift += 0.26
    if "200" in t:
        lift += 0.16
    if "200-run" in t or "200 run ground" in t:
        lift += 0.10
    if "more than 200" in t:
        lift += 0.12
    if "250" in t and ("average" in t or "par" in t):
        lift += 0.14
    # Broadcast: "250, runs galore" without tying 250 to the word "average" in the same clause
    if "250" in t and ("galore" in t or "runs galore" in t):
        lift += 0.16
    # Small ground + high-scoring language — extra toward 200+ innings totals
    if "small ground" in t and ("high scoring" in t or "batting" in t):
        lift += 0.06
    if "bat dominating" in t or "batting dominating" in t:
        lift += 0.08
    if "fours" in t and "sixes" in t:
        lift += 0.06
    if "ballpark" in t:
        lift += 0.06
    if "economy" in t and ("10." in t or "10.2" in t or " 10" in t):
        lift += 0.05
    return min(1.85, lift)


def parse_pitch_report(text: str) -> PitchMultipliers:
    """
    Map common commentary phrases to multipliers (conservative; tune over time).

    Positive batting surface / high scoring: nudge runs up, wickets slightly down.
    Bowling / tough batting: opposite. Chase / dew language: small second-innings runs bump.

    When the report reads like a **high-par T20 belter** (e.g. cited 199/200 averages, belter,
    chase 200+), we **lift** run multipliers so calibrated innings totals sit in the same
    ballpark as the commentary — the Ridge team-total head is often ~110–130 on flat decks.
    """
    t = text.lower()
    high_par = _is_high_scoring_par_track(t)

    # Batting-friendly signals
    bat_score = 0.0
    for w in (
        "batting",
        "runs",
        "high scoring",
        "high-scoring",
        "short boundaries",
        "boundary",
        "flat",
        "good for batting",
        "bat first",
        "score",
        "muscle",
        "belter",
        "sixes",
        "fours",
    ):
        if w in t:
            bat_score += 0.15
    # Grass can help seam early — small pull toward bowlers if "covering of grass" without "runs"
    if "grass" in t and "runs" not in t and "batting" not in t:
        bat_score -= 0.05
    if "bare patch" in t or "bare patches" in t:
        bat_score -= 0.04 if high_par else 0.08
    # Bowling / tough for batters — skip on explicit high-par tracks (commentary still names bowlers)
    if not high_par:
        for w in (
            "bowling",
            "bowlers",
            "turn",
            "grip",
            "difficult to bat",
            "two-paced",
            "two paced",
            "seam",
            "movement",
        ):
            if w in t:
                bat_score -= 0.12

    bat_score = max(-1.0, min(1.0, bat_score))
    # runs mult ~ [0.94, 1.08] before high-par lift; wickets inverse mild
    runs_m = 1.0 + 0.07 * bat_score
    wk_m = 1.0 - 0.05 * bat_score

    # Broadcast-cited par totals (e.g. 199 avg, 200 chase) — strong lift toward ~190–210 innings
    par_lift = _high_par_target_lift(t)
    runs_m = min(1.85, runs_m * par_lift)
    wk_m = max(0.82, wk_m - 0.03 * max(0.0, par_lift - 1.0))

    # Second innings / chase / dew (T20 chase advantage — mild)
    chase_bonus = 0.0
    for w in ("chase", "dew", "second innings", "second-innings", "prefer to chase", "chasing"):
        if w in t:
            chase_bonus += 0.012
    chase_bonus = min(0.06 if high_par else 0.04, chase_bonus)
    second_runs = 1.0 + chase_bonus

    return PitchMultipliers(
        first_innings_runs=runs_m,
        second_innings_runs=min(1.88, runs_m * second_runs),
        first_innings_wickets=wk_m,
        second_innings_wickets=wk_m * (1.0 - 0.5 * chase_bonus),
    )


def _venue_run_pitch_nudge(venue: str | None) -> float:
    """Mild extra run mult for venues that trend high-scoring in T20 (does not replace pitch text)."""
    if not venue or not str(venue).strip():
        return 1.0
    v = str(venue).lower()
    if "ahmedabad" in v or "narendra modi" in v or "motera" in v:
        return 1.028
    # Arun Jaitley / Kotla — short boundaries, T20 totals often ~190–210+ on good decks
    if "jaitley" in v or "arun jaitley" in v:
        return 1.032
    return 1.0


def _apply_venue_to_pitch(pm: PitchMultipliers, venue: str | None) -> PitchMultipliers:
    n = _venue_run_pitch_nudge(venue)
    if n <= 1.0:
        return pm
    return PitchMultipliers(
        first_innings_runs=min(1.88, pm.first_innings_runs * n),
        second_innings_runs=min(1.88, pm.second_innings_runs * n),
        first_innings_wickets=pm.first_innings_wickets,
        second_innings_wickets=pm.second_innings_wickets,
    )


def resolve_pitch_multipliers(ctx: MatchContext) -> PitchMultipliers:
    if ctx.pitch is not None:
        return ctx.pitch
    if ctx.pitch_text and ctx.pitch_text.strip():
        return _apply_venue_to_pitch(parse_pitch_report(ctx.pitch_text), ctx.venue)
    return _apply_venue_to_pitch(PitchMultipliers(), ctx.venue)


def parse_batting_first(
    value: str,
) -> BattingFirst | None:
    """Accept 'team1', 'team2', case-insensitive strip."""
    v = value.strip().lower()
    if v in ("team1", "t1", "1"):
        return "team1"
    if v in ("team2", "t2", "2"):
        return "team2"
    return None


def parse_toss_elected(
    toss_winner: str,
    elected_to: str,
    team1_name: str,
    team2_name: str,
) -> BattingFirst | None:
    """
    toss_winner: must match team1_name or team2_name (case-insensitive prefix ok).
    elected_to: 'bat' | 'field' | 'bowl' (field/bowl = sent the other team in).
    Returns which *input* side (team1 or team2) bats first.
    """
    t1 = team1_name.strip()
    t2 = team2_name.strip()
    tw = toss_winner.strip().lower()
    e = elected_to.strip().lower()

    def is_t1() -> bool:
        return tw == t1.lower() or t1.lower().startswith(tw) or tw in t1.lower()

    def is_t2() -> bool:
        return tw == t2.lower() or t2.lower().startswith(tw) or tw in t2.lower()

    winner_t1 = is_t1() and not is_t2()
    winner_t2 = is_t2() and not is_t1()
    if not winner_t1 and not winner_t2:
        return None

    bats_first_is_winner = e in ("bat", "batting")
    if e in ("field", "bowl", "bowling"):
        bats_first_is_winner = False

    if winner_t1:
        return "team1" if bats_first_is_winner else "team2"
    return "team2" if bats_first_is_winner else "team1"
