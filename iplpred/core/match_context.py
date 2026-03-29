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


def parse_pitch_report(text: str) -> PitchMultipliers:
    """
    Map common commentary phrases to multipliers (conservative; tune over time).

    Positive batting surface / high scoring: nudge runs up, wickets slightly down.
    Bowling / tough batting: opposite. Chase / dew language: small second-innings runs bump.
    """
    t = text.lower()
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
    ):
        if w in t:
            bat_score += 0.15
    # Grass can help seam early — small pull toward bowlers if "covering of grass" without "runs"
    if "grass" in t and "runs" not in t and "batting" not in t:
        bat_score -= 0.05
    if "bare patch" in t or "bare patches" in t:
        bat_score -= 0.08
    # Bowling / tough for batters
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
    # runs mult ~ [0.94, 1.08], wickets inverse mild
    runs_m = 1.0 + 0.07 * bat_score
    wk_m = 1.0 - 0.05 * bat_score

    # Second innings / chase / dew (T20 chase advantage — mild)
    chase_bonus = 0.0
    for w in ("chase", "dew", "second innings", "second-innings", "prefer to chase"):
        if w in t:
            chase_bonus += 0.012
    chase_bonus = min(0.04, chase_bonus)
    second_runs = 1.0 + chase_bonus

    return PitchMultipliers(
        first_innings_runs=runs_m,
        second_innings_runs=runs_m * second_runs,
        first_innings_wickets=wk_m,
        second_innings_wickets=wk_m * (1.0 - 0.5 * chase_bonus),
    )


def resolve_pitch_multipliers(ctx: MatchContext) -> PitchMultipliers:
    if ctx.pitch is not None:
        return ctx.pitch
    if ctx.pitch_text and ctx.pitch_text.strip():
        return parse_pitch_report(ctx.pitch_text)
    return PitchMultipliers()


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
