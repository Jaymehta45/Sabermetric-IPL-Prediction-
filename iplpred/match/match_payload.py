"""
Load structured match input (JSON) for predictions: XIs, toss, pitch, metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iplpred.core.match_context import MatchContext


@dataclass
class MatchPayload:
    """Validated structure from a match JSON file."""

    team1_name: str
    team2_name: str
    xi1: list[str]
    xi2: list[str]
    batting_first: str | None  # "team1" | "team2"
    pitch_text: str | None
    toss_winner: str | None
    elected_to: str | None  # bat | field
    venue: str | None
    match_id: str | None
    notes: str | None
    raw: dict[str, Any]


def load_match_json(path: Path | str) -> MatchPayload:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Match JSON must be an object")

    t1 = str(raw.get("team1_name", "")).strip()
    t2 = str(raw.get("team2_name", "")).strip()
    xi1 = _as_str_list(raw.get("xi1") or raw.get("playing_xi_team1"))
    xi2 = _as_str_list(raw.get("xi2") or raw.get("playing_xi_team2"))

    bf = raw.get("batting_first")
    if bf is not None:
        bf = str(bf).strip().lower()
        if bf not in ("team1", "team2", ""):
            raise ValueError("batting_first must be team1 or team2")
        if bf == "":
            bf = None
    else:
        bf = None

    pitch = raw.get("pitch_text") or raw.get("pitch_report")
    pitch_text = str(pitch).strip() if pitch else None

    return MatchPayload(
        team1_name=t1,
        team2_name=t2,
        xi1=xi1,
        xi2=xi2,
        batting_first=bf,
        pitch_text=pitch_text,
        toss_winner=(str(raw["toss_winner"]).strip() if raw.get("toss_winner") else None),
        elected_to=(str(raw["elected_to"]).strip() if raw.get("elected_to") else None),
        venue=(str(raw["venue"]).strip() if raw.get("venue") else None),
        match_id=(str(raw["match_id"]).strip() if raw.get("match_id") else None),
        notes=(str(raw["notes"]).strip() if raw.get("notes") else None),
        raw=raw,
    )


def _as_str_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    raise ValueError("Expected list or comma-separated string for XI")


def _psych_float(raw: dict[str, Any], key: str, default: float) -> float:
    v = raw.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def payload_to_match_context(payload: MatchPayload) -> MatchContext | None:
    from iplpred.core.match_context import parse_toss_elected

    raw = payload.raw
    d1 = _clip(_psych_float(raw, "team1_desperation", 0.0), 0.0, 1.0)
    d2 = _clip(_psych_float(raw, "team2_desperation", 0.0), 0.0, 1.0)
    chaos = _clip(_psych_float(raw, "match_chaos", 1.0), 0.0, 3.0)

    bf = payload.batting_first
    if payload.toss_winner and payload.elected_to and payload.team1_name and payload.team2_name:
        r = parse_toss_elected(
            payload.toss_winner,
            payload.elected_to,
            payload.team1_name,
            payload.team2_name,
        )
        if r is not None:
            bf = r
    has_psych = (d1 > 0.0 or d2 > 0.0 or chaos != 1.0) or any(
        k in raw for k in ("team1_desperation", "team2_desperation", "match_chaos")
    )
    if bf is None and not payload.pitch_text and not payload.venue and not has_psych:
        return None
    return MatchContext(
        batting_first=bf,
        pitch_text=payload.pitch_text,
        venue=payload.venue,
        team1_desperation=d1,
        team2_desperation=d2,
        match_chaos=chaos,
    )
