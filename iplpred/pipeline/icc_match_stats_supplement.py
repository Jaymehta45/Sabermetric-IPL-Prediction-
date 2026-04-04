"""
Supplement player_match_stats with ICC T20 WC **batting** lines from key_scorecards.csv.

Names map through player_aliases (name_resolution.resolve_display_to_stats_id).
"""

from __future__ import annotations

import pandas as pd

from iplpred.core.name_resolution import resolve_display_to_stats_id
from iplpred.paths import DATA_DIR

ICC_DIR = DATA_DIR / "icc-t20-wc-2026"
KEY_SC_PATH = ICC_DIR / "key_scorecards.csv"

_BASE_MID = 920_000_000


def _match_code_to_mid(code: str, innings: str, team: str) -> int:
    h = abs(hash((code, innings, team)))
    return _BASE_MID + (h % 9_000_000)


def _parse_dismissed(dismissal: str) -> float:
    s = str(dismissal or "").lower()
    if "not out" in s:
        return 0.0
    return 1.0


def build_icc_supplement_df() -> pd.DataFrame | None:
    if not KEY_SC_PATH.is_file():
        return None
    df = pd.read_csv(KEY_SC_PATH, low_memory=False)
    if df.empty or "player" not in df.columns:
        return None

    rows: list[dict] = []
    for _, r in df.iterrows():
        match = str(r.get("match", "")).strip()
        innings = str(r.get("innings", "")).strip()
        team = str(r.get("team", "")).strip()
        player = str(r.get("player", "")).strip()
        if not player:
            continue

        pid = resolve_display_to_stats_id(player)
        runs = float(pd.to_numeric(r.get("runs"), errors="coerce") or 0)
        balls = float(pd.to_numeric(r.get("balls"), errors="coerce") or 0)
        fours = float(pd.to_numeric(r.get("fours"), errors="coerce") or 0)
        sixes = float(pd.to_numeric(r.get("sixes"), errors="coerce") or 0)
        dismissed = _parse_dismissed(str(r.get("dismissal", "")))

        mid = _match_code_to_mid(match, innings, team)
        d = pd.Timestamp("2026-02-15")
        venue = "ICC T20 WC 2026"
        match_key = f"{mid}_{d.strftime('%Y-%m-%d')}"

        rows.append(
            {
                "match_key": match_key,
                "match_id": mid,
                "player_id": pid,
                "date": d,
                "venue": venue,
                "batting_team": team,
                "bowling_team": "unknown",
                "runs": runs,
                "balls": balls,
                "fours": fours,
                "sixes": sixes,
                "dismissed": dismissed,
                "wickets": 0.0,
                "balls_bowled": 0.0,
                "runs_conceded": 0.0,
                "total_balls_faced": balls,
                "total_balls_bowled": 0.0,
                "role": "batter",
            }
        )

    if not rows:
        return None
    return pd.DataFrame(rows)
