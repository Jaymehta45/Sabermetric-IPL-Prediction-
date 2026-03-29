#!/usr/bin/env python3
"""
Convert ICC T20 WC 2026 scorecard summaries (data/icc-t20-wc-2026/key_scorecards.csv)
into synthetic ball-by-ball rows and append to unified_ball_by_ball.csv.

Each real delivery is approximated by distributing runs across balls faced (integer split).
Bowlers are parsed from dismissal text when possible (e.g. "c X b Arshdeep").

Re-running skips matches already present: numeric match_id = 900000000 + match_no
(from matches.csv) so build_features.py (int match_id) stays valid.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iplpred.paths import DATA_DIR, PROCESSED_DIR

ICC_DIR = DATA_DIR / "icc-t20-wc-2026"
MATCHES_PATH = ICC_DIR / "matches.csv"
SCORECARDS_PATH = ICC_DIR / "key_scorecards.csv"
UNIFIED_PATH = PROCESSED_DIR / "unified_ball_by_ball.csv"

FINAL_COLUMNS = [
    "match_id",
    "date",
    "venue",
    "batting_team",
    "bowling_team",
    "batter",
    "bowler",
    "non_striker",
    "runs_off_bat",
    "extras",
    "is_wide",
    "is_noball",
    "dismissal_kind",
    "player_dismissed",
]


def distribute_runs(runs: int, balls: int) -> list[int]:
    if balls <= 0:
        return []
    base = runs // balls
    rem = runs % balls
    return [base + (1 if i < rem else 0) for i in range(balls)]


def dismissal_kind_for_row(dismissal: str, is_last_ball: bool) -> str:
    """Non-empty only on wicket ball; must match build_player_match_stats wicket logic."""
    if not is_last_ball:
        return ""
    s = (dismissal or "").strip()
    if not s or s.lower() == "not out":
        return ""
    if "run out" in s.lower():
        return "run out"
    return "caught"


def extract_bowler(dismissal: str) -> str:
    s = (dismissal or "").strip()
    if not s or s.lower() == "not out":
        return "Unknown"
    if "run out" in s.lower():
        return "Unknown"
    if "retired" in s.lower():
        return "Unknown"
    # c & b Pandya -> b Pandya
    parts = re.findall(r"\bb\s+([^,|]+?)(?:\s*$)", s, re.I)
    if parts:
        return parts[-1].strip()
    parts2 = re.findall(r"\bb\s+([^,|]+)", s, re.I)
    if parts2:
        return parts2[-1].strip()
    return "Unknown"


def opponent(team: str, t1: str, t2: str) -> str:
    if team == t1:
        return t2
    if team == t2:
        return t1
    return ""


def main() -> None:
    if not SCORECARDS_PATH.is_file() or not MATCHES_PATH.is_file():
        raise SystemExit(f"Missing {ICC_DIR} files")

    matches = pd.read_csv(MATCHES_PATH, low_memory=False)
    # group column holds SF1, SF2, Final, etc.
    gcol = "group" if "group" in matches.columns else None
    if gcol is None:
        raise SystemExit("matches.csv needs a 'group' column")

    match_meta: dict[str, dict] = {}
    for _, row in matches.iterrows():
        key = str(row[gcol]).strip()
        if not key or key.lower() == "nan":
            continue
        mno = int(row["match_no"]) if pd.notna(row.get("match_no")) else 0
        # High-range int so it does not collide with Cricsheet numeric ids; build_features needs int match_id.
        mid = 900_000_000 + mno
        match_meta[key] = {
            "match_id": mid,
            "match_no": mno,
            "date": row.get("date", ""),
            "venue": str(row.get("venue", "")).strip(),
            "team1": str(row.get("team1", "")).strip(),
            "team2": str(row.get("team2", "")).strip(),
        }

    sc = pd.read_csv(SCORECARDS_PATH, low_memory=False)
    existing = pd.read_csv(UNIFIED_PATH, low_memory=False, nrows=0)
    if "match_id" in existing.columns:
        existing_full = pd.read_csv(UNIFIED_PATH, usecols=["match_id"], low_memory=False)
        existing_ids = set(existing_full["match_id"].astype(str).unique()) | set(
            pd.to_numeric(existing_full["match_id"], errors="coerce").dropna().astype(int).unique()
        )
        existing_ids = {str(x) for x in existing_ids}
    else:
        existing_ids = set()

    new_rows: list[dict] = []
    for _, row in sc.iterrows():
        mkey = str(row["match"]).strip()
        meta = match_meta.get(mkey)
        if meta is None:
            print(f"WARN: no match row for group={mkey!r}, skipping")
            continue
        mid = meta["match_id"]
        if str(mid) in existing_ids:
            print(f"Skip (already in unified): {mid}")
            continue

        team = str(row["team"]).strip()
        player = str(row["player"]).strip()
        runs = int(pd.to_numeric(row["runs"], errors="coerce") or 0)
        balls = int(pd.to_numeric(row["balls"], errors="coerce") or 0)
        dismiss = str(row.get("dismissal", "")).strip()

        t1, t2 = meta["team1"], meta["team2"]
        opp = opponent(team, t1, t2)
        if not opp:
            print(f"WARN: team {team} not in {t1}/{t2} for {mid}")
            continue

        bat_team = team
        bowl_team = opp
        bowler_name = extract_bowler(dismiss)

        if balls <= 0:
            continue

        per_ball = distribute_runs(runs, balls)
        dismissal_lower = dismiss.lower()
        for i, ro in enumerate(per_ball):
            is_last = i == balls - 1
            dk = dismissal_kind_for_row(dismiss, is_last)
            player_out = player if dk else ""
            new_rows.append(
                {
                    "match_id": mid,
                    "date": meta["date"],
                    "venue": meta["venue"],
                    "batting_team": bat_team,
                    "bowling_team": bowl_team,
                    "batter": player,
                    "bowler": bowler_name,
                    "non_striker": f"{player}_{i}",
                    "runs_off_bat": float(ro),
                    "extras": 0.0,
                    "is_wide": 0,
                    "is_noball": 0,
                    "dismissal_kind": dk,
                    "player_dismissed": player_out,
                }
            )

    if not new_rows:
        print("No new rows to add (all icc2026wc match_ids present or no scorecard rows).")
        return

    add = pd.DataFrame(new_rows)
    for c in FINAL_COLUMNS:
        if c not in add.columns:
            add[c] = pd.NA
    add = add[FINAL_COLUMNS]

    if UNIFIED_PATH.is_file():
        old = pd.read_csv(UNIFIED_PATH, low_memory=False)
        for c in FINAL_COLUMNS:
            if c not in old.columns:
                old[c] = pd.NA
        old = old[FINAL_COLUMNS]
        out = pd.concat([old, add], ignore_index=True)
    else:
        out = add

    out.to_csv(UNIFIED_PATH, index=False)
    print(f"Appended {len(add)} rows to {UNIFIED_PATH}")
    print("Next: python build_player_match_stats.py && python build_features.py (then train_player_model.py if needed)")


if __name__ == "__main__":
    main()
