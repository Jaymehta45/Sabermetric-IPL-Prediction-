#!/usr/bin/env python3
"""
Check that a ball-by-ball CSV has columns required for build_player_match_stats.

Usage:
  python scripts/validate_bbb_csv.py path/to/unified_ball_by_ball.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REQUIRED = [
    "match_id",
    "date",
    "venue",
    "batting_team",
    "bowling_team",
    "batter",
    "bowler",
    "runs_off_bat",
    "extras",
    "is_wide",
    "is_noball",
    "dismissal_kind",
    "player_dismissed",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str)
    args = parser.parse_args()
    p = Path(args.csv_path)
    if not p.is_file():
        raise SystemExit(f"File not found: {p}")
    import pandas as pd

    df = pd.read_csv(p, nrows=0)
    cols = set(df.columns.str.strip())
    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        print("MISSING columns:", missing)
        sys.exit(1)
    print("OK: all required columns present:", REQUIRED)


if __name__ == "__main__":
    main()
