#!/usr/bin/env python3
"""Look up IPL 2026 schedule rows from data/processed/ipl_2026_fixtures.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iplpred.core.fixtures import get_match, load_fixtures, matches_on_date, venue_hint


def main() -> None:
    p = argparse.ArgumentParser(description="Show IPL 2026 fixture by match no or date")
    p.add_argument("--match", type=int, default=None, help="Match number 1–70")
    p.add_argument("--date", type=str, default=None, help="ISO date YYYY-MM-DD")
    args = p.parse_args()

    if args.match is not None:
        m = get_match(args.match)
        if m is None:
            raise SystemExit(f"No fixture for match_no={args.match}")
        print(f"Match {m['match_no']}: {m['date'].date()} {m['day_of_week']} {m['start_time']}")
        print(f"  {m['home_team']} (home) vs {m['away_team']} (away)")
        print(f"  City: {m['venue_city']}")
        print(f"  Venue hint: {m['venue_canonical']}")
        return

    if args.date is not None:
        df = matches_on_date(args.date)
        if df.empty:
            raise SystemExit(f"No matches on {args.date}")
        print(df.to_string(index=False))
        return

    df = load_fixtures()
    if df.empty:
        raise SystemExit("No fixtures CSV; see data/processed/ipl_2026_fixtures.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
