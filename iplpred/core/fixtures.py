"""
IPL 2026 schedule from data/processed/ipl_2026_fixtures.csv.

Use for pre-match defaults (date, home/away, venue hint) when building payloads
or validating user input. Schedule may change per BCCI notice.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from iplpred.paths import PROCESSED_DIR

FIXTURES_PATH = PROCESSED_DIR / "ipl_2026_fixtures.csv"

# City label in schedule -> common stadium string used in historical ball-by-ball
VENUE_CITY_TO_STADIUM: dict[str, str] = {
    "Bengaluru": "M Chinnaswamy Stadium, Bengaluru",
    "Mumbai": "Wankhede Stadium, Mumbai",
    "Chennai": "M. A. Chidambaram Stadium, Chennai",
    "Kolkata": "Eden Gardens, Kolkata",
    "Delhi": "Arun Jaitley Stadium, Delhi",
    "Ahmedabad": "Narendra Modi Stadium, Ahmedabad",
    "Hyderabad": "Rajiv Gandhi International Stadium, Hyderabad",
    "Jaipur": "Sawai Mansingh Stadium, Jaipur",
    "Lucknow": "BRSABV Ekana Cricket Stadium, Lucknow",
    "Guwahati": "Barsapara Cricket Stadium, Guwahati",
    "New Chandigarh": "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur",
    "Raipur": "Shaheed Veer Narayan Singh International Stadium, Raipur",
    "Dharamshala": "Himachal Pradesh Cricket Association Stadium, Dharamshala",
}


@lru_cache(maxsize=1)
def load_fixtures() -> pd.DataFrame:
    if not FIXTURES_PATH.is_file():
        return pd.DataFrame()
    df = pd.read_csv(FIXTURES_PATH, low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def venue_hint(city: str) -> str:
    """Best-effort stadium string for features / match context; fallback is city."""
    c = (city or "").strip()
    return VENUE_CITY_TO_STADIUM.get(c, c)


def get_match(match_no: int) -> dict | None:
    df = load_fixtures()
    if df.empty or "match_no" not in df.columns:
        return None
    row = df.loc[df["match_no"] == match_no]
    if row.empty:
        return None
    r = row.iloc[0].to_dict()
    vc = str(r.get("venue_city", "") or "")
    r["venue_canonical"] = venue_hint(vc)
    return r


def matches_on_date(iso_date: str) -> pd.DataFrame:
    """YYYY-MM-DD — may return 0–2 rows (double-headers)."""
    df = load_fixtures()
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    d = pd.to_datetime(iso_date).date()
    return df[df["date"].dt.date == d]


def home_away_for_match(match_no: int) -> tuple[str, str] | None:
    m = get_match(match_no)
    if m is None:
        return None
    return str(m.get("home_team", "")), str(m.get("away_team", ""))
