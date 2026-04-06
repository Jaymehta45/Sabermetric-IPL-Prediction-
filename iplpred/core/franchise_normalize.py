"""
Map common abbreviations and historical franchise names to one canonical string
so momentum / history lookups align with match_training_dataset.csv rows.
"""

from __future__ import annotations

# Lowercase key -> canonical display name (match training CSV conventions)
_CANONICAL_BY_LOWER: dict[str, str] = {
    "chennai super kings": "Chennai Super Kings",
    "csk": "Chennai Super Kings",
    "delhi capitals": "Delhi Capitals",
    "dc": "Delhi Capitals",
    "gujarat titans": "Gujarat Titans",
    "gt": "Gujarat Titans",
    "kolkata knight riders": "Kolkata Knight Riders",
    "kkr": "Kolkata Knight Riders",
    "lucknow super giants": "Lucknow Super Giants",
    "lsg": "Lucknow Super Giants",
    "mumbai indians": "Mumbai Indians",
    "mi": "Mumbai Indians",
    "punjab kings": "Punjab Kings",
    "pbks": "Punjab Kings",
    "kings xi punjab": "Punjab Kings",
    "rajasthan royals": "Rajasthan Royals",
    "rr": "Rajasthan Royals",
    "royal challengers bangalore": "Royal Challengers Bangalore",
    "royal challengers bengaluru": "Royal Challengers Bangalore",
    "rcb": "Royal Challengers Bangalore",
    "sunrisers hyderabad": "Sunrisers Hyderabad",
    "srh": "Sunrisers Hyderabad",
}


def canonical_franchise(name: str) -> str:
    """Normalize franchise name for matching; unknown strings returned stripped."""
    n = (name or "").strip()
    if not n:
        return n
    key = n.lower()
    return _CANONICAL_BY_LOWER.get(key, n)
