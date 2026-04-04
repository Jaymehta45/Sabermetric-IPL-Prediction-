"""
Resolve display / full names to stats ``player_id`` using player_aliases.csv.

Used by ICC ingest and identity build — single path for name → canonical_id.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from iplpred.paths import PROCESSED_DIR

ALIASES_PATH = PROCESSED_DIR / "player_aliases.csv"


@lru_cache(maxsize=1)
def load_alias_lookup() -> dict[str, str]:
    """
    Map normalized alias_string (lower, stripped) -> canonical_id (stats player_id).
    When duplicate alias_strings exist, highest priority wins.
    """
    if not ALIASES_PATH.is_file():
        return {}
    df = pd.read_csv(ALIASES_PATH, low_memory=False)
    if df.empty or "canonical_id" not in df.columns:
        return {}
    df = df.copy()
    df["canonical_id"] = df["canonical_id"].astype(str).str.strip()
    df["alias_string"] = df["alias_string"].astype(str).str.strip()
    if "priority" not in df.columns:
        df["priority"] = 0
    df["priority"] = pd.to_numeric(df["priority"], errors="coerce").fillna(0)
    df["_key"] = df["alias_string"].str.lower()
    df = df.sort_values(["_key", "priority", "alias_string"], ascending=[True, False, True])
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        k = str(row["_key"]).strip()
        if not k:
            continue
        if k not in out:
            out[k] = str(row["canonical_id"]).strip()
    return out


def normalize_key(s: str) -> str:
    return str(s or "").strip().lower()


def resolve_display_to_stats_id(name: str, *, fallback_to_display: bool = True) -> str:
    """
    Map a scorecard or squad display name to stats ``player_id``.

    If no alias matches and ``fallback_to_display`` is True, returns stripped
    ``name`` so new players can use display-as-id + prior fill.
    """
    raw = str(name or "").strip()
    if not raw:
        return ""
    lu = load_alias_lookup()
    sid = lu.get(normalize_key(raw))
    if sid:
        return sid
    return raw if fallback_to_display else ""


def clear_alias_lookup_cache() -> None:
    load_alias_lookup.cache_clear()
