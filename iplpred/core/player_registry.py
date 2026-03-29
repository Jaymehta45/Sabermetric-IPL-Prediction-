"""
Canonical player identity: registry + aliases + normalized name resolution.

Stats / model `player_id` strings (e.g. V Kohli) are the canonical keys used in
player_features.csv. Aliases map display names (Virat Kohli) to those keys.

If registry files are missing, resolution falls back to passthrough (input = output).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
import pandas as pd

from iplpred.paths import REPO_ROOT

REGISTRY_PATH = REPO_ROOT / "data" / "processed" / "player_registry.csv"
ALIASES_PATH = REPO_ROOT / "data" / "processed" / "player_aliases.csv"

FUZZY_THRESHOLD = 88.0

try:
    from rapidfuzz import fuzz, process

    def _fuzzy_best(query: str, choices: list[str]) -> tuple[str | None, float]:
        if not choices:
            return None, 0.0
        m = process.extractOne(query, choices, scorer=fuzz.WRatio)
        if m is None:
            return None, 0.0
        return m[0], float(m[1])

except ImportError:
    from difflib import get_close_matches

    def _fuzzy_best(query: str, choices: list[str]) -> tuple[str | None, float]:
        if not choices:
            return None, 0.0
        m = get_close_matches(query, choices, n=1, cutoff=0.75)
        if not m:
            return None, 0.0
        return m[0], 85.0


def normalize_name(s: str) -> str:
    """
    Normalize for lookup: NFKC, lower, strip, collapse whitespace,
    remove redundant dots in initials (optional light cleanup).
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    t = unicodedata.normalize("NFKC", str(s)).strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = t.replace(".", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


@dataclass(frozen=True)
class Resolution:
    """Result of resolving an input string to stats `player_id`."""

    input_raw: str
    stats_player_id: str
    method: str  # exact_registry | exact_alias | fuzzy | passthrough
    score: float  # 0-100 for fuzzy; 100 for exact


@lru_cache(maxsize=1)
def _load_registry_ids() -> frozenset[str]:
    if not REGISTRY_PATH.exists():
        return frozenset()
    df = pd.read_csv(REGISTRY_PATH, low_memory=False)
    if "canonical_id" not in df.columns:
        return frozenset()
    return frozenset(df["canonical_id"].astype(str).str.strip().unique())


@lru_cache(maxsize=1)
def _alias_map() -> dict[str, str]:
    """
    normalized_alias -> canonical_id (stats player_id).
    Later rows with higher priority can override if we sort by priority desc.
    """
    if not ALIASES_PATH.exists():
        return {}
    df = pd.read_csv(ALIASES_PATH, low_memory=False)
    required = {"canonical_id", "alias_string"}
    if not required.issubset(df.columns):
        return {}
    if "priority" in df.columns:
        df = df.sort_values("priority", ascending=False)
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        alias = normalize_name(str(row["alias_string"]))
        cid = str(row["canonical_id"]).strip()
        if alias and cid:
            out[alias] = cid
    return out


def clear_registry_cache() -> None:
    _load_registry_ids.cache_clear()
    _alias_map.cache_clear()


def resolve_player_name(raw: str, *, allow_fuzzy: bool = True) -> Resolution:
    """
    Map a human-entered name to the stats `player_id` used in player_features.

    Order: exact canonical → exact alias → fuzzy against all known canonical ids.
    If nothing matches, passthrough trimmed raw string (caller may error later).
    """
    s = str(raw).strip()
    if not s:
        return Resolution("", "", "empty", 0.0)

    reg = _load_registry_ids()
    alias_map = _alias_map()
    norm = normalize_name(s)

    if s in reg:
        return Resolution(s, s, "exact_registry", 100.0)

    if norm in alias_map:
        return Resolution(s, alias_map[norm], "exact_alias", 100.0)

    # Normalized raw might be canonical
    if norm and norm.replace(" ", "") in {x.replace(" ", "").lower() for x in reg}:
        for x in reg:
            if normalize_name(x) == norm:
                return Resolution(s, x, "exact_registry_norm", 100.0)

    if allow_fuzzy and reg:
        best_id, sc = _fuzzy_best(s, list(reg))
        if best_id is not None and sc >= FUZZY_THRESHOLD:
            return Resolution(s, best_id, "fuzzy", sc)

    if allow_fuzzy and alias_map:
        alias_keys = list(alias_map.keys())
        best_key, sc = _fuzzy_best(norm, alias_keys)
        if best_key is not None and sc >= FUZZY_THRESHOLD:
            return Resolution(s, alias_map[best_key], "fuzzy_alias", sc)

    return Resolution(s, s, "passthrough", 50.0)


def resolve_playing_xi(
    names: list[str],
    *,
    allow_fuzzy: bool = True,
) -> tuple[list[str], list[Resolution]]:
    """Resolve a full XI; order preserved."""
    out_ids: list[str] = []
    meta: list[Resolution] = []
    for n in names:
        r = resolve_player_name(n, allow_fuzzy=allow_fuzzy)
        out_ids.append(r.stats_player_id)
        meta.append(r)
    return out_ids, meta
