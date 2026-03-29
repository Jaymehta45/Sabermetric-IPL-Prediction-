"""
IPL 2026 squad loader, player matching, and playing-XI validation (overseas cap).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final

import pandas as pd

from iplpred.paths import REPO_ROOT

IPL_2026_SQUADS_PATH = REPO_ROOT / "data" / "processed" / "ipl_2026_squads.csv"

# Historical player_id (in player_features / match_stats) -> squad CSV player_id
PLAYER_ID_ALIASES: dict[tuple[str, str], str] = {
    ("mumbai indians", "rg sharma"): "Rohit Sharma",
    ("mumbai indians", "sa yadav"): "Suryakumar Yadav",
    ("mumbai indians", "q de kock"): "Quinton de Kock",
    ("mumbai indians", "hh pandya"): "Hardik Pandya",
    ("mumbai indians", "jj bumrah"): "Jasprit Bumrah",
    ("mumbai indians", "rd chahar"): "Deepak Chahar",
    ("chennai super kings", "rd gaikwad"): "Ruturaj Gaikwad",
    ("chennai super kings", "s dube"): "Shivam Dube",
    # Stats CSV `player_id` -> IPL 2026 squad CSV name (Royal Challengers Bengaluru)
    ("royal challengers bengaluru", "v kohli"): "Virat Kohli",
    ("royal challengers bengaluru", "pd salt"): "Phil Salt",
    ("royal challengers bengaluru", "rm patidar"): "Rajat Patidar",
    ("royal challengers bengaluru", "jm sharma"): "Jitesh Sharma",
    ("royal challengers bengaluru", "th david"): "Tim David",
    ("royal challengers bengaluru", "r shepherd"): "Romario Shepherd",
    ("royal challengers bengaluru", "kh pandya"): "Krunal Pandya",
    ("royal challengers bengaluru", "b kumar"): "Bhuvneshwar Kumar",
    ("royal challengers bengaluru", "jr hazlewood"): "Josh Hazlewood",
    ("royal challengers bengaluru", "rasikh salam"): "Rasikh Salam",
    ("royal challengers bengaluru", "suyash sharma"): "Suyash Sharma",
    # Sunrisers Hyderabad
    ("sunrisers hyderabad", "tm head"): "Travis Head",
    ("sunrisers hyderabad", "h klaasen"): "Heinrich Klaasen",
    ("sunrisers hyderabad", "nithish kumar reddy"): "Nitish Kumar Reddy",
    ("sunrisers hyderabad", "hv patel"): "Harshal Patel",
    ("sunrisers hyderabad", "jd unadkat"): "Jaydev Unadkat",
    ("sunrisers hyderabad", "e malinga"): "Eshan Malinga",
}

MAX_OVERSEAS_PLAYING_XI = 4

FUZZY_THRESHOLD: Final[float] = 85.0

try:
    from rapidfuzz import fuzz as _rfuzz

    def _similarity(a: str, b: str) -> float:
        return float(_rfuzz.ratio(a.strip(), b.strip()))

except ImportError:
    from difflib import SequenceMatcher

    def _similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio() * 100.0


# (team_key_lower, player_id_lower) -> canonical squad player_id or None
_resolve_cache: dict[tuple[str, str], str | None] = {}


def clear_resolve_cache() -> None:
    """Clear cached name resolutions (e.g. after squad CSV reload in tests)."""
    _resolve_cache.clear()


def parse_name_signature(name: str) -> tuple[str, str] | None:
    """
    Extract (first_initial, last_name) for signature filtering.

    - first_initial: first character of the first whitespace-separated token (lowercase)
    - last_name: last token, lowercased (exact string for comparison)

    Example: "RG Sharma" -> ('r', 'sharma')
    """
    name = name.strip()
    if not name:
        return None
    parts = name.split()
    if len(parts) < 2:
        return None
    first_tok, last_tok = parts[0], parts[-1]
    if not first_tok or not last_tok:
        return None
    return (first_tok[0].lower(), last_tok.lower())


def filter_squad_candidates_by_signature(raw: str, squad_names: list[str]) -> list[str]:
    """Squad players whose signature (first initial + last name) matches `raw` exactly."""
    sig = parse_name_signature(raw)
    if sig is None:
        return []
    out: list[str] = []
    for c in squad_names:
        c = c.strip()
        if not c:
            continue
        s = parse_name_signature(c)
        if s is not None and s == sig:
            out.append(c)
    return out


def resolve_name_fuzzy(name: str, squad_names: list[str], *, threshold: float = FUZZY_THRESHOLD) -> str | None:
    """
    Signature filter first (same first initial + same last name), then fuzzy on survivors.

    - If no squad row passes the signature filter → None (no fuzzy).
    - If more than one candidate passes → None and [AMBIGUOUS] log.
    - If exactly one candidate → fuzzy ratio must be strictly > threshold to match.
    """
    name = name.strip()
    if not name or not squad_names:
        return None

    candidates = filter_squad_candidates_by_signature(name, squad_names)
    if len(candidates) == 0:
        return None
    if len(candidates) >= 2:
        print(f"[AMBIGUOUS] {name} → multiple candidates: {candidates}")
        return None

    cand = candidates[0]
    s = _similarity(name, cand)
    if s > threshold:
        return cand
    return None


@lru_cache(maxsize=1)
def load_ipl_2026_squads() -> pd.DataFrame:
    if not IPL_2026_SQUADS_PATH.exists():
        return pd.DataFrame(columns=["team", "player_id", "is_overseas"])
    df = pd.read_csv(IPL_2026_SQUADS_PATH, low_memory=False)
    df["team"] = df["team"].astype(str).str.strip()
    df["player_id"] = df["player_id"].astype(str).str.strip()
    df["is_overseas"] = pd.to_numeric(df["is_overseas"], errors="coerce").fillna(0).astype(int)
    return df


def _team_key(team: str) -> str:
    return team.strip().lower()


def squad_rows_for_team(team_name: str) -> pd.DataFrame:
    df = load_ipl_2026_squads()
    if df.empty:
        return df
    tk = _team_key(team_name)
    return df[df["team"].str.strip().str.lower() == tk]


def resolve_to_squad_player_id(team_name: str, player_id: str) -> str | None:
    """
    Map a stats/model player_id to the canonical squad CSV name.

    Order: (1) exact case-insensitive match, (2) alias map,
    (3) signature filter + fuzzy on filtered candidates (similarity strictly > 85).
    Results are cached per (team, player_id).
    """
    tk = _team_key(team_name)
    raw = player_id.strip()
    lk = raw.lower()
    cache_key = (tk, lk)
    if cache_key in _resolve_cache:
        return _resolve_cache[cache_key]

    sub = squad_rows_for_team(team_name)
    if sub.empty:
        _resolve_cache[cache_key] = None
        return None

    squad_names = sub["player_id"].astype(str).str.strip().tolist()

    # 1) Exact match (case-insensitive)
    m = sub[sub["player_id"].str.strip().str.lower() == lk]
    if len(m):
        out = str(m.iloc[0]["player_id"])
        _resolve_cache[cache_key] = out
        return out

    # 2) Alias map
    alias = PLAYER_ID_ALIASES.get((tk, lk))
    if alias:
        m2 = sub[sub["player_id"].str.strip().str.lower() == alias.lower()]
        if len(m2):
            out = str(m2.iloc[0]["player_id"])
            _resolve_cache[cache_key] = out
            return out

    # 3) Signature filter + fuzzy (only when exactly one candidate passes signature)
    fuzzy = resolve_name_fuzzy(raw, squad_names, threshold=FUZZY_THRESHOLD)
    if fuzzy is not None:
        print(f"[FUZZY MATCH] {raw} → {fuzzy}")
        _resolve_cache[cache_key] = fuzzy
        return fuzzy

    _resolve_cache[cache_key] = None
    return None


def is_in_squad(team_name: str, player_id: str) -> bool:
    return resolve_to_squad_player_id(team_name, player_id) is not None


def is_overseas(team_name: str, player_id: str) -> bool:
    canon = resolve_to_squad_player_id(team_name, player_id)
    if canon is None:
        return False
    sub = squad_rows_for_team(team_name)
    row = sub[sub["player_id"].str.strip().str.lower() == canon.lower()]
    if row.empty:
        return False
    return int(row.iloc[0]["is_overseas"]) == 1


def count_overseas(team_name: str, player_ids: list[str]) -> int:
    return sum(1 for p in player_ids if is_overseas(team_name, p))


def validate_team_playing_xi(
    team_name: str,
    player_ids: list[str],
    *,
    max_overseas: int = MAX_OVERSEAS_PLAYING_XI,
) -> None:
    """
    Raises ValueError if any player is not in the IPL 2026 squad or overseas count exceeds cap.
    """
    df = load_ipl_2026_squads()
    if df.empty:
        raise ValueError(f"Missing squad file: {IPL_2026_SQUADS_PATH}")
    missing: list[str] = []
    for p in player_ids:
        if resolve_to_squad_player_id(team_name, p) is None:
            missing.append(p.strip())
    if missing:
        raise ValueError(
            f"Players not in IPL 2026 squad for {team_name!r}: {missing}"
        )
    oc = count_overseas(team_name, player_ids)
    if oc > max_overseas:
        raise ValueError(
            f"Overseas players {oc} exceed limit {max_overseas} for {team_name!r}"
        )


def filter_match_stats_to_squad(team_df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    """Keep rows whose player_id resolves to an IPL 2026 squad member."""
    df = load_ipl_2026_squads()
    if df.empty or squad_rows_for_team(team_name).empty:
        return team_df
    mask = team_df["player_id"].map(
        lambda pid: resolve_to_squad_player_id(team_name, str(pid)) is not None
    )
    return team_df.loc[mask].copy()


def print_integration_complete() -> None:
    print("IPL 2026 squad integration complete")


if __name__ == "__main__":
    _ = load_ipl_2026_squads()
    print_integration_complete()
