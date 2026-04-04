"""
Load (franchise_key, stats_player_id) -> IPL 2026 squad CSV player name.

Built by scripts/build_player_identity.py -> data/processed/squad_stats_bridge.csv
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from iplpred.paths import PROCESSED_DIR

BRIDGE_PATH = PROCESSED_DIR / "squad_stats_bridge.csv"


@lru_cache(maxsize=1)
def load_squad_stats_bridge() -> dict[tuple[str, str], str]:
    """
    Key: (franchise_key lower, stats_player_id lower) -> squad_player_name
    as in ipl_2026_squads.csv player_id column.
    """
    if not BRIDGE_PATH.is_file():
        return {}
    df = pd.read_csv(BRIDGE_PATH, low_memory=False)
    if df.empty:
        return {}
    out: dict[tuple[str, str], str] = {}
    for _, r in df.iterrows():
        fk = str(r.get("franchise_key", "")).strip().lower()
        sn = str(r.get("stats_player_id_norm", r.get("stats_player_id", ""))).strip().lower()
        sq = str(r.get("squad_player_name", "")).strip()
        if fk and sn and sq:
            out[(fk, sn)] = sq
    return out


def clear_squad_bridge_cache() -> None:
    load_squad_stats_bridge.cache_clear()
