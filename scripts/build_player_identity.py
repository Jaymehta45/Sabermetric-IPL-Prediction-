#!/usr/bin/env python3
"""
Build data/processed/player_identity.csv and squad_stats_bridge.csv from:

- ipl_2026_squads.csv (team + squad display name)
- player_aliases.csv (alias_string -> canonical_id)
- optional data/processed/player_identity_seed.csv overrides (team, squad_player_name, stats_player_id)

The bridge file replaces hand-maintained PLAYER_ID_ALIASES in squad_utils:
(franchise_key, stats_player_id.lower()) -> squad CSV player_id string.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iplpred.core.name_resolution import load_alias_lookup, normalize_key
from iplpred.paths import PROCESSED_DIR

SQUADS_PATH = PROCESSED_DIR / "ipl_2026_squads.csv"
IDENTITY_OUT = PROCESSED_DIR / "player_identity.csv"
BRIDGE_OUT = PROCESSED_DIR / "squad_stats_bridge.csv"
SEED_PATH = PROCESSED_DIR / "player_identity_seed.csv"


def _team_key(team: str) -> str:
    return team.strip().lower()


def main() -> None:
    if not SQUADS_PATH.is_file():
        raise SystemExit(f"Missing {SQUADS_PATH}")

    squads = pd.read_csv(SQUADS_PATH, low_memory=False)
    squads["team"] = squads["team"].astype(str).str.strip()
    squads["player_id"] = squads["player_id"].astype(str).str.strip()

    alias_lu = load_alias_lookup()
    seed_overrides: dict[tuple[str, str], str] = {}
    if SEED_PATH.is_file():
        sd = pd.read_csv(SEED_PATH, low_memory=False)
        for _, r in sd.iterrows():
            tk = _team_key(str(r["team"]))
            sk = normalize_key(str(r["squad_player_name"]))
            seed_overrides[(tk, sk)] = str(r["stats_player_id"]).strip()

    rows: list[dict] = []
    for _, r in squads.iterrows():
        team = str(r["team"]).strip()
        squad_name = str(r["player_id"]).strip()
        tk = _team_key(team)
        sk = normalize_key(squad_name)

        method = "alias"
        stats_id = seed_overrides.get((tk, sk))
        if stats_id is None:
            stats_id = alias_lu.get(sk)
        if stats_id is None:
            # Also try matching canonical_id == squad_name (already stats-shaped)
            if sk in {normalize_key(k) for k in alias_lu.values()}:
                stats_id = squad_name  # rare
                method = "passthrough"
            else:
                stats_id = squad_name
                method = "display_as_id"

        rows.append(
            {
                "team": team,
                "squad_player_name": squad_name,
                "stats_player_id": stats_id,
                "resolution_method": method,
            }
        )

    idf = pd.DataFrame(rows)
    idf = idf.drop_duplicates(subset=["team", "squad_player_name"], keep="last")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    idf.to_csv(IDENTITY_OUT, index=False)
    print(f"Wrote {len(idf)} rows -> {IDENTITY_OUT}")

    bridge_rows: list[dict] = []
    for _, r in idf.iterrows():
        tk = _team_key(str(r["team"]))
        sid = str(r["stats_player_id"]).strip()
        sq = str(r["squad_player_name"]).strip()
        bridge_rows.append(
            {
                "franchise_key": tk,
                "stats_player_id": sid,
                "stats_player_id_norm": sid.lower(),
                "squad_player_name": sq,
            }
        )
    br = pd.DataFrame(bridge_rows)
    br = br.drop_duplicates(subset=["franchise_key", "stats_player_id_norm"], keep="last")
    br.to_csv(BRIDGE_OUT, index=False)
    print(f"Wrote {len(br)} rows -> {BRIDGE_OUT}")


if __name__ == "__main__":
    main()
