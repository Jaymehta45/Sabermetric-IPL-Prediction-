#!/usr/bin/env python3
"""
Rebuild data/processed/player_registry.csv from player_features.csv and merge
player_aliases_seed.csv into player_aliases.csv (deduped).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iplpred.paths import PROCESSED_DIR

FEATURES_PATH = PROCESSED_DIR / "player_features.csv"
REGISTRY_OUT = PROCESSED_DIR / "player_registry.csv"
ALIASES_OUT = PROCESSED_DIR / "player_aliases.csv"
SEED_PATH = PROCESSED_DIR / "player_aliases_seed.csv"


def main() -> None:
    if not FEATURES_PATH.is_file():
        raise SystemExit(f"Missing {FEATURES_PATH}; run build_features.py first.")

    feat = pd.read_csv(FEATURES_PATH, low_memory=False)
    if "player_id" not in feat.columns:
        raise SystemExit("player_features.csv has no player_id column.")

    ids = set(feat["player_id"].astype(str).str.strip().unique())
    seed = None
    if SEED_PATH.is_file():
        seed = pd.read_csv(SEED_PATH, low_memory=False)
        if "canonical_id" in seed.columns:
            ids.update(seed["canonical_id"].astype(str).str.strip().unique())
    ids_sorted = sorted(ids)
    reg = pd.DataFrame(
        {"canonical_id": ids_sorted, "primary_name": ids_sorted, "notes": ""},
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    reg.to_csv(REGISTRY_OUT, index=False)
    print(f"Wrote {len(reg)} rows -> {REGISTRY_OUT}")

    base_rows = [
        {"canonical_id": c, "alias_string": c, "source": "stats_player_id", "priority": 0}
        for c in ids_sorted
    ]
    aliases = pd.DataFrame(base_rows)

    if seed is not None:
        required = {"canonical_id", "alias_string", "source", "priority"}
        if not required.issubset(seed.columns):
            raise SystemExit(f"{SEED_PATH} needs columns {sorted(required)}")
        aliases = pd.concat([aliases, seed[list(required)]], ignore_index=True)

    aliases = aliases.drop_duplicates(
        subset=["canonical_id", "alias_string"],
        keep="last",
    )
    aliases = aliases.sort_values(
        ["canonical_id", "priority", "alias_string"],
        kind="stable",
    ).reset_index(drop=True)
    aliases.to_csv(ALIASES_OUT, index=False)
    print(f"Wrote {len(aliases)} rows -> {ALIASES_OUT}")


if __name__ == "__main__":
    main()
