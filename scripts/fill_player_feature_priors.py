#!/usr/bin/env python3
"""Append prior feature rows for identity players not yet in player_features. See iplpred.pipeline.feature_priors."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iplpred.pipeline.feature_priors import append_squad_priors


def main() -> None:
    nf, nm = append_squad_priors()
    print(f"Appended prior features for {nf} players; match_stats rows +{nm}.")


if __name__ == "__main__":
    main()
