"""
IPL franchise team names (from squad file) for filtering domestic / league context.
"""

from __future__ import annotations

from pathlib import Path

from iplpred.paths import REPO_ROOT

DEFAULT_SQUADS_PATH = REPO_ROOT / "data" / "processed" / "ipl_2026_squads.csv"


def load_franchise_team_names(squads_path: Path | None = None) -> set[str]:
    """Normalized franchise names as in historical IPL / squad CSV."""
    path = squads_path or DEFAULT_SQUADS_PATH
    if not path.exists():
        return set()
    import pandas as pd

    df = pd.read_csv(path, low_memory=False)
    if "team" not in df.columns:
        return set()
    return {str(t).strip() for t in df["team"].dropna().unique() if str(t).strip()}
