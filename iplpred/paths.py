"""Repository root and standard directories (data/, models/)."""

from __future__ import annotations

from pathlib import Path

# iplpred/paths.py -> package dir is parent of this file
_PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _PACKAGE_DIR.parent

DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = REPO_ROOT / "models"
EXAMPLES_DIR = REPO_ROOT / "examples"
# Ball-by-ball CSV / Cricsheet JSON (e.g. one file per IPL match); scanned by build_unified_dataset
MATCHES_DIR = REPO_ROOT / "matches"
