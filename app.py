"""
Vercel Fluid / zero-config entry: exposes ``app`` at a supported filename (``app.py``).

Local: ``uvicorn app:app`` or ``uvicorn web.dashboard:app``
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Vercel runtime sometimes omits vercel.json env; repo + branch enable GitHub API + SHA-raw fetch.
os.environ.setdefault("PREDICTION_LOG_GITHUB_REPO", "Jaymehta45/Sabermetric-IPL-Prediction-")
os.environ.setdefault("PREDICTION_LOG_GITHUB_BRANCH", "main")

from web.dashboard import app

__all__ = ["app"]
