"""
Load ``data/processed/prediction_log.csv`` from GitHub (raw) at request time or from the local bundle.

On Vercel, ``VERCEL_GIT_REPO_SLUG`` and ``VERCEL_GIT_COMMIT_REF`` are set automatically, so the app
can fetch the CSV from the same branch you deploy — **no redeploy** is needed after you
``git push`` an updated ``prediction_log.csv``.

Override with ``PREDICTION_LOG_URL`` (full https URL to a CSV) or
``PREDICTION_LOG_GITHUB_REPO`` + ``PREDICTION_LOG_GITHUB_BRANCH``.
"""

from __future__ import annotations

import io
import os
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from iplpred.paths import PROCESSED_DIR

LOG_PATH = PROCESSED_DIR / "prediction_log.csv"

# Public repo — used when VERCEL_* slug/ref are missing in the Python runtime.
_DEFAULT_GITHUB_REPO = "Jaymehta45/Sabermetric-IPL-Prediction-"

_CACHE_LOCK = threading.Lock()
_CACHE_T: float | None = None
_CACHE_DF: pd.DataFrame | None = None
_CACHE_META: dict = {}


def _resolved_remote_url() -> str | None:
    explicit = os.environ.get("PREDICTION_LOG_URL", "").strip()
    if explicit:
        return explicit
    slug = os.environ.get("VERCEL_GIT_REPO_SLUG", "").strip()
    ref = os.environ.get("VERCEL_GIT_COMMIT_REF", "").strip()
    if slug and ref:
        # Vercel often sets SLUG to repo name only — raw.githubusercontent.com needs owner/repo.
        if "/" not in slug:
            owner = os.environ.get("VERCEL_GIT_REPO_OWNER", "").strip() or "Jaymehta45"
            slug = f"{owner}/{slug}"
        return (
            f"https://raw.githubusercontent.com/{slug}/{ref}"
            "/data/processed/prediction_log.csv"
        )
    repo = os.environ.get("PREDICTION_LOG_GITHUB_REPO", "").strip()
    branch = os.environ.get("PREDICTION_LOG_GITHUB_BRANCH", "main").strip()
    if repo:
        return (
            f"https://raw.githubusercontent.com/{repo}/{branch}"
            "/data/processed/prediction_log.csv"
        )
    # Vercel: system git env vars are sometimes unset in the serverless runtime.
    if os.environ.get("VERCEL", "").strip() == "1" or bool(
        os.environ.get("VERCEL_ENV", "").strip()
    ):
        ref = (
            os.environ.get("VERCEL_GIT_COMMIT_REF", "").strip()
            or os.environ.get("PREDICTION_LOG_GITHUB_BRANCH", "").strip()
            or "main"
        )
        r = os.environ.get("PREDICTION_LOG_GITHUB_REPO", "").strip() or _DEFAULT_GITHUB_REPO
        return (
            f"https://raw.githubusercontent.com/{r}/{ref}"
            "/data/processed/prediction_log.csv"
        )
    return None


def _cache_ttl_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("PREDICTION_LOG_CACHE_SECONDS", "30")))
    except ValueError:
        return 30.0


def _fetch_url(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "iplpred-web/1.0"})
    with urlopen(req, timeout=20) as resp:
        return resp.read()


def read_prediction_log_dataframe() -> pd.DataFrame | None:
    """
    Return the latest prediction log as a DataFrame, or None if unavailable.

    Prefer remote URL when configured (Vercel auto-URL or env). Falls back to the CSV
    bundled with the deployment. Uses a short in-process cache (see
    ``PREDICTION_LOG_CACHE_SECONDS``).
    """
    global _CACHE_T, _CACHE_DF, _CACHE_META
    ttl = _cache_ttl_seconds()
    now = time.monotonic()
    with _CACHE_LOCK:
        if (
            ttl > 0
            and _CACHE_T is not None
            and _CACHE_DF is not None
            and (now - _CACHE_T) < ttl
        ):
            return _CACHE_DF.copy()

    url = _resolved_remote_url()
    meta: dict = {"source": "local_file", "remote_url": None, "error": None}

    if url:
        try:
            raw = _fetch_url(url)
            df = pd.read_csv(io.BytesIO(raw), low_memory=False)
            meta = {"source": "remote_url", "remote_url": url, "error": None}
            with _CACHE_LOCK:
                _CACHE_T = now
                _CACHE_DF = df
                _CACHE_META = dict(meta)
            return df.copy()
        except (HTTPError, URLError, OSError, TimeoutError, ValueError) as e:
            meta = {
                "source": "remote_failed",
                "remote_url": url,
                "error": repr(e)[:240],
            }

    if LOG_PATH.is_file():
        try:
            df = pd.read_csv(LOG_PATH, low_memory=False)
            if meta.get("error"):
                meta["source"] = "local_fallback"
            else:
                meta["source"] = "local_file"
            meta["remote_url"] = url
            with _CACHE_LOCK:
                _CACHE_T = now
                _CACHE_DF = df
                _CACHE_META = dict(meta)
            return df.copy()
        except Exception as e:
            meta["error"] = (meta.get("error") or "") + f" local_read:{repr(e)[:120]}"

    with _CACHE_LOCK:
        _CACHE_T = now
        _CACHE_DF = None
        _CACHE_META = dict(meta)
    return None


def prediction_log_meta() -> dict:
    """Metadata from the last load (for /api/build-info)."""
    read_prediction_log_dataframe()
    with _CACHE_LOCK:
        out = dict(_CACHE_META)
        if _CACHE_DF is not None:
            out["row_count"] = int(len(_CACHE_DF))
        else:
            out["row_count"] = 0
    out["log_path"] = str(LOG_PATH)
    out["vercel_git_commit_sha"] = os.environ.get("VERCEL_GIT_COMMIT_SHA", "")
    out["vercel_git_commit_ref"] = os.environ.get("VERCEL_GIT_COMMIT_REF", "")
    out["vercel_git_repo_slug"] = os.environ.get("VERCEL_GIT_REPO_SLUG", "")
    return out
