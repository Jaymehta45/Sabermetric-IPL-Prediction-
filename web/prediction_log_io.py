"""
Load ``data/processed/prediction_log.csv`` from GitHub at request time or from the local bundle.

**Important:** ``raw.githubusercontent.com/.../BRANCH/...`` is CDN-cached and can lag **hours** behind
the real ``BRANCH`` tip after a push. That made the dashboard show stale rows even though ``main``
was updated. When ``PREDICTION_LOG_GITHUB_REPO`` is set, we **prefer the GitHub Contents API**
(``GET /repos/{owner}/{repo}/contents/...?ref=branch``), which tracks ``ref`` promptly. We still
fall back to raw URL + local file.

On Vercel, set ``PREDICTION_LOG_GITHUB_REPO`` + ``PREDICTION_LOG_GITHUB_BRANCH`` (see ``vercel.json``).
Optional ``GITHUB_TOKEN`` raises API rate limits (5000/hr vs 60/hr unauthenticated).

Override with ``PREDICTION_LOG_URL`` (full https URL) — used only if repo-based API fetch is skipped
(see ``PREDICTION_LOG_FETCH``).
"""

from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

from iplpred.paths import PROCESSED_DIR

LOG_PATH = PROCESSED_DIR / "prediction_log.csv"

# Public repo — CSV on `main` is the dashboard source of truth (see docs/DAILY_WORKFLOW.txt).
_DEFAULT_GITHUB_REPO = "Jaymehta45/Sabermetric-IPL-Prediction-"

_CACHE_LOCK = threading.Lock()
_CACHE_T: float | None = None
_CACHE_DF: pd.DataFrame | None = None
_CACHE_META: dict = {}


def _bootstrap_vercel_prediction_log_env() -> None:
    """
    Vercel often does not inject ``vercel.json`` ``env`` into the Python serverless runtime.
    Without ``PREDICTION_LOG_GITHUB_REPO``, we would fall back to ``VERCEL_GIT_REPO_SLUG`` (can
    mismatch the real GitHub path) or to the **bundled** CSV (frozen at deploy) after a failed fetch.
    """
    if not (
        os.environ.get("VERCEL", "").strip() == "1"
        or bool(os.environ.get("VERCEL_ENV", "").strip())
    ):
        return
    if not os.environ.get("PREDICTION_LOG_GITHUB_REPO", "").strip():
        os.environ["PREDICTION_LOG_GITHUB_REPO"] = _DEFAULT_GITHUB_REPO
    if not os.environ.get("PREDICTION_LOG_GITHUB_BRANCH", "").strip():
        os.environ["PREDICTION_LOG_GITHUB_BRANCH"] = "main"


_bootstrap_vercel_prediction_log_env()


def _resolved_remote_url() -> str | None:
    explicit = os.environ.get("PREDICTION_LOG_URL", "").strip()
    if explicit:
        return explicit
    # Prefer branch-based raw URL so pushes to GitHub update the live site without redeploying.
    # If we used VERCEL_GIT_COMMIT_REF here, the CSV would be pinned to the deployment commit
    # (immutable); prediction_log.csv updates on main would never appear until a new deploy.
    repo = os.environ.get("PREDICTION_LOG_GITHUB_REPO", "").strip()
    branch = os.environ.get("PREDICTION_LOG_GITHUB_BRANCH", "main").strip()
    if repo:
        return (
            f"https://raw.githubusercontent.com/{repo}/{branch}"
            "/data/processed/prediction_log.csv"
        )
    slug = os.environ.get("VERCEL_GIT_REPO_SLUG", "").strip()
    if slug:
        # Always use a branch name here, never VERCEL_GIT_COMMIT_REF — a deploy-SHA URL would
        # freeze prediction_log.csv at the bundle from that deploy until the next deploy.
        if "/" not in slug:
            owner = os.environ.get("VERCEL_GIT_REPO_OWNER", "").strip() or "Jaymehta45"
            slug = f"{owner}/{slug}"
        br = os.environ.get("PREDICTION_LOG_GITHUB_BRANCH", "main").strip() or "main"
        return (
            f"https://raw.githubusercontent.com/{slug}/{br}"
            "/data/processed/prediction_log.csv"
        )
    # Vercel: system git env vars are sometimes unset in the serverless runtime.
    if os.environ.get("VERCEL", "").strip() == "1" or bool(
        os.environ.get("VERCEL_ENV", "").strip()
    ):
        br = os.environ.get("PREDICTION_LOG_GITHUB_BRANCH", "main").strip() or "main"
        r = os.environ.get("PREDICTION_LOG_GITHUB_REPO", "").strip() or _DEFAULT_GITHUB_REPO
        return (
            f"https://raw.githubusercontent.com/{r}/{br}"
            "/data/processed/prediction_log.csv"
        )
    return None


def _cache_ttl_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("PREDICTION_LOG_CACHE_SECONDS", "30")))
    except ValueError:
        return 30.0


def _fetch_url(url: str) -> bytes:
    # GitHub raw uses CDN Cache-Control ~300s; varying the query string busts stale edge caches.
    sep = "&" if "?" in url else "?"
    bust = int(time.time() // 90)
    busted = f"{url}{sep}iplpred_cb={bust}"
    req = Request(busted, headers={"User-Agent": "iplpred-web/1.0"})
    with urlopen(req, timeout=20) as resp:
        return resp.read()


_DEFAULT_LOG_PATH_IN_REPO = "data/processed/prediction_log.csv"


def _github_repo_owner_name(repo_slug: str) -> tuple[str, str] | None:
    parts = [p for p in str(repo_slug).strip().split("/") if p]
    if len(parts) == 2:
        return parts[0], parts[1]
    return None


def _fetch_github_contents_file(
    owner: str,
    repo: str,
    path_in_repo: str,
    ref: str,
) -> bytes:
    """Latest file bytes for ``ref`` (branch or SHA) via Contents API — avoids stale raw CDN on ``main``."""
    enc_path = quote(path_in_repo, safe="")
    enc_ref = quote(ref, safe="")
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{enc_path}?ref={enc_ref}"
    headers = {
        "User-Agent": "iplpred-web/1.0",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(api_url, headers=headers)
    with urlopen(req, timeout=25) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("contents API: expected object")
    if payload.get("type") != "file":
        raise ValueError("contents API: not a file")
    b64 = str(payload.get("content") or "")
    if not b64:
        raise ValueError("contents API: empty content")
    return base64.b64decode(b64.replace("\n", ""))


def _fetch_mode() -> str:
    return os.environ.get("PREDICTION_LOG_FETCH", "api_then_raw").strip().lower()


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
    meta: dict = {
        "source": "local_file",
        "remote_url": None,
        "github_api_url": None,
        "error": None,
        "bundled_csv_stale_risk": False,
    }

    mode = _fetch_mode()
    repo_full = os.environ.get("PREDICTION_LOG_GITHUB_REPO", "").strip()
    branch = os.environ.get("PREDICTION_LOG_GITHUB_BRANCH", "main").strip() or "main"
    path_in_repo = os.environ.get("PREDICTION_LOG_PATH_IN_REPO", _DEFAULT_LOG_PATH_IN_REPO).strip()

    try_api = mode in ("api_then_raw", "api", "contents", "github_api") and bool(repo_full)
    if try_api:
        own_repo = _github_repo_owner_name(repo_full)
        if own_repo:
            owner, repo = own_repo
            api_url = (
                f"https://api.github.com/repos/{owner}/{repo}/contents/"
                f"{quote(path_in_repo, safe='')}?ref={quote(branch, safe='')}"
            )
            try:
                raw = _fetch_github_contents_file(owner, repo, path_in_repo, branch)
                df = pd.read_csv(io.BytesIO(raw), low_memory=False)
                meta = {
                    "source": "github_contents_api",
                    "remote_url": url,
                    "github_api_url": api_url,
                    "error": None,
                    "bundled_csv_stale_risk": False,
                }
                with _CACHE_LOCK:
                    _CACHE_T = now
                    _CACHE_DF = df
                    _CACHE_META = dict(meta)
                return df.copy()
            except (HTTPError, URLError, OSError, TimeoutError, ValueError, json.JSONDecodeError) as e:
                meta = {
                    "source": "github_contents_api_failed",
                    "remote_url": url,
                    "github_api_url": api_url,
                    "error": repr(e)[:240],
                    "bundled_csv_stale_risk": False,
                }
                if mode in ("api", "contents", "github_api"):
                    # strict: do not fall back to stale raw
                    on_vercel = bool(
                        os.environ.get("VERCEL", "").strip()
                        or os.environ.get("VERCEL_ENV", "").strip()
                    )
                    if LOG_PATH.is_file():
                        try:
                            df = pd.read_csv(LOG_PATH, low_memory=False)
                            meta["source"] = "local_fallback_after_api_error"
                            meta["bundled_csv_stale_risk"] = bool(on_vercel)
                            with _CACHE_LOCK:
                                _CACHE_T = now
                                _CACHE_DF = df
                                _CACHE_META = dict(meta)
                            return df.copy()
                        except Exception as e2:
                            meta["error"] = (meta.get("error") or "") + f" {repr(e2)[:120]}"
                    with _CACHE_LOCK:
                        _CACHE_T = now
                        _CACHE_DF = None
                        _CACHE_META = dict(meta)
                    return None

    if url:
        try:
            raw = _fetch_url(url)
            df = pd.read_csv(io.BytesIO(raw), low_memory=False)
            meta = {
                "source": "remote_url",
                "remote_url": url,
                "github_api_url": meta.get("github_api_url"),
                "error": None,
                "bundled_csv_stale_risk": False,
            }
            with _CACHE_LOCK:
                _CACHE_T = now
                _CACHE_DF = df
                _CACHE_META = dict(meta)
            return df.copy()
        except (HTTPError, URLError, OSError, TimeoutError, ValueError) as e:
            err_prev = (meta.get("error") or "").strip()
            err_new = repr(e)[:240]
            meta = {
                "source": "remote_failed",
                "remote_url": url,
                "github_api_url": meta.get("github_api_url"),
                "error": f"{err_prev} | {err_new}" if err_prev else err_new,
                "bundled_csv_stale_risk": False,
            }

    on_vercel = bool(
        os.environ.get("VERCEL", "").strip()
        or os.environ.get("VERCEL_ENV", "").strip()
    )
    if LOG_PATH.is_file():
        try:
            df = pd.read_csv(LOG_PATH, low_memory=False)
            if meta.get("error"):
                meta["source"] = "local_fallback"
                meta["bundled_csv_stale_risk"] = bool(on_vercel)
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
