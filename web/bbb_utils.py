"""Ball-by-ball CSV: actual top batters / bowlers for compare UI."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def slug_id(s: str) -> str:
    s = (s or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def top5_batters_from_bbb(path: Path) -> list[str]:
    df = pd.read_csv(path, low_memory=False)
    if "batter_id" not in df.columns or "runs_batter" not in df.columns:
        return []
    df = df.copy()
    df["runs_batter"] = pd.to_numeric(df["runs_batter"], errors="coerce").fillna(0.0)
    bid = df["batter_id"].astype(str).str.strip()
    r = df.groupby(bid, sort=False)["runs_batter"].sum().sort_values(ascending=False)
    return [str(x) for x in r.head(5).index.tolist()]


def top3_bowlers_from_bbb(path: Path) -> list[str]:
    df = pd.read_csv(path, low_memory=False)
    if "bowler_id" not in df.columns or "wicket" not in df.columns:
        return []
    wk = df[df["wicket"] == True].copy()  # noqa: E712
    if len(wk) == 0:
        wk = df[df["wicket"].astype(str).str.lower().isin(["true", "1"])]
    if len(wk) == 0:
        return []
    bid = wk["bowler_id"].astype(str).str.strip()
    c = wk.groupby(bid, sort=False).size().sort_values(ascending=False)
    return [str(x) for x in c.head(3).index.tolist()]


def ordered_lists_match_slugs(pred: list[str], actual: list[str]) -> bool:
    """Match iff same length and each position matches after slug (pred vs actual lineups)."""
    if len(pred) != len(actual) or not pred:
        return False
    return all(slug_id(a) == slug_id(b) for a, b in zip(pred, actual))


def pair_slug_match(pred: str, actual: str) -> bool:
    if not (pred or "").strip() or not (actual or "").strip():
        return False
    return slug_id(pred) == slug_id(actual)
