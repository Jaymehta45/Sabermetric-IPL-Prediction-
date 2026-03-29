"""
Venue-aware spin vs pace nudges on predicted wickets (small / batting-friendly
grounds hurt spinners; pace may get a mild bump). Uses optional CSV overrides.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import pandas as pd

from iplpred.paths import PROCESSED_DIR

STYLE_CSV = PROCESSED_DIR / "player_bowling_style.csv"

# 0 = neutral, 1 = strong (e.g. Chinnaswamy-style small ground)
_VENUE_SUBSTRINGS: tuple[tuple[str, float], ...] = (
    ("chinnaswamy", 0.92),
    ("bengaluru", 0.88),
    ("bangalore", 0.88),
    ("wankhede", 0.78),
    ("brabourne", 0.72),
    ("dy patil", 0.70),
    ("mumbai", 0.65),
    ("eden gardens", 0.55),
    ("eden", 0.50),
    ("kolkata", 0.50),
    ("barsapara", 0.48),
    ("guwahati", 0.45),
    ("punjab cricket", 0.58),
    ("mohali", 0.55),
    ("dharamsala", 0.42),
    ("hpca", 0.42),
    ("narendra modi", 0.40),
    ("ahmedabad", 0.40),
    ("rajiv gandhi", 0.35),
    ("hyderabad", 0.38),
    ("chepauk", 0.12),
    ("chennai", 0.15),
    ("ma chidambaram", 0.12),
    ("sawai mansingh", 0.48),
    ("jaipur", 0.48),
)

# Tighter spin wicket penalty at harsh venues; small pace bump (balance)
SPIN_WK_LOSS_PER_HARSH = 0.24
PACE_WK_GAIN_PER_HARSH = 0.07


def venue_spin_harshness(venue: str | None) -> float:
    """Roughly how much 'spinners get bashed' applies at this ground (0–1)."""
    if not venue or not str(venue).strip():
        return 0.0
    v = str(venue).lower()
    best = 0.0
    for sub, h in _VENUE_SUBSTRINGS:
        if sub in v:
            best = max(best, h)
    return max(0.0, min(1.0, best))


def _infer_style_from_id(player_id: str) -> str:
    """Heuristic spin vs pace from stats-style player_id (override via CSV when wrong)."""
    pid = (player_id or "").strip()
    if not pid:
        return "unknown"
    pl = pid.lower()

    # Spin (check before generic pace patterns)
    _spin_markers = (
        "rashid khan",
        "kuldeep yadav",
        "adam zampa",
        "ravi bishnoi",
        "varun chakaravarthy",
        "krunal pandya",
        "axar patel",
        "washington sundar",
        "ravindra jadeja",
        "suyash sharma",
        "shreyas gopal",
        "rahul chahar",
        "ys chahal",
        "mitchell santner",
        "akeal hosein",
        "maheesh theekshana",
        "sunil narine",
        "mujeeb ur rahman",
        "noor ahmad",
        "adil rashid",
        "tabraiz shamsi",
        "ashton agar",
        "fabian allen",
        "rahul tewatia",
        "shahbaz ahmed",
        "abhishek sharma",
        "anukul roy",
        "vipraj nigam",
        "mayank markande",
        "karn sharma",
        "piyush chawla",
        "amit mishra",
        "murugan ashwin",
        "r ashwin",
    )
    for m in _spin_markers:
        if m in pl:
            return "spin"
    if pl.endswith(" ashwin"):
        return "spin"
    if pl.endswith(" chahar") and "rahul" in pl:
        return "spin"
    if "chahal" in pl or "zampa" in pl or "bishnoi" in pl or "chakaravarthy" in pl:
        return "spin"

    # Pace / seam
    _pace_markers = (
        "bumrah",
        "mohammed shami",
        "shami",
        "siraj",
        "nortje",
        "rabada",
        "ferguson",
        "milne",
        "boult",
        "southee",
        "carse",
        "jansen",
        "coetzee",
        "wood",
        "topley",
        "willey",
        "jordan",
        "archer",
        "arshdeep",
        "avesh khan",
        "khaleel ahmed",
        "prasidh krishna",
        "harshal patel",
        "deepak chahar",
        "mukesh kumar",
        "yash dayal",
        "dwarshuis",
        "behrendorff",
        "ellis",
        "hazlewood",
        "cummins",
        "starc",
        "jaydev unadkat",
        "bhuvneshwar kumar",
        "t natarajan",
        "natarajan",
        "umran malik",
        "lockie ferguson",
        "alzarri joseph",
        "lungi ngidi",
        "marco jansen",
        "andre russell",  # hits as pace AR
        "sam curran",
        "tom curran",
        "hardik pandya",
        "vyshak",
        "vijaykumar",
        "yash thakur",
        "kulwant khejroliya",
        "akash singh",
        "kamlesh nagarkoti",
        "shardul thakur",
        "nathan ellis",
        "gus atkinson",
        "matheesha pathirana",
        "pathirana",
        "dushmantha chameera",
        "maheesh pathirana",
        "jacob duffy",
        "romario shepherd",
        "david payne",
        "harshal",
        "prasidh",
        "abhinandan singh",
        "eshan malinga",
    )
    for m in _pace_markers:
        if m in pl:
            return "pace"

    if "curran" in pl and ("sam" in pl or "tom" in pl):
        return "pace"

    return "unknown"


@lru_cache(maxsize=1)
def load_bowling_style_overrides() -> dict[str, str]:
    """Optional CSV: player_id, bowling_style (spin|pace|unknown)."""
    out: dict[str, str] = {}
    if not STYLE_CSV.is_file():
        return out
    with STYLE_CSV.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = (row.get("player_id") or "").strip()
            st = (row.get("bowling_style") or row.get("style") or "").strip().lower()
            if not pid or st not in ("spin", "pace", "unknown"):
                continue
            out[pid] = st
    return out


def bowling_style_for_player(player_id: str) -> str:
    pid = (player_id or "").strip()
    ovr = load_bowling_style_overrides()
    if pid in ovr:
        return ovr[pid]
    return _infer_style_from_id(pid)


def apply_venue_bowling_style_to_team_df(
    active: pd.DataFrame,
    venue: str | None,
) -> tuple[pd.DataFrame, float]:
    """
    Scale predicted_wickets / pred_wk_raw per player using spin/pace + venue harshness.
    Returns (adjusted_df, harshness_used).
    """
    h = venue_spin_harshness(venue)
    if h <= 0.0 or active is None or len(active) == 0:
        return active, 0.0

    out = active.copy()
    keys = out["player_id"].astype(str).str.strip()
    styles = [bowling_style_for_player(k) for k in keys]

    wk_mult = []
    for st in styles:
        if st == "spin":
            m = 1.0 - h * SPIN_WK_LOSS_PER_HARSH
        elif st == "pace":
            m = 1.0 + h * PACE_WK_GAIN_PER_HARSH
        else:
            m = 1.0
        wk_mult.append(max(0.35, min(1.35, m)))

    wm = pd.Series(wk_mult, index=out.index, dtype=float)
    for col in ("predicted_wickets", "pred_wk_raw"):
        if col in out.columns:
            out[col] = out[col].astype(float) * wm

    if "impact_base" in out.columns and "predicted_runs" in out.columns and "predicted_wickets" in out.columns:
        out["impact_base"] = out["predicted_runs"].astype(float) + 25.0 * out["predicted_wickets"].astype(
            float
        )

    out["venue_bowling_wk_mult"] = wm
    out["bowling_style_used"] = styles
    return out, h


def seed_bowling_style_csv_from_squads(path: Path = PROCESSED_DIR / "ipl_2026_squads.csv") -> None:
    """Dev helper: write player_bowling_style.csv for all squad players (heuristic + unknown)."""
    df = pd.read_csv(path, low_memory=False)
    rows = []
    for pid in sorted(df["player_id"].astype(str).str.strip().unique()):
        rows.append(
            {
                "player_id": pid,
                "bowling_style": _infer_style_from_id(pid),
            }
        )
    STYLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with STYLE_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["player_id", "bowling_style"])
        w.writeheader()
        w.writerows(rows)
