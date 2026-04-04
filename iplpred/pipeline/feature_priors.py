"""
Append synthetic **prior** rows for squad/identity players missing from player_features.

Uses **column-wise medians** for numeric features; ``match_id = -999``, ``team = unknown``.
Appends matching ``player_match_stats`` rows so ``train_player_model`` meta-merge works.
"""

from __future__ import annotations

import pandas as pd

from iplpred.paths import PROCESSED_DIR

FEATURES_PATH = PROCESSED_DIR / "player_features.csv"
IDENTITY_PATH = PROCESSED_DIR / "player_identity.csv"
MATCH_STATS_PATH = PROCESSED_DIR / "player_match_stats.csv"

PRIOR_MATCH_ID = -999
PRIOR_DATE = "2026-01-01"


def append_squad_priors() -> tuple[int, int]:
    """
    Returns (n_rows_appended_features, n_rows_appended_match_stats).
    """
    if not IDENTITY_PATH.is_file() or not FEATURES_PATH.is_file():
        return 0, 0

    idf = pd.read_csv(IDENTITY_PATH, low_memory=False)
    stats_ids = idf["stats_player_id"].astype(str).str.strip().unique().tolist()

    feat = pd.read_csv(FEATURES_PATH, low_memory=False)
    have = set(feat["player_id"].astype(str).str.strip().unique())
    missing = [s for s in stats_ids if s and s not in have]
    if not missing:
        return 0, 0

    med = feat.median(numeric_only=True)
    new_rows: list[dict] = []
    for pid in missing:
        row: dict = {}
        for c in feat.columns:
            if c == "player_id":
                row[c] = pid
            elif c == "match_id":
                row[c] = PRIOR_MATCH_ID
            elif c == "team":
                row[c] = "unknown"
            elif c == "opponent_team":
                row[c] = "unknown"
            elif c in med.index and pd.notna(med[c]):
                row[c] = float(med[c])
            else:
                v = feat[c].dropna()
                row[c] = v.iloc[0] if len(v) else ""
        new_rows.append(row)

    add_f = pd.DataFrame(new_rows)
    feat2 = pd.concat([feat, add_f], ignore_index=True)
    feat2.to_csv(FEATURES_PATH, index=False)

    ms_n = 0
    if MATCH_STATS_PATH.is_file():
        ms = pd.read_csv(MATCH_STATS_PATH, low_memory=False)
        mexist = set(
            zip(
                ms["player_id"].astype(str).str.strip(),
                pd.to_numeric(ms["match_id"], errors="coerce").fillna(0).astype(int),
            )
        )
        ms_rows = []
        r_med = float(med["runs"]) if "runs" in med.index and pd.notna(med["runs"]) else 15.0
        w_med = float(med["wickets"]) if "wickets" in med.index and pd.notna(med["wickets"]) else 0.5
        for pid in missing:
            key = (pid, PRIOR_MATCH_ID)
            if key in mexist:
                continue
            ms_rows.append(
                {
                    "match_key": f"{PRIOR_MATCH_ID}_{PRIOR_DATE}",
                    "match_id": PRIOR_MATCH_ID,
                    "player_id": pid,
                    "date": PRIOR_DATE,
                    "venue": "prior_synthetic",
                    "batting_team": "unknown",
                    "bowling_team": "unknown",
                    "runs": r_med,
                    "balls": 10.0,
                    "fours": 0.0,
                    "sixes": 0.0,
                    "dismissed": 0.0,
                    "wickets": w_med,
                    "balls_bowled": 12.0,
                    "runs_conceded": 30.0,
                    "total_balls_faced": 10.0,
                    "total_balls_bowled": 12.0,
                    "role": "allrounder",
                }
            )
        if ms_rows:
            ms2 = pd.concat([ms, pd.DataFrame(ms_rows)], ignore_index=True)
            ms2.to_csv(MATCH_STATS_PATH, index=False)
            ms_n = len(ms_rows)

    return len(missing), ms_n
