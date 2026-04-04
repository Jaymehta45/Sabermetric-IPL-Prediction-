"""
Build player-match level stats from unified ball-by-ball data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from iplpred.paths import PROCESSED_DIR

from iplpred.pipeline.icc_match_stats_supplement import build_icc_supplement_df

INPUT_PATH = PROCESSED_DIR / "unified_ball_by_ball.csv"
OUTPUT_PATH = PROCESSED_DIR / "player_match_stats.csv"

WICKET_EXCLUDED_KINDS = frozenset(
    {"run out", "retired hurt", "obstructing the field"}
)


def load_unified() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df["dismissal_kind"] = (
        df["dismissal_kind"].fillna("").astype(str).replace({"nan": "", "None": ""})
    )
    df["player_dismissed"] = (
        df["player_dismissed"].fillna("").astype(str).replace({"nan": "", "None": ""})
    )
    df["batter"] = df["batter"].fillna("").astype(str).str.strip()
    df["bowler"] = df["bowler"].fillna("").astype(str).str.strip()
    df["player_dismissed"] = df["player_dismissed"].str.strip()
    df["dismissal_kind"] = df["dismissal_kind"].str.strip()
    df["runs_off_bat"] = pd.to_numeric(df["runs_off_bat"], errors="coerce").fillna(0.0)
    df["extras"] = pd.to_numeric(df["extras"], errors="coerce").fillna(0.0)
    df["is_wide"] = pd.to_numeric(df["is_wide"], errors="coerce").fillna(0).astype(int)
    df["is_noball"] = pd.to_numeric(df["is_noball"], errors="coerce").fillna(0).astype(int)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["total_runs_conceded"] = df["runs_off_bat"] + df["extras"]
    return df


def build_batting(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["valid_ball_bat"] = (df["is_wide"] == 0).astype(int)
    df["is_four"] = (df["runs_off_bat"] == 4).astype(int)
    df["is_six"] = (df["runs_off_bat"] == 6).astype(int)
    df["dismissed_row"] = (df["player_dismissed"] == df["batter"]).astype(int)

    bat = (
        df.groupby(["batter", "match_id"], as_index=False)
        .agg(
            runs=("runs_off_bat", "sum"),
            balls=("valid_ball_bat", "sum"),
            fours=("is_four", "sum"),
            sixes=("is_six", "sum"),
            dismissed=("dismissed_row", "max"),
            batting_team=("batting_team", "first"),
            venue=("venue", "first"),
            date=("date", "first"),
        )
        .rename(columns={"batter": "player_id"})
    )
    return bat


def build_bowling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["valid_ball_bowl"] = ((df["is_wide"] == 0) & (df["is_noball"] == 0)).astype(int)
    dk = df["dismissal_kind"].str.strip().str.lower()
    df["is_wicket_credit"] = (dk != "") & (~dk.isin(WICKET_EXCLUDED_KINDS))

    bowl = (
        df.groupby(["bowler", "match_id"], as_index=False)
        .agg(
            wickets=("is_wicket_credit", "sum"),
            balls_bowled=("valid_ball_bowl", "sum"),
            runs_conceded=("total_runs_conceded", "sum"),
            bowling_team=("bowling_team", "first"),
            venue=("venue", "first"),
            date=("date", "first"),
        )
        .rename(columns={"bowler": "player_id"})
    )
    return bowl


def build_match_team_pairs(unified: pd.DataFrame) -> dict:
    """
    Per match_id: unique teams from batting_team column in unified_ball_by_ball.
    Returns match_id -> (team1, team2) sorted, or None if not exactly two teams.
    """
    pairs: dict = {}
    for mid, g in unified.groupby("match_id", sort=False):
        bt = g["batting_team"].dropna().astype(str).str.strip()
        bt = bt[bt != ""]
        bt = bt[bt.str.lower() != "nan"]
        teams = sorted(set(bt))
        if len(teams) == 2:
            pairs[mid] = (teams[0], teams[1])
        else:
            pairs[mid] = None
    return pairs


def assign_teams_two_sides(merged: pd.DataFrame, match_team_pairs: dict) -> pd.DataFrame:
    """
    If match has exactly two teams (t1, t2) from unified batting_team:
      player_team = non-empty batting_team from row, else bowling_team from row.
      if player_team == t1 -> batting_team=t1, bowling_team=t2
      elif player_team == t2 -> batting_team=t2, bowling_team=t1
      else -> unknown / unknown
    If not two teams: batting_team=bowling_team='unknown'
    """
    m = merged.copy()
    bt_in = m["batting_team"].fillna("").astype(str).str.strip()
    bwl_in = m["bowling_team"].fillna("").astype(str).str.strip()
    bt_in = bt_in.replace({"nan": ""})
    bwl_in = bwl_in.replace({"nan": ""})

    player_team = bt_in.where(bt_in != "", bwl_in)

    out_bt: list[str] = []
    out_bwl: list[str] = []

    for i in range(len(m)):
        mid = m["match_id"].iloc[i]
        pt = player_team.iloc[i]
        pair = match_team_pairs.get(mid)

        if pair is None:
            out_bt.append("unknown")
            out_bwl.append("unknown")
            continue

        t1, t2 = pair
        if pt == "" or pt.lower() == "nan":
            out_bt.append("unknown")
            out_bwl.append("unknown")
        elif pt == t1:
            out_bt.append(t1)
            out_bwl.append(t2)
        elif pt == t2:
            out_bt.append(t2)
            out_bwl.append(t1)
        else:
            out_bt.append("unknown")
            out_bwl.append("unknown")

    m["batting_team"] = out_bt
    m["bowling_team"] = out_bwl
    return m


def assign_role(balls: pd.Series, balls_bowled: pd.Series) -> pd.Series:
    b = balls.astype(float).fillna(0).to_numpy()
    bb = balls_bowled.astype(float).fillna(0).to_numpy()
    out = np.where(b > 2 * bb, "batter", np.where(bb > 2 * b, "bowler", "allrounder"))
    return pd.Series(out, index=balls.index)


def main() -> None:
    df = load_unified()

    bat = build_batting(df)
    bowl = build_bowling(df)

    bat = bat.rename(columns={"venue": "venue_bat", "date": "date_bat"})
    bowl = bowl.rename(columns={"venue": "venue_bowl", "date": "date_bowl"})

    merged = bat.merge(bowl, on=["player_id", "match_id"], how="outer")

    merged["date"] = merged["date_bat"].combine_first(merged["date_bowl"])
    merged["venue"] = merged["venue_bat"].combine_first(merged["venue_bowl"])
    merged = merged.drop(columns=["date_bat", "date_bowl", "venue_bat", "venue_bowl"])

    merged = merged[merged["player_id"].astype(str).str.strip() != ""]

    match_team_pairs = build_match_team_pairs(df)
    n_two_team_matches = sum(1 for v in match_team_pairs.values() if v is not None)
    n_unknown_fallback_matches = sum(1 for v in match_team_pairs.values() if v is None)
    print(f"matches with valid 2-team mapping: {n_two_team_matches}")
    print(f"matches falling back to unknown: {n_unknown_fallback_matches}")

    merged = assign_teams_two_sides(merged, match_team_pairs)

    merged["match_key"] = (
        merged["match_id"].astype(str)
        + "_"
        + pd.to_datetime(merged["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    )

    numeric_cols = [
        "runs",
        "balls",
        "fours",
        "sixes",
        "dismissed",
        "wickets",
        "balls_bowled",
        "runs_conceded",
    ]
    for c in numeric_cols:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0)

    merged["total_balls_faced"] = merged["balls"]
    merged["total_balls_bowled"] = merged["balls_bowled"]
    merged["role"] = assign_role(merged["balls"], merged["balls_bowled"])

    bt = merged["batting_team"].astype(str).str.strip()
    bwl = merged["bowling_team"].astype(str).str.strip()
    missing_team_mask = (
        merged["batting_team"].isna()
        | merged["bowling_team"].isna()
        | (bt == "")
        | (bwl == "")
    )
    pct_missing_teams = 100.0 * float(missing_team_mask.mean())

    out_cols = [
        "match_key",
        "match_id",
        "player_id",
        "date",
        "venue",
        "batting_team",
        "bowling_team",
        "runs",
        "balls",
        "fours",
        "sixes",
        "dismissed",
        "wickets",
        "balls_bowled",
        "runs_conceded",
        "total_balls_faced",
        "total_balls_bowled",
        "role",
    ]
    merged = merged[out_cols]

    numeric_out = [
        "runs",
        "balls",
        "fours",
        "sixes",
        "dismissed",
        "wickets",
        "balls_bowled",
        "runs_conceded",
        "total_balls_faced",
        "total_balls_bowled",
    ]

    icc_sup = build_icc_supplement_df()
    if icc_sup is not None and not icc_sup.empty:
        icc_sup = icc_sup[[c for c in out_cols if c in icc_sup.columns]]
        for c in out_cols:
            if c not in icc_sup.columns:
                icc_sup[c] = 0.0 if c in numeric_out else ""
        icc_sup = icc_sup[out_cols]
        merged = pd.concat([merged, icc_sup], ignore_index=True)
    for c in numeric_out:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"total rows: {len(merged)}")
    print(f"unique players: {merged['player_id'].nunique()}")
    print(f"unique matches: {merged['match_id'].nunique()}")
    print(f"% rows with missing teams: {pct_missing_teams:.4f}")
    print("role distribution:")
    print(merged["role"].value_counts().to_string())
    print("player_match_stats created")


if __name__ == "__main__":
    main()
