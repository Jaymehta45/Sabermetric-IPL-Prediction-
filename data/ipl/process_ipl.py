import pandas as pd

# -----------------------
# LOAD DATA
# -----------------------
deliveries = pd.read_csv("data/ipl/deliveries_updated_ipl_upto_2025.csv")
matches = pd.read_csv("data/ipl/matches_updated_ipl_upto_2025.csv")

# -----------------------
# FILTER (2020 onwards; source CSVs are pre-trimmed)
# -----------------------
matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
matches = matches[matches["date"] >= "2020-01-01"]

valid_match_ids = set(matches["matchId"])
deliveries = deliveries[deliveries["matchId"].isin(valid_match_ids)]

# -----------------------
# STANDARDIZE COLUMNS
# -----------------------
df = deliveries.rename(columns={
    "matchId": "match_id",
    "batsman": "batter",
    "bowler": "bowler",
    "batsman_runs": "runs_off_bat",
    "extras": "extras",
    "isWide": "is_wide",
    "isNoBall": "is_noball",
    "dismissal_kind": "wicket_type",
    "player_dismissed": "player_out"
})

# -----------------------
# DERIVED FLAGS
# -----------------------
df["is_ball_faced"] = (df["is_wide"] == 0).astype(int)
df["is_ball_bowled"] = ((df["is_wide"] == 0) & (df["is_noball"] == 0)).astype(int)

df["is_wicket"] = (
    df["player_out"].notna() &
    (df["wicket_type"] != "run out")
).astype(int)

df["total_runs_conceded"] = df["runs_off_bat"] + df["extras"]

# -----------------------
# BATTING AGG
# -----------------------
batting = df.groupby("batter").agg(
    runs=("runs_off_bat", "sum"),
    balls=("is_ball_faced", "sum"),
    fours=("runs_off_bat", lambda x: (x == 4).sum()),
    sixes=("runs_off_bat", lambda x: (x == 6).sum()),
    dismissals=("is_wicket", "sum")
).reset_index()

batting["strike_rate"] = (batting["runs"] / batting["balls"]) * 100

# -----------------------
# BOWLING AGG
# -----------------------
bowling = df.groupby("bowler").agg(
    wickets=("is_wicket", "sum"),
    balls_bowled=("is_ball_bowled", "sum"),
    runs_conceded=("total_runs_conceded", "sum")
).reset_index()

bowling["overs"] = bowling["balls_bowled"] / 6
bowling["economy"] = bowling["runs_conceded"] / bowling["overs"]

# -----------------------
# SAVE OUTPUT
# -----------------------
batting.to_csv("ipl_batting.csv", index=False)
bowling.to_csv("ipl_bowling.csv", index=False)

print("IPL processing done.")