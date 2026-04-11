#!/usr/bin/env python3
"""
Build overview charts for Royal Challengers Bangalore / Bengaluru (RCB) from processed CSVs.

Reads:
  data/processed/match_training_dataset.csv
  data/processed/player_match_stats.csv

Writes (default):
  reports/rcb_team_overview.png

Usage (repo root):
  python3 scripts/visualize_rcb_team.py
  python3 scripts/visualize_rcb_team.py --out reports/my_rcb.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MATCH_TRAINING = REPO_ROOT / "data" / "processed" / "match_training_dataset.csv"
PLAYER_MATCH_STATS = REPO_ROOT / "data" / "processed" / "player_match_stats.csv"

RCB_NAMES = frozenset(
    {
        "Royal Challengers Bangalore",
        "Royal Challengers Bengaluru",
        "RCB",
    }
)


def is_rcb(name: object) -> bool:
    return str(name).strip() in RCB_NAMES


def load_rcb_matches() -> pd.DataFrame:
    df = pd.read_csv(MATCH_TRAINING, low_memory=False)
    df = df[df["winner"].isin(["team1", "team2"])].copy()
    m = df["team1_name"].map(is_rcb) | df["team2_name"].map(is_rcb)
    df = df.loc[m].copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("match_date")

    def rcb_runs(row: pd.Series) -> float:
        if is_rcb(row["team1_name"]):
            return float(row["team1_total_runs"])
        return float(row["team2_total_runs"])

    def opp_runs(row: pd.Series) -> float:
        if is_rcb(row["team1_name"]):
            return float(row["team2_total_runs"])
        return float(row["team1_total_runs"])

    def rcb_won(row: pd.Series) -> int:
        if row["winner"] == "team1" and is_rcb(row["team1_name"]):
            return 1
        if row["winner"] == "team2" and is_rcb(row["team2_name"]):
            return 1
        return 0

    df["rcb_runs"] = df.apply(rcb_runs, axis=1)
    df["opp_runs"] = df.apply(opp_runs, axis=1)
    df["rcb_win"] = df.apply(rcb_won, axis=1)
    return df


def load_rcb_batting_totals() -> pd.Series:
    pms = pd.read_csv(PLAYER_MATCH_STATS, low_memory=False)
    bat = pms[pms["batting_team"].map(is_rcb)].copy()
    bat["runs"] = pd.to_numeric(bat.get("runs"), errors="coerce").fillna(0.0)
    return bat.groupby("player_id", sort=False)["runs"].sum().sort_values(ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="RCB team visualization from processed data.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(REPO_ROOT / "reports" / "rcb_team_overview.png"),
        help="Output PNG path",
    )
    args = parser.parse_args()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()

    if not MATCH_TRAINING.is_file():
        raise SystemExit(f"Missing {MATCH_TRAINING}")
    if not PLAYER_MATCH_STATS.is_file():
        raise SystemExit(f"Missing {PLAYER_MATCH_STATS}")

    mt = load_rcb_matches()
    top_batters = load_rcb_batting_totals().head(12)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(
        "RCB — Royal Challengers Bangalore / Bengaluru (processed training data)",
        fontsize=14,
        fontweight="bold",
    )

    # 1) Team score timeline
    ax = axes[0, 0]
    wins = mt["rcb_win"] == 1
    ax.scatter(
        mt.loc[wins, "match_date"],
        mt.loc[wins, "rcb_runs"],
        c="#2ecc71",
        s=28,
        alpha=0.75,
        label="Win",
    )
    ax.scatter(
        mt.loc[~wins, "match_date"],
        mt.loc[~wins, "rcb_runs"],
        c="#e74c3c",
        s=28,
        alpha=0.65,
        label="Loss",
    )
    ax.set_title("RCB innings total (match_training rows)")
    ax.set_ylabel("Runs")
    ax.legend(loc="upper left", frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right")

    # 2) Rolling win rate (last 15 matches, min_periods=5)
    ax = axes[0, 1]
    roll = mt.set_index("match_date")["rcb_win"].sort_index().rolling(15, min_periods=5).mean()
    ax.plot(roll.index, roll.values * 100.0, color="#9b59b6", linewidth=2)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Rolling win rate (15-match window)")
    ax.set_ylabel("Win %")
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right")

    # 3) Margin: RCB runs − opponent runs
    ax = axes[1, 0]
    margin = (mt["rcb_runs"] - mt["opp_runs"]).values
    ax.hist(margin, bins=20, color="#3498db", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_title("Run margin vs opponent (RCB − opp)")
    ax.set_xlabel("Runs")
    ax.set_ylabel("Matches")

    # 4) Top run-scorers (player_match_stats, batting as RCB)
    ax = axes[1, 1]
    if len(top_batters):
        y_pos = np.arange(len(top_batters))
        ax.barh(y_pos, top_batters.values, color="#e67e22", alpha=0.85)
        ax.set_yticks(y_pos)
        labels = [str(p)[:22] + "…" if len(str(p)) > 23 else str(p) for p in top_batters.index]
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title("Top batters — career runs while batting as RCB (pms)")
        ax.set_xlabel("Total runs (sum of rows)")
    else:
        ax.text(0.5, 0.5, "No RCB batting rows", ha="center", va="center")

    n_matches = len(mt)
    n_wins = int(mt["rcb_win"].sum()) if n_matches else 0
    subtitle = f"Matches in table: {n_matches}  |  Wins: {n_wins} ({100 * n_wins / max(n_matches, 1):.1f}%)"
    fig.text(0.5, 0.01, subtitle, ha="center", fontsize=10, color="#333")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
