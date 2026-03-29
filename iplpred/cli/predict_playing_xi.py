"""
Predict probable playing XI from recent player_match_stats history.
"""

from __future__ import annotations

import argparse
import pandas as pd

from iplpred.core.squad_utils import (
    count_overseas,
    filter_match_stats_to_squad,
    is_overseas,
)
from iplpred.paths import PROCESSED_DIR

PLAYER_MATCH_STATS_PATH = PROCESSED_DIR / "player_match_stats.csv"

MIN_BATTERS = 4
MIN_BOWLERS = 3
MIN_ALLROUNDERS = 2
XI_SIZE = 11
BENCH_SIZE = 2

ROLE_ORDER = ("batter", "bowler", "allrounder")


def load_match_stats() -> pd.DataFrame:
    df = pd.read_csv(PLAYER_MATCH_STATS_PATH, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["player_id", "date", "match_id"])
    return df


def filter_team_rows(df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    tn = team_name.strip()
    bt = df["batting_team"].astype(str).str.strip()
    bw = df["bowling_team"].astype(str).str.strip()
    mask = (bt == tn) | (bw == tn)
    if not mask.any():
        mask = (bt.str.lower() == tn.lower()) | (bw.str.lower() == tn.lower())
    return df.loc[mask].copy()


def normalize_role(role: object) -> str:
    r = str(role).strip().lower()
    if r in ROLE_ORDER:
        return r
    return "allrounder"


def player_selection_table(team_df: pd.DataFrame) -> pd.DataFrame:
    """One row per player: matches_last_5/10, selection_score, latest role."""
    rows: list[dict] = []
    for pid in team_df["player_id"].unique():
        sub = team_df[team_df["player_id"] == pid].sort_values(["date", "match_id"])
        sub = sub.drop_duplicates(["match_id"], keep="last")
        if sub.empty:
            continue
        last5 = sub.tail(5)
        last10 = sub.tail(10)
        m5 = int(last5["match_id"].nunique())
        m10 = int(last10["match_id"].nunique())
        score = 0.6 * (m5 / 5.0) + 0.4 * (m10 / 10.0)
        role = normalize_role(sub.iloc[-1]["role"])
        rows.append(
            {
                "player_id": pid,
                "role": role,
                "matches_last_5": m5,
                "matches_last_10": m10,
                "selection_score": score,
            }
        )
    return pd.DataFrame(rows)


def team_match_rosters(team_df: pd.DataFrame, team_query: str) -> dict[str, list[str]]:
    """match_id -> player_ids who appeared for this team in that match."""
    q = team_query.strip().lower()
    out: dict[str, list[str]] = {}
    for mid, g in team_df.groupby("match_id", sort=False):
        g = g.sort_values(["date", "match_id"])
        bt = g["batting_team"].astype(str).str.strip().str.lower()
        bw = g["bowling_team"].astype(str).str.strip().str.lower()
        if not ((bt == q).any() or (bw == q).any()):
            continue
        out[str(mid)] = g["player_id"].drop_duplicates().tolist()
    return out


def resolve_team_display_name(team_df: pd.DataFrame, query: str) -> str:
    q = query.strip().lower()
    for col in ("batting_team", "bowling_team"):
        for v in team_df[col].astype(str).str.strip():
            if v.lower() == q:
                return v
    return str(team_df["batting_team"].iloc[0])


def build_playing_xi_and_bench(players: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """
    Pick XI: at least 4 batters, 3 bowlers, 2 allrounders; fill to 11 by score.
    Bench: next 2 by score among remaining.
    """
    if players.empty:
        return [], []

    players = players.sort_values("selection_score", ascending=False).reset_index(drop=True)
    by_role = {r: players[players["role"] == r].copy() for r in ROLE_ORDER}
    mins = [("batter", MIN_BATTERS), ("bowler", MIN_BOWLERS), ("allrounder", MIN_ALLROUNDERS)]
    picked: list[dict] = []
    picked_ids: set[str] = set()

    for role, need in mins:
        pool = by_role[role]
        pool = pool[~pool["player_id"].isin(picked_ids)].sort_values(
            "selection_score", ascending=False
        )
        take = min(need, len(pool))
        for i in range(take):
            row = pool.iloc[i]
            picked.append(row.to_dict())
            picked_ids.add(row["player_id"])

    need_more = XI_SIZE - len(picked)
    if need_more > 0:
        rest = players[~players["player_id"].isin(picked_ids)].sort_values(
            "selection_score", ascending=False
        )
        for i in range(min(need_more, len(rest))):
            row = rest.iloc[i]
            picked.append(row.to_dict())
            picked_ids.add(row["player_id"])

    picked.sort(key=lambda d: -d["selection_score"])
    xi = picked[:XI_SIZE]

    xi_ids = {d["player_id"] for d in xi}
    bench_pool = players[~players["player_id"].isin(xi_ids)].sort_values(
        "selection_score", ascending=False
    )
    bench = [bench_pool.iloc[i].to_dict() for i in range(min(BENCH_SIZE, len(bench_pool)))]

    return xi, bench


def enforce_overseas_cap(
    xi: list[dict],
    players: pd.DataFrame,
    team_name: str,
    max_overseas: int = 4,
) -> tuple[list[dict], str | None]:
    """Swap out lowest-scoring overseas if XI exceeds max overseas (IPL rule)."""
    xi = [dict(x) for x in xi]
    warn: str | None = None
    for _ in range(24):
        ids = [d["player_id"] for d in xi]
        if count_overseas(team_name, ids) <= max_overseas:
            return xi, warn
        overseas = [d for d in xi if is_overseas(team_name, d["player_id"])]
        if not overseas:
            warn = "Overseas cap exceeded and no overseas slot to swap"
            return xi, warn
        worst = min(overseas, key=lambda d: d["selection_score"])
        xi = [d for d in xi if d["player_id"] != worst["player_id"]]
        xi_ids = {d["player_id"] for d in xi}
        pool = players[
            (~players["player_id"].isin(xi_ids))
            & (~players["player_id"].map(lambda p: is_overseas(team_name, str(p))))
        ].sort_values("selection_score", ascending=False)
        if pool.empty:
            xi.append(worst)
            warn = "No domestic player available to satisfy overseas cap"
            return xi, warn
        xi.append(pool.iloc[0].to_dict())
    return xi, warn


def predict_playing_xi(team_name: str) -> dict:
    """
    Predict playing XI for a franchise team using recent appearances and role balance.

    Returns keys: team, playing_xi (list of dicts), bench, players (full table), rosters (optional).
    """
    df = load_match_stats()
    team_df = filter_team_rows(df, team_name)
    if team_df.empty:
        resolved = team_name.strip()
        return {
            "team": resolved,
            "playing_xi": [],
            "bench": [],
            "players": pd.DataFrame(),
            "error": f"No rows for team '{team_name}' in {PLAYER_MATCH_STATS_PATH}",
            "rosters": {},
            "squad_warning": None,
        }

    resolved = resolve_team_display_name(team_df, team_name)
    team_df = filter_match_stats_to_squad(team_df, resolved)
    if team_df.empty:
        return {
            "team": str(resolved),
            "playing_xi": [],
            "bench": [],
            "players": pd.DataFrame(),
            "error": (
                f"No rows left after IPL 2026 squad filter for '{resolved}'. "
                "Align player_id with data/processed/ipl_2026_squads.csv (or add aliases in squad_utils)."
            ),
            "rosters": {},
            "squad_warning": None,
        }
    players = player_selection_table(team_df)
    rosters = team_match_rosters(team_df, resolved)
    xi, _ = build_playing_xi_and_bench(players)
    xi, squad_warning = enforce_overseas_cap(xi, players, resolved)
    xi_ids = {d["player_id"] for d in xi}
    bench_pool = players[~players["player_id"].isin(xi_ids)].sort_values(
        "selection_score", ascending=False
    )
    bench = [bench_pool.iloc[i].to_dict() for i in range(min(BENCH_SIZE, len(bench_pool)))]

    return {
        "team": str(resolved),
        "playing_xi": xi,
        "bench": bench,
        "players": players,
        "rosters": rosters,
        "error": None,
        "squad_warning": squad_warning,
    }


def print_prediction(out: dict) -> None:
    if out.get("error"):
        print(out["error"])
        return
    if out.get("squad_warning"):
        print(f"Note: {out['squad_warning']}")
        print()
    team = out["team"]
    print(f"Team: {team}")
    print()
    print("Playing XI:")
    for i, p in enumerate(out["playing_xi"], start=1):
        print(
            f"{i}. {p['player_id']} ({p['role']}, score={p['selection_score']:.3f})"
        )
    print()
    print("Bench:")
    for p in out["bench"]:
        print(f"- {p['player_id']}")
    if not out["bench"]:
        print("- (not enough players)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict playing XI for a team.")
    parser.add_argument(
        "team",
        nargs="?",
        default="Mumbai Indians",
        help="Team name as in batting_team / bowling_team (e.g. Mumbai Indians)",
    )
    args = parser.parse_args()
    out = predict_playing_xi(args.team)
    print_prediction(out)
    print("Playing XI prediction ready")


if __name__ == "__main__":
    main()
