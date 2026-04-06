"""
Build match-level training dataset from per-player features (no ML).

Winner labels: prefer official IPL results from data/ipl/matches_updated_mens_ipl.csv
(matchId, winner, team1, team2). Falls back to runs-proxy when no official row matches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from iplpred.core.team_momentum import attach_momentum_columns_chronological
from iplpred.paths import DATA_DIR, PROCESSED_DIR

PLAYER_FEATURES_PATH = PROCESSED_DIR / "player_features.csv"
PLAYER_MATCH_STATS_PATH = PROCESSED_DIR / "player_match_stats.csv"
OUTPUT_PATH = PROCESSED_DIR / "match_training_dataset.csv"
IPL_MATCHES_PATH = DATA_DIR / "ipl" / "matches_updated_mens_ipl.csv"


def load_official_ipl_matches() -> pd.DataFrame:
    """match_id, winner, team1, team2 from IPL matches file (empty if missing)."""
    if not IPL_MATCHES_PATH.exists():
        return pd.DataFrame(columns=["match_id", "winner", "team1", "team2"])
    m = pd.read_csv(IPL_MATCHES_PATH, low_memory=False)
    if "matchId" not in m.columns:
        return pd.DataFrame(columns=["match_id", "winner", "team1", "team2"])
    m["match_id"] = pd.to_numeric(m["matchId"], errors="coerce")
    out = m[["match_id", "winner", "team1", "team2"]].copy()
    out["match_id"] = out["match_id"].astype("int64")
    out["winner"] = out["winner"].fillna("").astype(str).str.strip()
    out["team1"] = out["team1"].fillna("").astype(str).str.strip()
    out["team2"] = out["team2"].fillna("").astype(str).str.strip()
    return out.dropna(subset=["match_id"])


def teams_match_franchise(a: str, b: str) -> bool:
    """Loose equality for franchise renames (e.g. Kings XI Punjab vs Punjab Kings)."""
    a, b = a.strip().lower(), b.strip().lower()
    if a == b:
        return True
    aliases = {
        frozenset({"kings xi punjab", "punjab kings"}),
    }
    if frozenset({a, b}) in aliases:
        return True
    return False


def resolve_winner_label(
    match_id: object,
    team1_name: str,
    team2_name: str,
    official: pd.DataFrame,
    proxy_winners: pd.DataFrame,
) -> tuple[object, str]:
    """
    Returns (winner_label, source) where winner_label is team1|team2|tie|nan.
    source is official|proxy|missing.
    """
    mid = match_id
    if official is not None and len(official) > 0:
        o = official[official["match_id"] == mid]
        if o.empty and mid is not None:
            try:
                o = official[official["match_id"] == int(float(mid))]
            except (TypeError, ValueError):
                pass
        if not o.empty:
            w = str(o.iloc[0]["winner"]).strip()
            if not w or w.lower() in ("nan", "none", ""):
                return "tie", "official"
            if teams_match_franchise(w, team1_name):
                return "team1", "official"
            if teams_match_franchise(w, team2_name):
                return "team2", "official"
            # Official row exists but winner doesn't match sorted teams — fall back
    pw = proxy_winners[proxy_winners["match_id"] == mid]
    if pw.empty and mid is not None:
        try:
            pw = proxy_winners[proxy_winners["match_id"] == int(float(mid))]
        except (TypeError, ValueError):
            pass
    if not pw.empty:
        return pw.iloc[0]["winner"], "proxy"
    return np.nan, "missing"


def build_player_features_from_match_stats() -> pd.DataFrame:
    """Derive player_features from player_match_stats (pre-match signals only)."""
    pms = pd.read_csv(PLAYER_MATCH_STATS_PATH, low_memory=False)
    pms["date"] = pd.to_datetime(pms["date"], errors="coerce")
    pms = pms.sort_values(["player_id", "date", "match_id"])

    pms["team"] = pms["batting_team"].fillna("").astype(str).str.strip()
    mask = pms["team"].eq("") | pms["team"].str.lower().eq("unknown")
    pms.loc[mask, "team"] = (
        pms.loc[mask, "bowling_team"].fillna("").astype(str).str.strip()
    )

    balls = pd.to_numeric(pms["balls"], errors="coerce").fillna(0.0)
    bb = pd.to_numeric(pms["balls_bowled"], errors="coerce").fillna(0.0)
    runs = pd.to_numeric(pms["runs"], errors="coerce").fillna(0.0)
    rc = pd.to_numeric(pms["runs_conceded"], errors="coerce").fillna(0.0)

    pms["match_sr"] = np.where(balls > 0, runs / balls * 100.0, np.nan)
    pms["match_econ"] = np.where(bb > 0, rc / (bb / 6.0), np.nan)

    pms["form_runs"] = (
        pms.groupby("player_id")["runs"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0.0)
    )
    pms["form_wickets"] = (
        pms.groupby("player_id")["wickets"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0.0)
    )
    pms["strike_rate"] = (
        pms.groupby("player_id")["match_sr"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0.0)
    )
    pms["economy"] = (
        pms.groupby("player_id")["match_econ"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0.0)
    )

    out = pms[
        [
            "match_id",
            "player_id",
            "team",
            "runs",
            "wickets",
            "form_runs",
            "form_wickets",
            "strike_rate",
            "economy",
        ]
    ].copy()
    out["form_runs_ipl"] = out["form_runs"]
    out["form_wickets_ipl"] = out["form_wickets"]
    out["days_since_prev_match"] = 0.0
    out.to_csv(PLAYER_FEATURES_PATH, index=False)
    return out


def load_player_features_for_training() -> pd.DataFrame:
    """Prefer existing player_features.csv (full build_features pipeline); else derive from match stats."""
    if PLAYER_FEATURES_PATH.exists():
        return pd.read_csv(PLAYER_FEATURES_PATH, low_memory=False)
    if PLAYER_MATCH_STATS_PATH.exists():
        return build_player_features_from_match_stats()
    raise FileNotFoundError(
        f"Need {PLAYER_MATCH_STATS_PATH} or {PLAYER_FEATURES_PATH}"
    )


def team_aggregates(pf: pd.DataFrame) -> pd.DataFrame:
    """Pre-match team features only (no current-match runs/wickets)."""
    agg_kw: dict = dict(
        team_avg_form_runs=("form_runs", "mean"),
        team_avg_form_wickets=("form_wickets", "mean"),
        team_avg_strike_rate=("strike_rate", "mean"),
        team_avg_economy=("economy", "mean"),
        team_n_players=("player_id", "count"),
    )
    if "form_runs_ipl" in pf.columns:
        agg_kw["team_avg_form_runs_ipl"] = ("form_runs_ipl", "mean")
    if "form_wickets_ipl" in pf.columns:
        agg_kw["team_avg_form_wickets_ipl"] = ("form_wickets_ipl", "mean")
    g = pf.groupby(["match_id", "team"], as_index=False).agg(**agg_kw)
    fr = g["team_avg_form_runs"]
    fw = g["team_avg_form_wickets"]
    if "team_avg_form_runs_ipl" in g.columns:
        fr = g["team_avg_form_runs_ipl"].fillna(g["team_avg_form_runs"])
    if "team_avg_form_wickets_ipl" in g.columns:
        fw = g["team_avg_form_wickets_ipl"].fillna(g["team_avg_form_wickets"])
    g["team_strength_score"] = 0.4 * fr + 0.3 * g["team_avg_strike_rate"] + 0.3 * fw
    return g


def match_winner_proxy(pf: pd.DataFrame) -> pd.DataFrame:
    """
    Temporary target: winner from total match runs by team.
    """
    # WARNING: winner is approximated from runs, replace with real match results later
    runs = pd.to_numeric(pf["runs"], errors="coerce").fillna(0.0)
    team_runs = (
        pf.assign(_r=runs)
        .groupby(["match_id", "team"], as_index=False)["_r"]
        .sum()
        .rename(columns={"_r": "team_total_runs"})
    )

    rows = []
    for mid, sub in team_runs.groupby("match_id"):
        teams = sorted(sub["team"].astype(str).unique().tolist())
        if len(teams) != 2:
            continue
        t1, t2 = teams[0], teams[1]
        r1 = float(sub[sub["team"] == t1]["team_total_runs"].sum())
        r2 = float(sub[sub["team"] == t2]["team_total_runs"].sum())
        if r1 > r2:
            win_label = "team1"
        elif r2 > r1:
            win_label = "team2"
        else:
            win_label = "tie"
        rows.append({"match_id": mid, "winner": win_label})

    return pd.DataFrame(rows)


def build_match_rows(
    team_df: pd.DataFrame,
    meta: pd.DataFrame,
    proxy_winners: pd.DataFrame,
    official_matches: pd.DataFrame | None,
    team_totals: pd.DataFrame | None,
) -> pd.DataFrame:
    rows = []
    official = official_matches if official_matches is not None else pd.DataFrame()

    for mid, sub in team_df.groupby("match_id"):
        teams = sorted(sub["team"].astype(str).unique().tolist())
        if len(teams) != 2:
            continue
        t1, t2 = teams[0], teams[1]
        r1 = sub[sub["team"] == t1]
        r2 = sub[sub["team"] == t2]
        if r1.empty or r2.empty:
            continue

        win_label, win_src = resolve_winner_label(mid, t1, t2, official, proxy_winners)

        row = {
            "match_id": mid,
            "team1_name": t1,
            "team2_name": t2,
            "team1_strength": float(r1["team_strength_score"].iloc[0]),
            "team2_strength": float(r2["team_strength_score"].iloc[0]),
            "team1_form_runs": float(r1["team_avg_form_runs"].iloc[0]),
            "team2_form_runs": float(r2["team_avg_form_runs"].iloc[0]),
            "team1_strike_rate": float(r1["team_avg_strike_rate"].iloc[0]),
            "team2_strike_rate": float(r2["team_avg_strike_rate"].iloc[0]),
            "team1_economy": float(r1["team_avg_economy"].iloc[0]),
            "team2_economy": float(r2["team_avg_economy"].iloc[0]),
            "winner": win_label,
            "winner_source": win_src,
        }
        if "team_avg_form_runs_ipl" in r1.columns:
            row["team1_form_runs_ipl"] = float(r1["team_avg_form_runs_ipl"].iloc[0])
            row["team2_form_runs_ipl"] = float(r2["team_avg_form_runs_ipl"].iloc[0])
        else:
            row["team1_form_runs_ipl"] = row["team1_form_runs"]
            row["team2_form_runs_ipl"] = row["team2_form_runs"]
        if team_totals is not None and len(team_totals):
            tt = team_totals[
                (team_totals["match_id"] == mid)
            ]
            t1r = tt[tt["team"].astype(str) == t1]["team_total_runs"]
            t2r = tt[tt["team"].astype(str) == t2]["team_total_runs"]
            row["team1_total_runs"] = float(t1r.iloc[0]) if len(t1r) else float("nan")
            row["team2_total_runs"] = float(t2r.iloc[0]) if len(t2r) else float("nan")
        else:
            row["team1_total_runs"] = float("nan")
            row["team2_total_runs"] = float("nan")

        m = meta[meta["match_id"] == mid] if len(meta) else pd.DataFrame()
        if not m.empty:
            row["venue"] = m["venue"].iloc[0]
            row["match_date"] = (
                m["match_date"].iloc[0] if "match_date" in m.columns else ""
            )
        else:
            row["venue"] = ""
            row["match_date"] = ""

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    pf = load_player_features_for_training()
    for c in ("runs", "wickets", "form_runs", "form_wickets", "strike_rate", "economy"):
        if c not in pf.columns:
            raise ValueError(f"player_features missing column: {c}")

    meta = None
    if PLAYER_MATCH_STATS_PATH.exists():
        m = pd.read_csv(
            PLAYER_MATCH_STATS_PATH,
            usecols=["match_id", "venue", "date"],
            low_memory=False,
        )
        meta = m.groupby("match_id", as_index=False).agg(
            venue=("venue", "first"),
            match_date=("date", "min"),
        )

    team_df = team_aggregates(pf)
    runs_num = pd.to_numeric(pf["runs"], errors="coerce").fillna(0.0)
    team_totals = (
        pf.assign(_r=runs_num)
        .groupby(["match_id", "team"], as_index=False)["_r"]
        .sum()
        .rename(columns={"_r": "team_total_runs"})
    )
    proxy_winners = match_winner_proxy(pf)
    official = load_official_ipl_matches()

    out = build_match_rows(
        team_df,
        meta if meta is not None else pd.DataFrame(),
        proxy_winners,
        official if len(official) else None,
        team_totals,
    )
    out = attach_momentum_columns_chronological(out)

    final_cols = [
        "match_id",
        "team1_name",
        "team2_name",
        "team1_strength",
        "team2_strength",
        "team1_form_runs",
        "team2_form_runs",
        "team1_form_runs_ipl",
        "team2_form_runs_ipl",
        "team1_strike_rate",
        "team2_strike_rate",
        "team1_economy",
        "team2_economy",
        "team1_total_runs",
        "team2_total_runs",
        "venue",
        "match_date",
        "team1_momentum",
        "team2_momentum",
        "winner",
        "winner_source",
    ]
    out = out[final_cols]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    strength_diff = out["team1_strength"] - out["team2_strength"]
    mask = out["winner"].isin(["team1", "team2"])
    y = (out.loc[mask, "winner"] == "team1").astype(float)
    x = strength_diff.loc[mask]
    corr = x.corr(y) if len(x) > 1 else float("nan")

    n_off = (out["winner_source"] == "official").sum() if "winner_source" in out.columns else 0
    n_px = (out["winner_source"] == "proxy").sum() if "winner_source" in out.columns else 0
    print(f"number of matches: {len(out)}")
    print(f"winner labels — official: {n_off}, proxy: {n_px}")
    print(f"correlation(team_strength_diff, team1_win): {corr:.4f}")
    print("sample rows:")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(out.head(3).to_string())
    print("match_training_dataset created")


if __name__ == "__main__":
    main()
