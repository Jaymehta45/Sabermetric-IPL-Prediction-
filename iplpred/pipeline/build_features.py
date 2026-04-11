"""
Build player_features.csv with venue and matchup features (past data only).

IPL 2026 squad filtering (see squad_utils / ipl_2026_squads.csv) is used by
predict_playing_xi and match_simulator only; this build uses full historical
player_match_stats so models retain complete past seasons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from iplpred.core.ipl_franchises import load_franchise_team_names
from iplpred.paths import PROCESSED_DIR

PLAYER_MATCH_STATS_PATH = PROCESSED_DIR / "player_match_stats.csv"
UNIFIED_PATH = PROCESSED_DIR / "unified_ball_by_ball.csv"
PLAYER_FEATURES_PATH = PROCESSED_DIR / "player_features.csv"


def load_match_stats() -> pd.DataFrame:
    pms = pd.read_csv(PLAYER_MATCH_STATS_PATH, low_memory=False)
    pms["date"] = pd.to_datetime(pms["date"], errors="coerce")
    return pms.sort_values(["player_id", "date", "match_id"])


def build_base_columns(pms: pd.DataFrame) -> pd.DataFrame:
    """Form / SR / economy (pre-match rolling)."""
    pms = pms.copy()
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
    return pms


def add_ipl_and_calendar_features(pms: pd.DataFrame) -> pd.DataFrame:
    """
    IPL-only rolling form (last 5 franchise matches) and calendar gaps.
    Non-IPL rows do not contribute to the IPL rolling window.
    """
    pms = pms.copy()
    franchises = load_franchise_team_names()
    if not franchises:
        pms["form_runs_ipl"] = pms["form_runs"]
        pms["form_wickets_ipl"] = pms["form_wickets"]
        pms["days_since_prev_match"] = 0.0
        pms["is_franchise_match"] = 0
        return pms

    bt = pms["batting_team"].fillna("").astype(str).str.strip()
    bw = pms["bowling_team"].fillna("").astype(str).str.strip()
    ipl_ctx = bt.isin(franchises) | bw.isin(franchises)
    pms["is_franchise_match"] = ipl_ctx.astype(int)

    def _ipl_form_block(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        r = pd.to_numeric(g["runs"], errors="coerce").fillna(0.0).values
        w = pd.to_numeric(g["wickets"], errors="coerce").fillna(0.0).values
        ig = g["_ipl_ctx"].values
        n = len(g)
        out_r = np.zeros(n, dtype=np.float64)
        out_w = np.zeros(n, dtype=np.float64)
        hist_r: list[float] = []
        hist_w: list[float] = []
        for i in range(n):
            if i > 0:
                out_r[i] = float(np.mean(hist_r[-5:])) if hist_r else 0.0
                out_w[i] = float(np.mean(hist_w[-5:])) if hist_w else 0.0
            if ig[i]:
                hist_r.append(float(r[i]))
                hist_w.append(float(w[i]))
        return out_r, out_w

    pms["_ipl_ctx"] = ipl_ctx.values
    parts_r: list[np.ndarray] = []
    parts_w: list[np.ndarray] = []
    for _, g in pms.groupby("player_id", sort=False):
        orr, ow = _ipl_form_block(g)
        parts_r.append(orr)
        parts_w.append(ow)
    pms["form_runs_ipl"] = np.concatenate(parts_r)
    pms["form_wickets_ipl"] = np.concatenate(parts_w)
    pms.drop(columns=["_ipl_ctx"], inplace=True, errors="ignore")

    dt = pms.groupby("player_id")["date"].diff().dt.days
    pms["days_since_prev_match"] = (
        pd.to_numeric(dt, errors="coerce").fillna(180.0).clip(0.0, 730.0)
    )

    return pms


def add_venue_features(pms: pd.DataFrame) -> pd.DataFrame:
    """Prior-only cumulative stats at (player_id, venue)."""
    df = pms.sort_values(["player_id", "venue", "date", "match_id"]).copy()
    gcols = ["player_id", "venue"]
    gv = df.groupby(gcols, sort=False)

    df["venue_runs_sum"] = gv["runs"].cumsum().shift(1).fillna(0.0)
    df["venue_balls_sum"] = gv["balls"].cumsum().shift(1).fillna(0.0)
    df["venue_wickets_sum"] = gv["wickets"].cumsum().shift(1).fillna(0.0)
    df["venue_balls_bowled_sum"] = gv["balls_bowled"].cumsum().shift(1).fillna(0.0)
    df["venue_runs_conceded_sum"] = gv["runs_conceded"].cumsum().shift(1).fillna(0.0)
    df["matches_at_venue"] = gv.cumcount()

    df["real_venue"] = df["matches_at_venue"] > 0

    df["venue_avg_runs"] = np.where(
        df["matches_at_venue"] > 0,
        df["venue_runs_sum"] / df["matches_at_venue"],
        np.nan,
    )
    df["venue_strike_rate"] = np.where(
        df["venue_balls_sum"] > 0,
        df["venue_runs_sum"] / df["venue_balls_sum"] * 100.0,
        np.nan,
    )
    df["venue_avg_wickets"] = np.where(
        df["matches_at_venue"] > 0,
        df["venue_wickets_sum"] / df["matches_at_venue"],
        np.nan,
    )
    df["venue_economy"] = np.where(
        df["venue_balls_bowled_sum"] > 0,
        df["venue_runs_conceded_sum"] / (df["venue_balls_bowled_sum"] / 6.0),
        np.nan,
    )

    return df.sort_values(["player_id", "date", "match_id"]).reset_index(drop=True)


def add_venue_league_baselines(pms: pd.DataFrame) -> pd.DataFrame:
    """
    Venue-only priors: per match aggregate at venue, then expanding mean by venue (shift=1).
    """
    m = (
        pms.groupby(["match_id", "venue", "date"], as_index=False)
        .agg(
            sum_runs=("runs", "sum"),
            sum_balls=("balls", "sum"),
            sum_wk=("wickets", "sum"),
            sum_bb=("balls_bowled", "sum"),
            sum_rc=("runs_conceded", "sum"),
            npl=("player_id", "count"),
        )
        .sort_values(["venue", "date", "match_id"])
    )
    m["match_sr"] = np.where(m["sum_balls"] > 0, m["sum_runs"] / m["sum_balls"] * 100.0, np.nan)
    m["match_econ"] = np.where(m["sum_bb"] > 0, m["sum_rc"] / (m["sum_bb"] / 6.0), np.nan)
    m["avg_runs_per_player"] = m["sum_runs"] / m["npl"].replace(0, np.nan)
    m["avg_wk_per_player"] = m["sum_wk"] / m["npl"].replace(0, np.nan)

    gv = m.groupby("venue", sort=False)
    for col, outn in [
        ("avg_runs_per_player", "venue_avg_runs_all"),
        ("avg_wk_per_player", "venue_avg_wickets_all"),
        ("match_sr", "venue_strike_rate_all"),
        ("match_econ", "venue_economy_all"),
    ]:
        m[outn] = gv[col].transform(lambda s: s.expanding().mean().shift(1))

    keep = m[
        [
            "match_id",
            "venue_avg_runs_all",
            "venue_avg_wickets_all",
            "venue_strike_rate_all",
            "venue_economy_all",
        ]
    ].drop_duplicates("match_id")
    pms = pms.merge(keep, on="match_id", how="left")
    for c in [
        "venue_avg_runs_all",
        "venue_avg_wickets_all",
        "venue_strike_rate_all",
        "venue_economy_all",
    ]:
        pms[c] = pd.to_numeric(pms[c], errors="coerce")
    return pms


def build_pair_match_table(unified: pd.DataFrame, match_dates: pd.DataFrame) -> pd.DataFrame:
    u = unified.copy()
    u["batter"] = u["batter"].fillna("").astype(str).str.strip()
    u["bowler"] = u["bowler"].fillna("").astype(str).str.strip()
    u["runs_off_bat"] = pd.to_numeric(u["runs_off_bat"], errors="coerce").fillna(0.0)
    u["is_wide"] = pd.to_numeric(u["is_wide"], errors="coerce").fillna(0).astype(int)
    u["is_ball"] = (u["is_wide"] == 0).astype(int)
    pd_ = u["player_dismissed"].fillna("").astype(str).str.strip()
    bt_ = u["batter"].astype(str).str.strip()
    dk = u["dismissal_kind"].fillna("").astype(str).str.strip()
    u["dismissal"] = (
        (pd_ == bt_) & (dk != "") & (dk.str.lower() != "nan")
    ).astype(int)

    agg = (
        u.groupby(["match_id", "batter", "bowler"], as_index=False)
        .agg(
            pruns=("runs_off_bat", "sum"),
            pballs=("is_ball", "sum"),
            pdism=("dismissal", "sum"),
        )
    )
    agg["match_id"] = agg["match_id"].astype(int)
    match_dates = match_dates.copy()
    match_dates["match_id"] = match_dates["match_id"].astype(int)
    agg = agg.merge(match_dates, on="match_id", how="left")
    agg = agg.sort_values(["batter", "bowler", "date", "match_id"])
    return agg


def add_pair_priors(pair_agg: pd.DataFrame) -> pd.DataFrame:
    pair_agg = pair_agg.copy()
    g = pair_agg.groupby(["batter", "bowler"], sort=False)
    pair_agg["prev_runs"] = g["pruns"].cumsum().shift(1)
    pair_agg["prev_balls"] = g["pballs"].cumsum().shift(1)
    pair_agg["prev_dismissals"] = g["pdism"].cumsum().shift(1)
    pair_agg["prior_sr"] = np.where(
        pair_agg["prev_balls"] > 0,
        pair_agg["prev_runs"] / pair_agg["prev_balls"] * 100.0,
        np.nan,
    )
    return pair_agg


def weighted_sr_from_sub(sub: pd.DataFrame) -> tuple[float, bool]:
    """Ball-weighted prior SR from pair rows; real if any pair has prior balls."""
    parts: list[tuple[float, float]] = []
    real = False
    for _, r in sub.iterrows():
        w = float(r["pballs"])
        if w <= 0:
            continue
        prb = r["prev_balls"]
        psr = r["prior_sr"]
        if pd.notna(prb) and float(prb) > 0 and pd.notna(psr):
            real = True
            parts.append((w, float(psr)))
    if not parts:
        return np.nan, False
    tw = sum(p[0] for p in parts)
    return sum(p[0] * p[1] for p in parts) / tw, real


def aggregate_matchups_split(
    pair_agg: pd.DataFrame, match_id: int, player_id: str
) -> tuple[float, float, bool, bool]:
    """Batting SR (batter-only), bowling SR (bowler-only), real flags."""
    mid = int(match_id)
    pid = str(player_id).strip()
    sub_bat = pair_agg[(pair_agg["match_id"] == mid) & (pair_agg["batter"] == pid)]
    sub_bwl = pair_agg[(pair_agg["match_id"] == mid) & (pair_agg["bowler"] == pid)]
    b_sr, b_real = weighted_sr_from_sub(sub_bat)
    bw_sr, bw_real = weighted_sr_from_sub(sub_bwl)
    return b_sr, bw_sr, b_real, bw_real


def build_team_match_strength_past(pms: pd.DataFrame) -> pd.DataFrame:
    """
    Per (match_id, team): batting = sum of top 5 runs by players; bowling = sum of top 5 wickets.
    Then past_batting_strength / past_bowling_strength =
    shift(1).rolling(5).mean() over team history (no current match in window).
    """
    df = pms.copy()
    df["team"] = df["team"].fillna("").astype(str).str.strip()
    df["runs"] = pd.to_numeric(df["runs"], errors="coerce").fillna(0.0)
    df["wickets"] = pd.to_numeric(df["wickets"], errors="coerce").fillna(0.0)

    rows: list[dict] = []
    for (mid, team), g in df.groupby(["match_id", "team"], sort=False):
        g_runs = g.sort_values("runs", ascending=False).head(5)
        g_wk = g.sort_values("wickets", ascending=False).head(5)
        top5_runs = float(g_runs["runs"].sum())
        top5_wickets = float(g_wk["wickets"].sum())
        sum_runs_all = float(g["runs"].sum())
        sum_wickets_all = float(g["wickets"].sum())
        date = g["date"].iloc[0]
        rows.append(
            {
                "match_id": mid,
                "team": team,
                "top5_runs": top5_runs,
                "top5_wickets": top5_wickets,
                "sum_runs_all": sum_runs_all,
                "sum_wickets_all": sum_wickets_all,
                "date": date,
            }
        )

    tm = pd.DataFrame(rows)
    tm = tm.sort_values(["team", "date", "match_id"])
    tm["past_batting_strength"] = tm.groupby("team", sort=False)["top5_runs"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    tm["past_bowling_strength"] = tm.groupby("team", sort=False)["top5_wickets"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    return tm


def opponent_team_for_row(row: pd.Series) -> str:
    t = str(row.get("team", "")).strip()
    bt = str(row.get("batting_team", "")).strip()
    bw = str(row.get("bowling_team", "")).strip()
    if t == bt:
        return bw
    if t == bw:
        return bt
    return ""


def add_opponent_strength_features(pms: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    tm = tm.copy()
    tm["team"] = tm["team"].astype(str).str.strip()
    opp = tm.rename(
        columns={
            "team": "opponent_team",
            "past_batting_strength": "opponent_past_batting_strength",
            "past_bowling_strength": "opponent_past_bowling_strength",
        }
    )[
        [
            "match_id",
            "opponent_team",
            "opponent_past_batting_strength",
            "opponent_past_bowling_strength",
        ]
    ]

    pms = pms.copy()
    pms["team"] = pms["team"].fillna("").astype(str).str.strip()
    pms["opponent_team"] = pms.apply(opponent_team_for_row, axis=1)
    pms["opponent_team"] = pms["opponent_team"].replace("", np.nan).astype(str).str.strip()
    pms["opponent_team"] = pms["opponent_team"].replace("", np.nan)

    pms = pms.merge(opp, on=["match_id", "opponent_team"], how="left")

    pms["role"] = pms["role"].fillna("").astype(str).str.strip().str.lower()
    role = pms["role"]
    pms["opponent_strength"] = np.where(
        role == "batter",
        pms["opponent_past_bowling_strength"],
        np.where(
            role == "bowler",
            pms["opponent_past_batting_strength"],
            np.where(
                role == "allrounder",
                (
                    pms["opponent_past_bowling_strength"].astype(float)
                    + pms["opponent_past_batting_strength"].astype(float)
                )
                / 2.0,
                np.nan,
            ),
        ),
    )

    for c in (
        "opponent_past_batting_strength",
        "opponent_past_bowling_strength",
        "opponent_strength",
    ):
        pms[c] = pd.to_numeric(pms[c], errors="coerce")
        pms[c] = pms[c].fillna(pms[c].mean())

    return pms


def add_play_style_archetypes(pms: pd.DataFrame) -> pd.DataFrame:
    """
    Compact role-aware style tags from rolling strike rate / economy / form
    (aggressive vs anchor batters; economical vs expensive wicket-taking bowlers).
    """
    pms = pms.copy()
    sr = pd.to_numeric(pms["strike_rate"], errors="coerce").fillna(120.0)
    er = pd.to_numeric(pms["economy"], errors="coerce").fillna(8.5)
    fr = pd.to_numeric(pms["form_runs"], errors="coerce").fillna(0.0)
    fw = pd.to_numeric(pms["form_wickets"], errors="coerce").fillna(0.0)
    role = pms["role"].fillna("").astype(str).str.strip().str.lower()
    is_bat = role.eq("batter") | role.eq("allrounder")
    is_bowl = role.eq("bowler") | role.eq("allrounder")
    pms["style_bat_aggressive"] = np.where(
        is_bat, np.clip((sr - 126.0) / 38.0, -1.2, 1.4), 0.0
    )
    pms["style_bat_anchor"] = np.where(
        is_bat,
        np.clip((122.0 - sr) / 28.0, 0.0, 1.3) * np.clip(fr / 32.0, 0.15, 1.0),
        0.0,
    )
    pms["style_bowl_economical"] = np.where(
        is_bowl, np.clip((8.25 - er) / 2.8, -0.8, 1.4), 0.0
    )
    pms["style_bowl_expensive_wicket_taker"] = np.where(
        is_bowl,
        np.clip((er - 8.1) / 2.9, 0.0, 1.5) * np.clip(fw / 2.2, 0.0, 1.1),
        0.0,
    )
    return pms


def leakage_check_opponent_past(pms: pd.DataFrame, tm: pd.DataFrame) -> None:
    """Confirm opponent_past_* is not identical to opponent's current-match top5 totals."""
    cur = tm.rename(
        columns={
            "team": "opponent_team",
            "top5_runs": "opp_top5_runs_this_match",
            "top5_wickets": "opp_top5_wickets_this_match",
        }
    )[
        [
            "match_id",
            "opponent_team",
            "opp_top5_runs_this_match",
            "opp_top5_wickets_this_match",
        ]
    ]
    chk = pms[["match_id", "player_id", "opponent_team", "opponent_past_batting_strength"]].merge(
        cur, on=["match_id", "opponent_team"], how="left"
    )
    chk = chk.dropna(subset=["opp_top5_runs_this_match", "opponent_past_batting_strength"])
    if len(chk) == 0:
        print("Leakage check: insufficient rows for correlation check.")
        return
    r_runs = chk["opponent_past_batting_strength"].corr(chk["opp_top5_runs_this_match"])
    exact_same = (chk["opponent_past_batting_strength"] == chk["opp_top5_runs_this_match"]).mean()
    print(
        "Leakage check: opponent_past_batting_strength vs opponent current match top5_runs "
        f"(same row) — correlation={r_runs:.4f}, fraction exact match={exact_same:.4f} "
        "(expect correlation < 1 and low exact-match rate; past uses shift(1).rolling(5))."
    )


def print_old_vs_new_strength_sample(tm: pd.DataFrame, n: int = 8) -> None:
    """Compare full squad sums vs top-5 sums (per team-match, raw match row)."""
    samp = tm.head(n)[
        [
            "match_id",
            "team",
            "sum_runs_all",
            "top5_runs",
            "sum_wickets_all",
            "top5_wickets",
        ]
    ].copy()
    samp["batting: old(sum) vs new(top5)"] = (
        samp["sum_runs_all"].round(1).astype(str) + " vs " + samp["top5_runs"].round(1).astype(str)
    )
    samp["bowling: old(sum) vs new(top5)"] = (
        samp["sum_wickets_all"].round(1).astype(str) + " vs " + samp["top5_wickets"].round(1).astype(str)
    )
    print()
    print("Old vs new team strength (same match, before rolling) — sample rows:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(
            samp[
                [
                    "match_id",
                    "team",
                    "batting: old(sum) vs new(top5)",
                    "bowling: old(sum) vs new(top5)",
                ]
            ].to_string(index=False)
        )


def apply_venue_fallback(
    pms: pd.DataFrame,
) -> None:
    """When no player+venue history: 0.7 * player global + 0.3 * venue league prior."""
    mask = ~pms["real_venue"]
    pms["uses_venue_fallback"] = mask

    pms.loc[mask, "venue_avg_runs"] = (
        0.7 * pms.loc[mask, "pglobal_avg_runs"]
        + 0.3 * pms.loc[mask, "venue_avg_runs_all"].fillna(pms.loc[mask, "pglobal_avg_runs"])
    )
    pms.loc[mask, "venue_avg_wickets"] = (
        0.7 * pms.loc[mask, "pglobal_avg_wk"]
        + 0.3 * pms.loc[mask, "venue_avg_wickets_all"].fillna(pms.loc[mask, "pglobal_avg_wk"])
    )
    pms.loc[mask, "venue_strike_rate"] = (
        0.7 * pms.loc[mask, "pglobal_sr"]
        + 0.3 * pms.loc[mask, "venue_strike_rate_all"].fillna(pms.loc[mask, "pglobal_sr"])
    )
    pms.loc[mask, "venue_economy"] = (
        0.7 * pms.loc[mask, "pglobal_econ"]
        + 0.3 * pms.loc[mask, "venue_economy_all"].fillna(pms.loc[mask, "pglobal_econ"])
    )

    still = pms["venue_avg_runs"].isna() & mask
    if still.any():
        pms.loc[still, "venue_avg_runs"] = pms.loc[still, "pglobal_avg_runs"]
    for col, pg in [
        ("venue_avg_wickets", "pglobal_avg_wk"),
        ("venue_strike_rate", "pglobal_sr"),
        ("venue_economy", "pglobal_econ"),
    ]:
        st = pms[col].isna()
        if st.any():
            pms.loc[st, col] = pms.loc[st, pg]


def main() -> None:
    pms = load_match_stats()
    pms = build_base_columns(pms)
    pms = add_ipl_and_calendar_features(pms)
    pms = add_venue_features(pms)
    pms = add_venue_league_baselines(pms)

    unified = pd.read_csv(UNIFIED_PATH, low_memory=False)
    match_dates = pms[["match_id", "date"]].drop_duplicates("match_id")
    pair_agg = build_pair_match_table(unified, match_dates)
    pair_agg = add_pair_priors(pair_agg)

    fb = (
        pms.groupby("player_id", as_index=False)
        .agg(
            pglobal_avg_runs=("runs", "mean"),
            pglobal_avg_wk=("wickets", "mean"),
            pglobal_sr=("match_sr", "mean"),
            pglobal_econ=("match_econ", "mean"),
        )
    )
    pms = pms.merge(fb, on="player_id", how="left")

    bat_sr: list[float] = []
    bow_sr: list[float] = []
    real_bat: list[bool] = []
    real_bow: list[bool] = []

    for _, row in pms.iterrows():
        b_sr, bw_sr, br, bwr = aggregate_matchups_split(
            pair_agg, int(row["match_id"]), str(row["player_id"])
        )
        bat_sr.append(b_sr)
        bow_sr.append(bw_sr)
        real_bat.append(br)
        real_bow.append(bwr)

    pms["batting_matchup_sr"] = bat_sr
    pms["bowling_matchup_sr"] = bow_sr
    pms["batting_matchup_sr"] = pms["batting_matchup_sr"].fillna(pms["pglobal_sr"])
    pms["bowling_matchup_sr"] = pms["bowling_matchup_sr"].fillna(pms["pglobal_sr"])

    apply_venue_fallback(pms)

    tm_team = build_team_match_strength_past(pms)
    pms = add_opponent_strength_features(pms, tm_team)
    pms = add_play_style_archetypes(pms)
    print_old_vs_new_strength_sample(tm_team)
    leakage_check_opponent_past(pms, tm_team)

    pct_real_bat = 100.0 * float(np.mean(real_bat))
    pct_real_bow = 100.0 * float(np.mean(real_bow))
    pct_venue_fb = 100.0 * float(pms["uses_venue_fallback"].mean())

    out = pms[
        [
            "match_id",
            "player_id",
            "team",
            "runs",
            "wickets",
            "form_runs",
            "form_wickets",
            "form_runs_ipl",
            "form_wickets_ipl",
            "days_since_prev_match",
            "is_franchise_match",
            "strike_rate",
            "economy",
            "venue_avg_runs",
            "venue_avg_wickets",
            "venue_strike_rate",
            "venue_economy",
            "venue_avg_runs_all",
            "venue_avg_wickets_all",
            "venue_strike_rate_all",
            "venue_economy_all",
            "batting_matchup_sr",
            "bowling_matchup_sr",
            "opponent_team",
            "opponent_past_batting_strength",
            "opponent_past_bowling_strength",
            "opponent_strength",
            "style_bat_aggressive",
            "style_bat_anchor",
            "style_bowl_economical",
            "style_bowl_expensive_wicket_taker",
        ]
    ].copy()

    PLAYER_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PLAYER_FEATURES_PATH, index=False)

    print(f"% rows with real batting matchups: {pct_real_bat:.2f}")
    print(f"% rows with real bowling matchups: {pct_real_bow:.2f}")
    print(f"% rows using venue fallback: {pct_venue_fb:.2f}")
    print()
    print("Sample rows (opponent_strength context):")
    sample_cols = [
        "match_id",
        "player_id",
        "team",
        "role",
        "opponent_team",
        "opponent_past_batting_strength",
        "opponent_past_bowling_strength",
        "opponent_strength",
    ]
    samp = pms[sample_cols].head(5)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(samp.to_string(index=False))
    print()
    print("Saved:", PLAYER_FEATURES_PATH)


if __name__ == "__main__":
    main()
