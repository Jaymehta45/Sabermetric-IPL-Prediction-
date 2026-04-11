"""
Aggregate franchise-level IPL signals from unified_ball_by_ball (innings-aware):

- Batting first vs chasing run productivity (vs league innings baselines)
- Powerplay batting burstiness
- Bowling stinginess when bowling in innings 1 vs 2 (runs conceded vs league)
- Dominance indices (batting / bowling / fielding) as league z-scores

Output: data/processed/team_franchise_profiles.csv (canonical franchise names).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from iplpred.core.franchise_normalize import canonical_franchise
from iplpred.paths import PROCESSED_DIR

UNIFIED_PATH = PROCESSED_DIR / "unified_ball_by_ball.csv"
OUTPUT_PATH = PROCESSED_DIR / "team_franchise_profiles.csv"

WICKET_EXCLUDED = frozenset({"run out", "retired hurt", "obstructing the field"})


def _ipl_rows(df: pd.DataFrame) -> pd.DataFrame:
    c = df["competition"].fillna("").astype(str).str.strip().str.upper()
    raw = df["competition"].fillna("").astype(str).str.lower()
    return df[c.eq("IPL") | raw.str.contains("indian premier", na=False) | raw.eq("")]


def _innings_table(u: pd.DataFrame) -> pd.DataFrame:
    u = u.copy()
    u["total_runs"] = pd.to_numeric(u["runs_off_bat"], errors="coerce").fillna(0.0) + pd.to_numeric(
        u["extras"], errors="coerce"
    ).fillna(0.0)
    u["innings"] = pd.to_numeric(u["innings"], errors="coerce")
    u["over"] = pd.to_numeric(u["over"], errors="coerce")
    u = u[u["innings"].notna() & (u["innings"] > 0)]

    u["is_pp"] = ((u["over"].notna()) & (u["over"] >= 0) & (u["over"] < 6)).astype(int)
    u["pp_row_runs"] = np.where(u["is_pp"] == 1, u["total_runs"], 0.0)

    dk = u["dismissal_kind"].fillna("").astype(str).str.strip().str.lower()
    pd_ = u["player_dismissed"].fillna("").astype(str).str.strip()
    bt = u["batter"].astype(str).str.strip()
    u["bat_out"] = ((pd_ == bt) & (dk != "") & (~dk.isin(WICKET_EXCLUDED))).astype(int)
    u["run_out_row"] = dk.str.contains("run out", na=False).astype(int)

    ag = (
        u.groupby(["match_id", "innings", "batting_team", "bowling_team"], as_index=False)
        .agg(
            runs=("total_runs", "sum"),
            pp_runs=("pp_row_runs", "sum"),
            wickets=("bat_out", "sum"),
            run_outs=("run_out_row", "sum"),
        )
    )
    ag["batting_team"] = ag["batting_team"].astype(str).str.strip()
    ag["bowling_team"] = ag["bowling_team"].astype(str).str.strip()
    return ag


def _z(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    m = float(x.mean())
    sd = float(x.std())
    if sd < 1e-9:
        return x * 0.0
    return (x - m) / sd


def build_profiles() -> pd.DataFrame:
    if not UNIFIED_PATH.is_file():
        return pd.DataFrame()

    raw = pd.read_csv(UNIFIED_PATH, low_memory=False)
    if "competition" not in raw.columns:
        raw["competition"] = ""
    u = _ipl_rows(raw)

    inn = _innings_table(u)
    if inn.empty:
        return pd.DataFrame()

    league_1 = float(inn.loc[inn["innings"] == 1, "runs"].mean())
    league_2 = float(inn.loc[inn["innings"] == 2, "runs"].mean())
    league_1 = max(league_1, 1.0)
    league_2 = max(league_2, 1.0)
    league_pp = float(inn["pp_runs"].mean())
    league_pp = max(league_pp, 1.0)

    bat1 = inn[inn["innings"] == 1].groupby("batting_team", as_index=False)["runs"].mean()
    bat1 = bat1.rename(columns={"batting_team": "team_raw", "runs": "mean_runs_bf"})
    bat2 = inn[inn["innings"] == 2].groupby("batting_team", as_index=False)["runs"].mean()
    bat2 = bat2.rename(columns={"batting_team": "team_raw", "runs": "mean_runs_chase"})

    ppb = inn.groupby("batting_team", as_index=False)["pp_runs"].mean()
    ppb = ppb.rename(columns={"batting_team": "team_raw", "pp_runs": "mean_pp_bat"})

    bowl1 = inn[inn["innings"] == 1].groupby("bowling_team", as_index=False)["runs"].mean()
    bowl1 = bowl1.rename(columns={"bowling_team": "team_raw", "runs": "mean_runs_conc_bf"})
    bowl2 = inn[inn["innings"] == 2].groupby("bowling_team", as_index=False)["runs"].mean()
    bowl2 = bowl2.rename(columns={"bowling_team": "team_raw", "runs": "mean_runs_conc_chase"})

    wk1 = inn[inn["innings"] == 1].groupby("bowling_team", as_index=False).agg(
        wk=("wickets", "mean"), ro=("run_outs", "mean")
    )
    wk1 = wk1.rename(columns={"bowling_team": "team_raw"})
    wk2 = inn[inn["innings"] == 2].groupby("bowling_team", as_index=False).agg(
        wk=("wickets", "mean"), ro=("run_outs", "mean")
    )
    wk2 = wk2.rename(columns={"bowling_team": "team_raw"})

    teams = (
        set(bat1["team_raw"])
        | set(bat2["team_raw"])
        | set(bowl1["team_raw"])
        | set(bowl2["team_raw"])
    )
    rows = []
    for t in sorted(teams):
        if not t or t.lower() == "nan":
            continue
        fr = canonical_franchise(t)
        r1 = bat1.loc[bat1["team_raw"] == t, "mean_runs_bf"]
        r2 = bat2.loc[bat2["team_raw"] == t, "mean_runs_chase"]
        pp = ppb.loc[ppb["team_raw"] == t, "mean_pp_bat"]
        c1 = bowl1.loc[bowl1["team_raw"] == t, "mean_runs_conc_bf"]
        c2 = bowl2.loc[bowl2["team_raw"] == t, "mean_runs_conc_chase"]
        wkf = wk1.loc[wk1["team_raw"] == t]
        wks = wk2.loc[wk2["team_raw"] == t]

        mbf = float(r1.iloc[0]) if len(r1) else float("nan")
        mch = float(r2.iloc[0]) if len(r2) else float("nan")
        mpp = float(pp.iloc[0]) if len(pp) else float("nan")
        mcb1 = float(c1.iloc[0]) if len(c1) else float("nan")
        mcb2 = float(c2.iloc[0]) if len(c2) else float("nan")
        w_a = float(wkf["wk"].iloc[0]) if len(wkf) else float("nan")
        r_a = float(wkf["ro"].iloc[0]) if len(wkf) else float("nan")
        w_b = float(wks["wk"].iloc[0]) if len(wks) else float("nan")
        r_b = float(wks["ro"].iloc[0]) if len(wks) else float("nan")

        n1 = int(((inn["innings"] == 1) & (inn["batting_team"] == t)).sum())
        n2 = int(((inn["innings"] == 2) & (inn["batting_team"] == t)).sum())

        rows.append(
            {
                "franchise": fr,
                "bat_first_ratio": mbf / league_1 if mbf == mbf else 1.0,
                "chase_ratio": mch / league_2 if mch == mch else 1.0,
                "pp_bat_ratio": mpp / league_pp if mpp == mpp else 1.0,
                "bowl_first_stingy": league_1 / mcb1 if mcb1 == mcb1 and mcb1 > 0 else 1.0,
                "bowl_second_stingy": league_2 / mcb2 if mcb2 == mcb2 and mcb2 > 0 else 1.0,
                "bat_dom_raw": (mbf + mch) / 2.0 if mbf == mbf and mch == mch else np.nan,
                "bowl_dom_raw": np.nanmean([w_a, w_b]) + 0.35 * np.nanmean([r_a, r_b])
                if all(x == x for x in (w_a, w_b))
                else np.nan,
                "field_dom_raw": np.nanmean([w_a, w_b]) + 0.25 * np.nanmean([r_a, r_b])
                if all(x == x for x in (w_a, w_b))
                else np.nan,
                "n_innings_bat_1": n1,
                "n_innings_bat_2": n2,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    med_bat = float(np.nanmedian(out["bat_dom_raw"]))
    med_bowl = float(np.nanmedian(out["bowl_dom_raw"]))
    med_f = float(np.nanmedian(out["field_dom_raw"]))
    out["bat_dom"] = _z(out["bat_dom_raw"].fillna(med_bat))
    out["bowl_dom"] = _z(out["bowl_dom_raw"].fillna(med_bowl))
    out["field_dom"] = _z(out["field_dom_raw"].fillna(med_f))
    out["bat_dom"] = pd.to_numeric(out["bat_dom"], errors="coerce").fillna(0.0)
    out["bowl_dom"] = pd.to_numeric(out["bowl_dom"], errors="coerce").fillna(0.0)
    out["field_dom"] = pd.to_numeric(out["field_dom"], errors="coerce").fillna(0.0)

    for c in (
        "bat_first_ratio",
        "chase_ratio",
        "pp_bat_ratio",
        "bowl_first_stingy",
        "bowl_second_stingy",
    ):
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[c] = out[c].fillna(1.0).clip(0.65, 1.45)

    out = out.drop(columns=["bat_dom_raw", "bowl_dom_raw", "field_dom_raw"], errors="ignore")

    merged_rows = []
    for fr, g in out.groupby("franchise", sort=False):
        ww = (
            g["n_innings_bat_1"].fillna(0).astype(float)
            + g["n_innings_bat_2"].fillna(0).astype(float)
            + 1.0
        )
        d: dict = {"franchise": fr}
        num_cols = [
            "bat_first_ratio",
            "chase_ratio",
            "pp_bat_ratio",
            "bowl_first_stingy",
            "bowl_second_stingy",
            "bat_dom",
            "bowl_dom",
            "field_dom",
        ]
        for c in num_cols:
            d[c] = float(np.average(pd.to_numeric(g[c], errors="coerce"), weights=ww))
        d["n_innings_bat_1"] = int(g["n_innings_bat_1"].fillna(0).sum())
        d["n_innings_bat_2"] = int(g["n_innings_bat_2"].fillna(0).sum())
        merged_rows.append(d)
    return pd.DataFrame(merged_rows)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    prof = build_profiles()
    if prof.empty:
        print("No franchise profiles (need innings + IPL rows in unified_ball_by_ball).")
        prof = pd.DataFrame(
            columns=[
                "franchise",
                "bat_first_ratio",
                "chase_ratio",
                "pp_bat_ratio",
                "bowl_first_stingy",
                "bowl_second_stingy",
                "bat_dom",
                "bowl_dom",
                "field_dom",
                "n_innings_bat_1",
                "n_innings_bat_2",
            ]
        )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prof.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(prof)} franchise rows → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
