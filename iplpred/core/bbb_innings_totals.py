"""Per-innings team totals from a ball-by-ball CSV (matches/ schema)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def innings_runs_and_wickets_from_bbb(csv_path: Path) -> tuple[
    float | None,
    float | None,
    int | None,
    int | None,
]:
    """
    Returns (1st_inn_runs, 2nd_inn_runs, 1st_inn_wkts, 2nd_inn_wkts).
    Runs = sum of runs_batter + runs_extras per innings; wickets = count of wicket rows per innings.
    """
    if not csv_path.is_file():
        return None, None, None, None
    df = pd.read_csv(csv_path, low_memory=False)
    if "innings" not in df.columns:
        return None, None, None, None
    rb = "runs_batter" in df.columns
    rx = "runs_extras" in df.columns
    if not rb and not rx:
        return None, None, None, None
    d = df.copy()
    if rb:
        d["runs_batter"] = pd.to_numeric(d["runs_batter"], errors="coerce").fillna(0.0)
    else:
        d["runs_batter"] = 0.0
    if rx:
        d["runs_extras"] = pd.to_numeric(d["runs_extras"], errors="coerce").fillna(0.0)
    else:
        d["runs_extras"] = 0.0
    d["runs_total"] = d["runs_batter"] + d["runs_extras"]
    wcol = d["wicket"] if "wicket" in d.columns else None
    out_r: list[float | None] = []
    out_w: list[int | None] = []
    for inn in (1, 2):
        sub = d[d["innings"] == inn]
        if len(sub) == 0:
            out_r.append(None)
            out_w.append(None)
            continue
        out_r.append(float(sub["runs_total"].sum()))
        if wcol is not None:
            wk = sub["wicket"]
            if wk.dtype == object:
                mask = wk.astype(str).str.lower().isin(["true", "1", "yes"])
            else:
                mask = wk == True  # noqa: E712
            out_w.append(int(mask.sum()))
        else:
            out_w.append(None)
    return out_r[0], out_r[1], out_w[0], out_w[1]
