"""
IPLpred web UI — FastAPI + Jinja2 (read-only: prediction log + pred vs actual).

Run from repository root:
  pip install -r requirements-web.txt
  uvicorn web.dashboard:app --reload --host 127.0.0.1 --port 8000

Vercel: root ``app.py`` re-exports this app (Fluid / zero-config). Install ``pip install -r requirements.txt``.

For a public URL via Cloudflare Try, use ``make tunnel`` (see ``scripts/cloudflared_tunnel.sh``)
after the app responds on the same host/port (``127.0.0.1`` avoids some localhost DNS issues).

Pre-match predictions: use predict_match_outcomes.py and scripts/log_prediction.py;
this app does not accept XI or match parameters.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

import pandas as pd

from iplpred.core.bbb_innings_totals import innings_runs_and_wickets_from_bbb

from web.bbb_utils import (
    ordered_lists_match_slugs,
    pair_slug_match,
    top3_bowlers_from_bbb,
    top5_batters_from_bbb,
)
from web.prediction_log_io import prediction_log_meta, read_prediction_log_dataframe

WEB_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))
# Bump when CSS/JS change so browsers fetch fresh static files (cache-bust query string).
UI_BUILD_ID = "202604282"
templates.env.globals["asset_version"] = UI_BUILD_ID
templates.env.globals["ui_build_id"] = UI_BUILD_ID

app = FastAPI(title="IPLpred", version="1.0")


class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    """Stop browsers from keeping stale JS/CSS (common reason UI 'never updates')."""

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        path = request.scope.get("path") or ""
        if path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
        return response


app.add_middleware(NoCacheStaticMiddleware)
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def page_home(request: Request):
    stats = _dashboard_stats()
    recent = []
    df = read_prediction_log_dataframe()
    if df is not None and len(df) > 0:
        recent = _comparison_rows(df)[:8]
    return templates.TemplateResponse(
        request,
        "index.html",
        {"request": request, "stats": stats, "recent": recent},
    )


@app.get("/history", response_class=HTMLResponse)
def page_history(request: Request):
    df = read_prediction_log_dataframe()
    if df is None or len(df) == 0:
        return templates.TemplateResponse(
            request,
            "history.html",
            {
                "request": request,
                "comparison_rows": [],
                "match_filter_options": [],
                "chart_summary": _log_chart_summary(pd.DataFrame()),
                "stats": _dashboard_stats(),
                "log_exists": False,
            },
        )
    chart_summary = _log_chart_summary(df)
    stats = _dashboard_stats()
    comparison_rows = _comparison_rows(df)
    match_filter_options = _match_filter_options(df)
    return templates.TemplateResponse(
        request,
        "history.html",
        {
            "request": request,
            "comparison_rows": comparison_rows,
            "match_filter_options": match_filter_options,
            "chart_summary": chart_summary,
            "stats": stats,
            "log_exists": True,
        },
    )


def _process_step_flags(row: pd.Series) -> tuple[bool, bool, bool]:
    """P1 pre logged, P2 post/actuals logged, P3 review logged (see DAILY_WORKFLOW)."""
    p1 = bool(_safe_str(row.get("logged_at_pre")))
    p2 = bool(_safe_str(row.get("logged_at_post"))) and bool(
        _safe_str(row.get("actual_winner"))
    )
    p3 = bool(_safe_str(row.get("process3_completed_at")))
    return p1, p2, p3


def _dashboard_stats() -> dict:
    out = {
        "n_logged": 0,
        "n_with_actual": 0,
        "n_process1": 0,
        "n_process2": 0,
        "n_process3": 0,
        "winner_hits": 0,
        "winner_total": 0,
        "potm_hits": 0,
        "potm_total": 0,
        "winner_pct_label": "—",
        "potm_pct_label": "—",
        "innings_mae_avg_label": "—",
        "innings_mae_n": 0,
    }
    df = read_prediction_log_dataframe()
    if df is None or len(df) == 0:
        return out
    out["n_logged"] = len(df)
    for _, row in df.iterrows():
        p1, p2, p3 = _process_step_flags(row)
        if p1:
            out["n_process1"] += 1
        if p2:
            out["n_process2"] += 1
        if p3:
            out["n_process3"] += 1
    if "actual_winner" not in df.columns:
        return out
    has = df["actual_winner"].fillna("").astype(str).str.strip().ne("")
    out["n_with_actual"] = int(has.sum())
    sub = df[has]
    if len(sub) == 0:
        return out
    wc = sub.get("winner_correct")
    if wc is not None:
        valid = wc.notna() & (wc.astype(str).str.lower().isin(["true", "false"]))
        hits = (sub.loc[valid, "winner_correct"].astype(str).str.lower() == "true").sum()
        tot = valid.sum()
        out["winner_hits"] = int(hits)
        out["winner_total"] = int(tot)
    pc = sub.get("potm_correct")
    if pc is not None:
        valid = pc.notna() & (pc.astype(str).str.lower().isin(["true", "false"]))
        hits = (sub.loc[valid, "potm_correct"].astype(str).str.lower() == "true").sum()
        tot = valid.sum()
        out["potm_hits"] = int(hits)
        out["potm_total"] = int(tot)
    if out["winner_total"] > 0:
        out["winner_pct_label"] = f"{round(100 * out['winner_hits'] / out['winner_total'])}%"
    if out["potm_total"] > 0:
        out["potm_pct_label"] = f"{round(100 * out['potm_hits'] / out['potm_total'])}%"

    mae_list: list[float] = []
    for _, row in df.iterrows():
        pex = _parse_pred_extra_json(row.get("pred_extra_json"))
        a1 = _parse_optional_float(row.get("actual_first_innings_team_runs"))
        a2 = _parse_optional_float(row.get("actual_second_innings_team_runs"))
        _, mae = _innings_run_compare(pex, a1, a2)
        if mae is not None:
            mae_list.append(mae)
    if mae_list:
        out["innings_mae_n"] = len(mae_list)
        out["innings_mae_avg_label"] = f"{sum(mae_list) / len(mae_list):.1f}"
    return out


def _bool_cell(val) -> bool | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def _safe_str(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    if s.lower() in ("nan", "none", "<na>"):
        return ""
    return s


def _team_initials(name: str) -> str:
    """Short label for tables, e.g. Royal Challengers Bengaluru -> RCB."""
    name = (name or "").strip()
    if not name:
        return "?"
    parts = [p for p in re.split(r"\s+", name) if p]
    if len(parts) >= 2:
        return "".join(p[0].upper() for p in parts[: min(4, len(parts))])
    return (parts[0][:5] + "…") if len(parts[0]) > 6 else parts[0]


def _grade_label(wc: bool | None, has_actual: bool) -> str:
    if not has_actual:
        return "waiting"
    if wc is True:
        return "right"
    if wc is False:
        return "wrong"
    return "unknown"


def _parse_pred_extra_json(val) -> dict:
    """Decode pred_extra_json column from the prediction log (may be empty)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return {}
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "<na>"):
        return {}
    try:
        j = json.loads(s)
        return j if isinstance(j, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _parse_optional_float(val) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "<na>"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _format_pred_innings_line(
    team1_name: str,
    team2_name: str,
    pex: dict,
) -> str:
    """Human-readable runs/wickets per innings from pred_extra (team1 = 1st)."""
    if not pex:
        return ""
    try:
        r1 = pex.get("pred_first_innings_team_runs")
        r2 = pex.get("pred_second_innings_team_runs")
        w1 = pex.get("pred_first_innings_team_wickets")
        w2 = pex.get("pred_second_innings_team_wickets")
        if r1 is None or r2 is None:
            return ""
        a1 = _team_initials(team1_name)
        a2 = _team_initials(team2_name)
        if w1 is not None and w2 is not None:
            return (
                f"{a1} {float(r1):.0f}/{float(w1):.0f} (1st) · "
                f"{a2} {float(r2):.0f}/{float(w2):.0f} (2nd)"
            )
        return f"{a1} {float(r1):.0f} (1st) · {a2} {float(r2):.0f} (2nd)"
    except (TypeError, ValueError):
        return ""


def _format_actual_innings_line(
    team1_name: str,
    team2_name: str,
    runs1: float | None,
    runs2: float | None,
    wk1: int | None,
    wk2: int | None,
) -> str:
    """Same shape as pred line when we have actual runs (wk optional)."""
    if runs1 is None or runs2 is None:
        return ""
    try:
        a1 = _team_initials(team1_name)
        a2 = _team_initials(team2_name)
        r1, r2 = float(runs1), float(runs2)
        if wk1 is not None and wk2 is not None:
            return (
                f"{a1} {r1:.0f}/{int(wk1)} (1st) · "
                f"{a2} {r2:.0f}/{int(wk2)} (2nd)"
            )
        return f"{a1} {r1:.0f} (1st) · {a2} {r2:.0f} (2nd)"
    except (TypeError, ValueError):
        return ""


def _innings_run_compare(
    pex: dict,
    actual_r1: float | None,
    actual_r2: float | None,
) -> tuple[str, float | None]:
    """
    Returns (short summary string, MAE in runs) for pred vs actual team totals.
    Delta = actual − predicted (positive ⇒ team scored more than predicted).
    """
    if not pex or actual_r1 is None or actual_r2 is None:
        return "", None
    try:
        pr1 = pex.get("pred_first_innings_team_runs")
        pr2 = pex.get("pred_second_innings_team_runs")
        if pr1 is None or pr2 is None:
            return "", None
        pr1, pr2 = float(pr1), float(pr2)
        d1 = float(actual_r1) - pr1
        d2 = float(actual_r2) - pr2
        mae = (abs(d1) + abs(d2)) / 2.0
        s = f"1st Δ {d1:+.0f} · 2nd Δ {d2:+.0f} · MAE {mae:.1f}"
        return s, mae
    except (TypeError, ValueError):
        return "", None


def _parse_player_list(val) -> list[str]:
    """CSV may store JSON array string for pred_top5_bats / pred_top3_bowls."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "<na>"):
        return []
    try:
        j = json.loads(s)
        if isinstance(j, list):
            return [str(x).strip() for x in j if str(x).strip()]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _resolve_bbb_path(rel: str) -> Path | None:
    if not (rel or "").strip():
        return None
    rel = rel.strip().lstrip("/")
    cand = ROOT / rel
    return cand if cand.is_file() else None


def _grade_top5_ordered(
    pred: list[str], actual: list[str], has_actual: bool, can_compare_full: bool
) -> str:
    if not has_actual:
        return "waiting"
    if not can_compare_full:
        return "na"
    return "right" if ordered_lists_match_slugs(pred, actual[:5]) else "wrong"


def _grade_top3_ordered(
    pred: list[str], actual: list[str], has_actual: bool, can_compare_full: bool
) -> str:
    if not has_actual:
        return "waiting"
    if not can_compare_full:
        return "na"
    return "right" if ordered_lists_match_slugs(pred, actual[:3]) else "wrong"


def _grade_best_pick(pred: str, actual: str, has_actual: bool) -> str:
    if not has_actual:
        return "waiting"
    if not (pred or "").strip() or not (actual or "").strip():
        return "waiting"
    return "right" if pair_slug_match(pred, actual) else "wrong"


def _sort_log_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Newest match_date first; same calendar day → lexicographically later match_key first (match9 before match8)."""
    if df.empty:
        return df
    d = df.copy()
    if "match_date" not in d.columns:
        return d.sort_index(ascending=False)
    d["_sort"] = pd.to_datetime(d["match_date"], errors="coerce")
    if "match_key" in d.columns:
        d["_mk"] = d["match_key"].astype(str)
        d = d.sort_values(["_sort", "_mk"], ascending=[False, False], na_position="last")
        d = d.drop(columns=["_mk"], errors="ignore")
    else:
        d = d.sort_values("_sort", ascending=False, na_position="last")
    return d.drop(columns=["_sort"], errors="ignore")


def _comparison_rows(df: pd.DataFrame) -> list[dict]:
    """Human-readable predicted vs actual rows for the UI (newest first)."""
    if df.empty:
        return []
    d = _sort_log_for_display(df)
    rows: list[dict] = []
    for _, row in d.iterrows():
        mk = _safe_str(row.get("match_key"))
        md = _safe_str(row.get("match_date"))
        t1 = _safe_str(row.get("team1_name"))
        t2 = _safe_str(row.get("team2_name"))
        teams = f"{t1} vs {t2}" if (t1 and t2) else "—"
        actual_w = _safe_str(row.get("actual_winner"))
        has_actual = bool(actual_w)
        pred_w = _safe_str(row.get("pred_winner"))
        wc = _bool_cell(row.get("winner_correct"))
        pred_potm = _safe_str(row.get("pred_potm"))
        actual_potm = _safe_str(row.get("official_potm"))
        pc = _bool_cell(row.get("potm_correct"))
        p1_raw = row.get("pred_p_team1_win")
        p1_pct = ""
        p1_f: float | None = None
        try:
            if p1_raw is not None and str(p1_raw).strip() != "" and not (
                isinstance(p1_raw, float) and pd.isna(p1_raw)
            ):
                p1_f = float(p1_raw)
                p1_f = max(0.0, min(1.0, p1_f))
                p1_pct = f"{p1_f * 100:.0f}%"
        except (TypeError, ValueError):
            p1_f = None
        a1, a2 = _team_initials(t1), _team_initials(t2)
        if p1_f is not None and t1 and t2:
            p2_pct = f"{(1.0 - p1_f) * 100:.0f}%"
            win_chance_readable = f"{a1} {p1_pct} · {a2} {p2_pct}"
            win_chance_note = (
                f"About {p1_pct} for {t1.split()[0] if t1 else a1}, "
                f"{p2_pct} for {t2.split()[0] if t2 else a2} (model estimate)."
            )
        elif p1_f is not None:
            win_chance_readable = f"{a1} {p1_pct} (other side {(1.0 - p1_f) * 100:.0f}%)"
            win_chance_note = "Split is for the two teams in the log, in order."
        else:
            win_chance_readable = "—"
            win_chance_note = ""
        actual_ts = _safe_str(row.get("actual_top_scorer"))
        actual_tb = _safe_str(row.get("actual_top_wicket_bowler"))
        has_logged_top_scorer = bool(actual_ts.strip())
        has_logged_top_wicket = bool(actual_tb.strip())
        pred_top5_list = _parse_player_list(row.get("pred_top5_bats"))
        pred_top3_list = _parse_player_list(row.get("pred_top3_bowls"))
        pred_best_bat = _safe_str(row.get("pred_best_bat"))
        pred_best_bowl = _safe_str(row.get("pred_best_bowl"))

        bbb_path = _resolve_bbb_path(_safe_str(row.get("bbb_match_file")))
        actual_top5_bbb: list[str] = []
        actual_top3_bbb: list[str] = []
        has_bbb = False
        if bbb_path is not None:
            try:
                actual_top5_bbb = top5_batters_from_bbb(bbb_path)
                actual_top3_bbb = top3_bowlers_from_bbb(bbb_path)
                has_bbb = bool(actual_top5_bbb or actual_top3_bbb)
            except Exception:
                actual_top5_bbb, actual_top3_bbb = [], []

        has_match_evidence = has_actual or has_bbb

        can_top5 = (
            has_bbb
            and len(actual_top5_bbb) >= 5
            and len(pred_top5_list) >= 5
        )
        can_top3 = (
            has_bbb
            and len(actual_top3_bbb) >= 3
            and len(pred_top3_list) >= 3
        )
        top5_list_grade = _grade_top5_ordered(
            pred_top5_list, actual_top5_bbb, has_match_evidence, can_top5
        )
        top3_list_grade = _grade_top3_ordered(
            pred_top3_list, actual_top3_bbb, has_match_evidence, can_top3
        )
        best_bat_pick_grade = _grade_best_pick(
            pred_best_bat, actual_ts, has_logged_top_scorer
        )
        best_bowl_pick_grade = _grade_best_pick(
            pred_best_bowl, actual_tb, has_logged_top_wicket
        )

        p1_done, p2_done, p3_done = _process_step_flags(row)
        p3_notes = _safe_str(row.get("process3_notes"))
        pex = _parse_pred_extra_json(row.get("pred_extra_json"))
        pred_innings_line = _format_pred_innings_line(t1, t2, pex)

        ar1 = _parse_optional_float(row.get("actual_first_innings_team_runs"))
        ar2 = _parse_optional_float(row.get("actual_second_innings_team_runs"))
        aw1: int | None = None
        aw2: int | None = None
        bbb_path_innings = _resolve_bbb_path(_safe_str(row.get("bbb_match_file")))
        if bbb_path_innings is not None and ar1 is not None and ar2 is not None:
            try:
                _, _, w1b, w2b = innings_runs_and_wickets_from_bbb(bbb_path_innings)
                aw1, aw2 = w1b, w2b
            except Exception:
                aw1, aw2 = None, None
        actual_innings_line = _format_actual_innings_line(t1, t2, ar1, ar2, aw1, aw2)
        innings_err_summary, innings_mae = _innings_run_compare(pex, ar1, ar2)
        has_innings_actual = ar1 is not None and ar2 is not None

        rows.append(
            {
                "match_key": mk,
                "match_date": md,
                "teams": teams,
                "process_p1_done": p1_done,
                "process_p2_done": p2_done,
                "process_p3_done": p3_done,
                "process3_notes": p3_notes,
                "team1_short": t1,
                "team2_short": t2,
                "pred_winner": pred_w or "—",
                "actual_winner": actual_w or "—",
                "winner_grade": _grade_label(wc, has_actual),
                "pred_star_player": pred_potm or "—",
                "actual_star_player": actual_potm or "—",
                "star_player_grade": _grade_label(pc, has_actual),
                "pred_top_runs": pred_best_bat or "—",
                "actual_top_runs": actual_ts or "—",
                "pred_top_wickets": pred_best_bowl or "—",
                "actual_top_wickets": actual_tb or "—",
                "pred_top5_list": pred_top5_list,
                "pred_top3_list": pred_top3_list,
                "actual_top5_bbb_list": actual_top5_bbb,
                "actual_top3_bbb_list": actual_top3_bbb,
                "has_bbb": has_bbb,
                "top5_list_grade": top5_list_grade,
                "top3_list_grade": top3_list_grade,
                "best_bat_pick_grade": best_bat_pick_grade,
                "best_bowl_pick_grade": best_bowl_pick_grade,
                "win_chance_readable": win_chance_readable,
                "win_chance_note": win_chance_note,
                "pred_innings_line": pred_innings_line,
                "actual_innings_line": actual_innings_line,
                "innings_err_summary": innings_err_summary,
                "innings_mae": innings_mae,
                "has_innings_actual": has_innings_actual,
            }
        )
    return rows


def _match_filter_options(df: pd.DataFrame) -> list[dict]:
    """Dropdown: value = match_key, label = date + teams."""
    if df.empty or "match_key" not in df.columns:
        return []
    d = _sort_log_for_display(df)
    opts = []
    seen: set[str] = set()
    for _, row in d.iterrows():
        mk = _safe_str(row.get("match_key"))
        if not mk or mk in seen:
            continue
        seen.add(mk)
        md = _safe_str(row.get("match_date"))
        t1 = _safe_str(row.get("team1_name"))
        t2 = _safe_str(row.get("team2_name"))
        tail = f"{t1} vs {t2}" if t1 and t2 else mk
        label = f"{md} · {tail}" if md else tail
        opts.append({"value": mk, "label": label})
    return opts


def _slugify_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s


def _log_chart_summary(df: pd.DataFrame) -> dict:
    """Aggregates for history page charts (JSON-safe)."""
    empty = {
        "winner": {"correct": 0, "wrong": 0, "pending": 0},
        "potm": {"correct": 0, "wrong": 0, "pending": 0},
        "timeline": [],
        "calibration": [],
        "picks": {"top_bat_match": 0, "top_bat_total": 0, "top_bowl_match": 0, "top_bowl_total": 0},
    }
    if df.empty:
        return empty

    has_actual = (
        df["actual_winner"].fillna("").astype(str).str.strip().ne("")
        if "actual_winner" in df.columns
        else pd.Series(False, index=df.index)
    )

    for i in df.index:
        ha = bool(has_actual.at[i])
        if not ha:
            empty["winner"]["pending"] += 1
            empty["potm"]["pending"] += 1
            continue
        wc = _bool_cell(df.at[i, "winner_correct"]) if "winner_correct" in df.columns else None
        pc = _bool_cell(df.at[i, "potm_correct"]) if "potm_correct" in df.columns else None
        if wc is True:
            empty["winner"]["correct"] += 1
        elif wc is False:
            empty["winner"]["wrong"] += 1
        else:
            empty["winner"]["pending"] += 1
        if pc is True:
            empty["potm"]["correct"] += 1
        elif pc is False:
            empty["potm"]["wrong"] += 1
        else:
            empty["potm"]["pending"] += 1

    # Top bat/bowl pick match (slug compare when both sides present)
    for i in df.index:
        if not bool(has_actual.at[i]):
            continue
        pr = str(df.at[i, "pred_best_bat"] if "pred_best_bat" in df.columns else "") or ""
        ar = str(df.at[i, "actual_top_scorer"] if "actual_top_scorer" in df.columns else "") or ""
        pr, ar = pr.strip(), ar.strip()
        if pr and ar:
            empty["picks"]["top_bat_total"] += 1
            if _slugify_name(pr) == _slugify_name(ar):
                empty["picks"]["top_bat_match"] += 1
        pb = str(df.at[i, "pred_best_bowl"] if "pred_best_bowl" in df.columns else "") or ""
        ab = str(df.at[i, "actual_top_wicket_bowler"] if "actual_top_wicket_bowler" in df.columns else "") or ""
        pb, ab = pb.strip(), ab.strip()
        if pb and ab:
            empty["picks"]["top_bowl_total"] += 1
            if _slugify_name(pb) == _slugify_name(ab):
                empty["picks"]["top_bowl_match"] += 1

    # Timeline: chronological last 20 matches with a date
    dfc = df.copy()
    dfc["_d"] = pd.to_datetime(dfc["match_date"], errors="coerce")
    dfc = dfc.sort_values("_d").dropna(subset=["_d"]).tail(20)
    for _, row in dfc.iterrows():
        mk = str(row.get("match_key", "") or "")[:22]
        md = row["_d"]
        label = mk or (md.strftime("%Y-%m-%d") if hasattr(md, "strftime") else "")
        try:
            p1 = float(row.get("pred_p_team1_win", 0.5) or 0.5)
        except (TypeError, ValueError):
            p1 = 0.5
        p1 = max(0.0, min(1.0, p1))
        ha = str(row.get("actual_winner", "") or "").strip() != ""
        wc = _bool_cell(row.get("winner_correct")) if ha else None
        outcome = "pending" if wc is None else ("hit" if wc else "miss")
        empty["timeline"].append(
            {
                "label": label,
                "p_team1_pct": round(p1 * 100, 1),
                "outcome": outcome,
            }
        )

    # Calibration bins: P(team1) vs winner graded rows
    sub = df[has_actual].copy()
    if len(sub) and "pred_p_team1_win" in sub.columns:
        bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
        labels = ["0–25%", "25–50%", "50–75%", "75–100%"]
        for (lo, hi), lab in zip(bins, labels):
            mask = pd.Series(True, index=sub.index)
            try:
                p = sub["pred_p_team1_win"].astype(float)
                mask = (p >= lo) & (p < hi)
            except Exception:
                pass
            chunk = sub[mask]
            n = len(chunk)
            corr = 0
            tot = 0
            for j in chunk.index:
                b = _bool_cell(chunk.loc[j].get("winner_correct"))
                if b is not None:
                    tot += 1
                    if b:
                        corr += 1
            empty["calibration"].append(
                {
                    "label": lab,
                    "n": n,
                    "graded": tot,
                    "correct": corr,
                    "rate": round(100 * corr / tot, 1) if tot else None,
                }
            )

    return empty


@app.get("/api/prediction-log")
def api_prediction_log():
    df = read_prediction_log_dataframe()
    if df is None:
        return {"rows": []}
    df = df.fillna("")
    return {"rows": df.to_dict(orient="records")}


@app.get("/api/build-info")
def api_build_info():
    """Which CSV source the app used (GitHub raw vs bundle) — use to verify Vercel sees latest pushes."""
    return prediction_log_meta()


@app.get("/health")
def health():
    return {"status": "ok"}
