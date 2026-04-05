#!/usr/bin/env python3
"""
Append or update rows in data/processed/prediction_log.csv for pre-match and post-match records.

  Pre-match (one row per match_key):
    python3 scripts/log_prediction.py pre examples/prediction_log_pre.example.json

  Post-match (updates same match_key):
    python3 scripts/log_prediction.py post examples/prediction_log_post.example.json

JSON fields — pre: match_key, match_date, venue, team1_name, team2_name, team1_bats_first,
  pred_winner, pred_p_team1_win, pred_best_bat, pred_best_bowl,
  pred_top5_bats (list), pred_top3_bowls (list), pred_potm, pred_extra (optional dict).

Post: match_key, actual_winner, actual_top_scorer, actual_top_scorer_runs (optional),
  actual_top_wicket_bowler, actual_top_wickets (optional), official_potm,
  bbb_match_file, post_notes (optional),
  actual_first_innings_team_runs, actual_second_innings_team_runs (optional — if omitted and
  bbb_match_file points at a CSV, totals are filled from ball-by-ball).

Process 3 (after post-match review — updates same match_key):
    python3 scripts/log_prediction.py process3 examples/prediction_log_process3.example.json

  Fields: match_key, process3_notes (optional). Sets process3_completed_at (UTC).
  The web app reads these columns to show P1 → P2 → P3 completion per match.

  After each process, push so Vercel sees the log (GitHub main):
    python3 scripts/log_prediction.py pre  <json> --push
    python3 scripts/log_prediction.py post <json> --push
    python3 scripts/log_prediction.py process3 <json> --push
  Or export IPLPRED_LOG_AUTO_PUSH=1 once to always push after logging.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iplpred.core.bbb_innings_totals import innings_runs_and_wickets_from_bbb
from iplpred.paths import PROCESSED_DIR

LOG_PATH = PROCESSED_DIR / "prediction_log.csv"


def _vercel_sync_hint() -> None:
    print(
        "→ Vercel: add --push to commit + push (syncs main), or: bash scripts/push_prediction_log.sh"
    )


def _git_push_prediction_log(action: str, match_key: str) -> None:
    """Commit prediction_log.csv, push current branch, then sync origin/main for Vercel raw CSV."""
    rel = LOG_PATH.relative_to(REPO_ROOT)
    try:
        subprocess.run(
            ["git", "add", str(rel)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        st = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=REPO_ROOT,
        )
        if st.returncode == 0:
            print("Git: nothing new to commit for prediction_log.csv")
            return
        mk = (match_key or "").strip()
        msg = f"chore(prediction_log): {action}" + (f" {mk}" if mk else "")
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        br = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(
            ["git", "push", "origin", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
        )
        print(f"Git: pushed branch {br!r}")
        if br != "main":
            r = subprocess.run(
                ["git", "push", "origin", f"{br}:main"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            if r.returncode == 0:
                print("Git: updated origin/main (Vercel prediction_log.csv)")
            else:
                print(
                    "Git: could not push to main (merge or push manually). stderr:",
                    (r.stderr or "")[:500],
                )
        else:
            print("Git: on main — origin/main updated for Vercel")
    except subprocess.CalledProcessError as e:
        print(f"Git push failed: {e}", file=sys.stderr)
        if e.stderr:
            print(e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr, file=sys.stderr)
        raise SystemExit(1) from e


def _maybe_push(args: argparse.Namespace, action: str, match_key: str) -> None:
    want = getattr(args, "push", False) or os.environ.get(
        "IPLPRED_LOG_AUTO_PUSH", ""
    ).strip().lower() in ("1", "true", "yes")
    if want:
        _git_push_prediction_log(action, match_key)

REQUIRED_PRE = [
    "match_key",
    "match_date",
    "venue",
    "team1_name",
    "team2_name",
    "team1_bats_first",
    "pred_winner",
    "pred_p_team1_win",
    "pred_best_bat",
    "pred_best_bowl",
    "pred_top5_bats",
    "pred_top3_bowls",
    "pred_potm",
]

POST_KEYS = [
    "actual_winner",
    "actual_top_scorer",
    "actual_top_scorer_runs",
    "actual_top_wicket_bowler",
    "actual_top_wickets",
    "official_potm",
    "bbb_match_file",
    "post_notes",
    "actual_first_innings_team_runs",
    "actual_second_innings_team_runs",
]

POST_FLOAT_KEYS = frozenset(
    {
        "actual_top_scorer_runs",
        "actual_top_wickets",
        "actual_first_innings_team_runs",
        "actual_second_innings_team_runs",
    }
)


def _load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("JSON root must be an object")
    return data


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    base = [
        "match_key",
        "match_date",
        "venue",
        "team1_name",
        "team2_name",
        "team1_bats_first",
        "pred_winner",
        "pred_p_team1_win",
        "pred_best_bat",
        "pred_best_bowl",
        "pred_top5_bats",
        "pred_top3_bowls",
        "pred_potm",
        "pred_extra_json",
        "actual_winner",
        "actual_top_scorer",
        "actual_top_scorer_runs",
        "actual_top_wicket_bowler",
        "actual_top_wickets",
        "official_potm",
        "bbb_match_file",
        "post_notes",
        "actual_first_innings_team_runs",
        "actual_second_innings_team_runs",
        "winner_correct",
        "potm_correct",
        "logged_at_pre",
        "logged_at_post",
        "process3_completed_at",
        "process3_notes",
    ]
    for c in base:
        if c not in df.columns:
            df[c] = pd.NA
    return df[base]


def cmd_pre(args: argparse.Namespace) -> None:
    data = _load_json(Path(args.json))
    for k in REQUIRED_PRE:
        if k not in data:
            raise SystemExit(f"Missing required field: {k}")

    pred_extra = data.get("pred_extra")
    pred_extra_json = json.dumps(pred_extra) if pred_extra is not None else ""

    top5 = data["pred_top5_bats"]
    top3 = data["pred_top3_bowls"]
    if not isinstance(top5, list) or len(top5) != 5:
        raise SystemExit("pred_top5_bats must be a list of 5 player ids")
    if not isinstance(top3, list) or len(top3) != 3:
        raise SystemExit("pred_top3_bowls must be a list of 3 player ids")

    row = {
        "match_key": str(data["match_key"]).strip(),
        "match_date": str(data["match_date"]).strip(),
        "venue": str(data["venue"]).strip(),
        "team1_name": str(data["team1_name"]).strip(),
        "team2_name": str(data["team2_name"]).strip(),
        "team1_bats_first": bool(data["team1_bats_first"]),
        "pred_winner": str(data["pred_winner"]).strip(),
        "pred_p_team1_win": float(data["pred_p_team1_win"]),
        "pred_best_bat": str(data["pred_best_bat"]).strip(),
        "pred_best_bowl": str(data["pred_best_bowl"]).strip(),
        "pred_top5_bats": json.dumps(top5),
        "pred_top3_bowls": json.dumps(top3),
        "pred_potm": str(data["pred_potm"]).strip(),
        "pred_extra_json": pred_extra_json,
        "actual_winner": "",
        "actual_top_scorer": "",
        "actual_top_scorer_runs": pd.NA,
        "actual_top_wicket_bowler": "",
        "actual_top_wickets": pd.NA,
        "official_potm": "",
        "bbb_match_file": "",
        "post_notes": "",
        "actual_first_innings_team_runs": pd.NA,
        "actual_second_innings_team_runs": pd.NA,
        "winner_correct": "",
        "potm_correct": "",
        "logged_at_pre": datetime.now(timezone.utc).isoformat(),
        "logged_at_post": "",
        "process3_completed_at": "",
        "process3_notes": "",
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.is_file():
        df = pd.read_csv(LOG_PATH, low_memory=False)
        df = _ensure_columns(df)
        if row["match_key"] in df["match_key"].astype(str).values and not args.force:
            raise SystemExit(
                f"match_key {row['match_key']!r} already exists; use --force to replace that row"
            )
        if args.force and row["match_key"] in df["match_key"].astype(str).values:
            df = df[df["match_key"].astype(str) != row["match_key"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df = _ensure_columns(df)
    df.to_csv(LOG_PATH, index=False)
    print(f"Wrote pre-match row -> {LOG_PATH} (match_key={row['match_key']})")
    _vercel_sync_hint()
    _maybe_push(args, "pre", row["match_key"])


def _normalize(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def _df_for_update() -> pd.DataFrame:
    df = pd.read_csv(LOG_PATH, low_memory=False)
    df = _ensure_columns(df)
    for c in df.columns:
        df[c] = df[c].astype(object)
    return df


def cmd_post(args: argparse.Namespace) -> None:
    data = _load_json(Path(args.json))
    if "match_key" not in data:
        raise SystemExit("post JSON must include match_key")
    mk = str(data["match_key"]).strip()
    if not LOG_PATH.is_file():
        raise SystemExit(f"No log file at {LOG_PATH}; run pre first")

    df = _df_for_update()
    idx = df.index[df["match_key"].astype(str) == mk]
    if len(idx) != 1:
        raise SystemExit(f"Expected exactly one row for match_key={mk!r}, found {len(idx)}")

    i = idx[0]
    for k in POST_KEYS:
        if k in data:
            if k in POST_FLOAT_KEYS:
                v = data[k]
                df.iat[i, df.columns.get_loc(k)] = (
                    "" if v is None or v == "" else float(v)
                )
            else:
                df.iat[i, df.columns.get_loc(k)] = (
                    "" if data[k] is None else str(data[k])
                )

    # Fill innings totals from BBB when not supplied (Process 2)
    loc = df.columns.get_loc
    fr = df.iat[i, loc("actual_first_innings_team_runs")]
    sr = df.iat[i, loc("actual_second_innings_team_runs")]
    need_fr = fr is None or (isinstance(fr, float) and pd.isna(fr)) or str(fr).strip() == ""
    need_sr = sr is None or (isinstance(sr, float) and pd.isna(sr)) or str(sr).strip() == ""
    bbb_rel = str(df.iat[i, loc("bbb_match_file")]).strip()
    if (need_fr or need_sr) and bbb_rel:
        bbb_path = REPO_ROOT / bbb_rel.lstrip("/")
        r1, r2, _, _ = innings_runs_and_wickets_from_bbb(bbb_path)
        if need_fr and r1 is not None:
            df.iat[i, loc("actual_first_innings_team_runs")] = float(r1)
        if need_sr and r2 is not None:
            df.iat[i, loc("actual_second_innings_team_runs")] = float(r2)

    df.iat[i, df.columns.get_loc("logged_at_post")] = datetime.now(
        timezone.utc
    ).isoformat()

    pred_w = _normalize(str(df.iat[i, df.columns.get_loc("pred_winner")]))
    act_w = _normalize(str(df.iat[i, df.columns.get_loc("actual_winner")]))
    if act_w:
        df.iat[i, df.columns.get_loc("winner_correct")] = (
            pred_w == act_w if pred_w else ""
        )

    pp = _normalize(str(df.iat[i, df.columns.get_loc("pred_potm")]))
    op = _normalize(str(df.iat[i, df.columns.get_loc("official_potm")]))
    if op:
        df.iat[i, df.columns.get_loc("potm_correct")] = pp == op if pp else ""

    df.to_csv(LOG_PATH, index=False)
    print(f"Updated post-match fields -> {LOG_PATH} (match_key={mk})")
    _vercel_sync_hint()
    _maybe_push(args, "post", mk)


def cmd_process3(args: argparse.Namespace) -> None:
    data = _load_json(Path(args.json))
    if "match_key" not in data:
        raise SystemExit("process3 JSON must include match_key")
    mk = str(data["match_key"]).strip()
    if not LOG_PATH.is_file():
        raise SystemExit(f"No log file at {LOG_PATH}; run pre first")

    df = _df_for_update()
    idx = df.index[df["match_key"].astype(str) == mk]
    if len(idx) != 1:
        raise SystemExit(f"Expected exactly one row for match_key={mk!r}, found {len(idx)}")

    i = idx[0]
    if "process3_notes" in data:
        v = data["process3_notes"]
        df.iat[i, df.columns.get_loc("process3_notes")] = (
            "" if v is None else str(v).strip()
        )
    df.iat[i, df.columns.get_loc("process3_completed_at")] = datetime.now(
        timezone.utc
    ).isoformat()

    df.to_csv(LOG_PATH, index=False)
    print(f"Marked Process 3 complete -> {LOG_PATH} (match_key={mk})")
    _vercel_sync_hint()
    _maybe_push(args, "process3", mk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Log pre/post match predictions to prediction_log.csv")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pre = sub.add_parser("pre", help="Append pre-match prediction row")
    p_pre.add_argument("json", type=str, help="Path to JSON with required pre fields")
    p_pre.add_argument(
        "--force",
        action="store_true",
        help="Replace existing row with same match_key",
    )
    p_pre.add_argument(
        "--push",
        action="store_true",
        help="Git commit + push prediction_log.csv; sync origin/main for Vercel",
    )

    p_post = sub.add_parser("post", help="Fill actuals for existing match_key")
    p_post.add_argument("json", type=str, help="Path to JSON with match_key + actual fields")
    p_post.add_argument(
        "--push",
        action="store_true",
        help="Git commit + push prediction_log.csv; sync origin/main for Vercel",
    )

    p_p3 = sub.add_parser(
        "process3",
        help="Mark Process 3 (post-match model review) complete for match_key",
    )
    p_p3.add_argument("json", type=str, help="Path to JSON with match_key, optional process3_notes")
    p_p3.add_argument(
        "--push",
        action="store_true",
        help="Git commit + push prediction_log.csv; sync origin/main for Vercel",
    )

    args = parser.parse_args()
    if args.cmd == "pre":
        cmd_pre(args)
    elif args.cmd == "post":
        cmd_post(args)
    else:
        cmd_process3(args)


if __name__ == "__main__":
    main()
