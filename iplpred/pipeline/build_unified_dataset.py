"""
Build a unified ball-by-ball dataset from CSV and Cricsheet JSON under:

- data/raw/ (recursive)
- matches/ at repo root (recursive) — e.g. one file per match (~80 IPL games)
"""

from __future__ import annotations

import json
import sys
import traceback
import os
from pathlib import Path

import pandas as pd

from iplpred.paths import DATA_DIR, MATCHES_DIR, PROCESSED_DIR

RAW_DIR = DATA_DIR / "raw"
OUTPUT_PATH = PROCESSED_DIR / "unified_ball_by_ball.csv"

FINAL_COLUMNS = [
    "match_id",
    "date",
    "venue",
    "batting_team",
    "bowling_team",
    "batter",
    "bowler",
    "non_striker",
    "runs_off_bat",
    "extras",
    "is_wide",
    "is_noball",
    "dismissal_kind",
    "player_dismissed",
]


def _strip_str(x) -> str:
    if pd.isna(x) or x is None:
        return ""
    return str(x).strip()


def _safe_float(v, default=0.0) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int01(v) -> int:
    n = _safe_float(v, 0.0)
    return 1 if n > 0 else 0


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known aliases to schema names; compute derived columns."""
    df = df.copy()
    # Feed-style exports (e.g. manual IPL match CSVs)
    if "batter_id" in df.columns and "batter" not in df.columns:
        df = df.rename(columns={"batter_id": "batter"})
    if "bowler_id" in df.columns and "bowler" not in df.columns:
        df = df.rename(columns={"bowler_id": "bowler"})
    if "non_striker_id" in df.columns and "non_striker" not in df.columns:
        df = df.rename(columns={"non_striker_id": "non_striker"})
    if "batter_out_id" in df.columns and "player_dismissed" not in df.columns:
        df = df.rename(columns={"batter_out_id": "player_dismissed"})

    colmap = {}
    lower = {c.lower(): c for c in df.columns}

    def pick(*names: str) -> str | None:
        for n in names:
            if n in df.columns:
                return n
            if n.lower() in lower:
                return lower[n.lower()]
        return None

    if pick("match_id", "matchId") is not None:
        colmap[pick("match_id", "matchId")] = "match_id"
    if pick("start_date", "date") is not None:
        colmap[pick("start_date", "date")] = "date"
    if pick("striker") is not None and pick("batter", "batsman") is None:
        colmap[pick("striker")] = "batter"
    if pick("batsman") is not None and pick("batter") is None:
        colmap[pick("batsman")] = "batter"
    if pick("wicket_type") is not None and pick("dismissal_kind") is None:
        colmap[pick("wicket_type")] = "dismissal_kind"

    df = df.rename(columns={k: v for k, v in colmap.items() if k and v})

    if "runs_batter" in df.columns and "runs_off_bat" not in df.columns:
        df["runs_off_bat"] = pd.to_numeric(df["runs_batter"], errors="coerce").fillna(0.0)
    if "runs_extras" in df.columns and "extras" not in df.columns:
        df["extras"] = pd.to_numeric(df["runs_extras"], errors="coerce").fillna(0.0)

    # runs_off_bat
    if "runs_off_bat" not in df.columns and "batsman_runs" in df.columns:
        df["runs_off_bat"] = df["batsman_runs"]
    if "runs_off_bat" not in df.columns and "batter" in df.columns:
        pass  # may be filled below

    # total_runs vs extras + runs_off_bat
    if "total_runs" in df.columns:
        if "extras" not in df.columns:
            df["extras"] = 0.0
        if "runs_off_bat" not in df.columns:
            df["runs_off_bat"] = df["total_runs"] - df["extras"].fillna(0)

    # extras from components
    comp_cols = ["wides", "noballs", "byes", "legbyes", "penalty"]
    present = [c for c in comp_cols if c in df.columns]
    if present:
        ex = sum(df[c].fillna(0).astype(float) for c in present)
        if "extras" not in df.columns or df["extras"].isna().all():
            df["extras"] = ex
        else:
            df["extras"] = df["extras"].fillna(ex)

    if "extras" not in df.columns:
        df["extras"] = 0.0
    df["extras"] = pd.to_numeric(df["extras"], errors="coerce").fillna(0.0)

    # isWide / isNoBall
    if "extras_type" in df.columns:
        et = df["extras_type"].fillna("").astype(str).str.lower().str.strip()
        df["is_wide"] = et.eq("wide").astype(int)
        df["is_noball"] = et.eq("noball").astype(int)
    elif "is_wide" not in df.columns and "isWide" in df.columns:
        df["is_wide"] = df["isWide"].map(_safe_int01)
    elif "is_wide" not in df.columns and "wides" in df.columns:
        df["is_wide"] = df["wides"].map(_safe_int01)
    if "extras_type" not in df.columns:
        if "is_noball" not in df.columns and "isNoBall" in df.columns:
            df["is_noball"] = df["isNoBall"].map(_safe_int01)
        elif "is_noball" not in df.columns and "noballs" in df.columns:
            df["is_noball"] = df["noballs"].map(_safe_int01)

    if "is_wide" not in df.columns:
        df["is_wide"] = 0
    if "is_noball" not in df.columns:
        df["is_noball"] = 0

    df["is_wide"] = df["is_wide"].map(_safe_int01)
    df["is_noball"] = df["is_noball"].map(_safe_int01)

    if "runs_off_bat" not in df.columns:
        df["runs_off_bat"] = 0.0
    df["runs_off_bat"] = pd.to_numeric(df["runs_off_bat"], errors="coerce").fillna(0.0)

    if "innings" in df.columns:
        inn = pd.to_numeric(df["innings"], errors="coerce")
        if "batting_team" not in df.columns:
            df["batting_team"] = ""
        if "bowling_team" not in df.columns:
            df["bowling_team"] = ""
        df["batting_team"] = df["batting_team"].astype(object)
        df["bowling_team"] = df["bowling_team"].astype(object)
        if df["batting_team"].fillna("").astype(str).str.strip().eq("").all():
            df.loc[inn == 1, "batting_team"] = "Sunrisers Hyderabad"
            df.loc[inn == 2, "batting_team"] = "Royal Challengers Bengaluru"
        if df["bowling_team"].fillna("").astype(str).str.strip().eq("").all():
            df.loc[inn == 1, "bowling_team"] = "Royal Challengers Bengaluru"
            df.loc[inn == 2, "bowling_team"] = "Sunrisers Hyderabad"

    if "venue" not in df.columns:
        df["venue"] = ""
    if df["venue"].fillna("").astype(str).str.strip().eq("").all():
        df["venue"] = "M Chinnaswamy Stadium, Bengaluru"
    if "date" not in df.columns:
        df["date"] = ""
    if df["date"].fillna("").astype(str).str.strip().eq("").all():
        df["date"] = "2026-03-28"

    for c in ("batter", "bowler", "non_striker", "venue", "batting_team", "bowling_team", "dismissal_kind", "player_dismissed", "date"):
        if c in df.columns:
            df[c] = df[c].apply(lambda x: _strip_str(x) if pd.notna(x) else "")

    return df


def csv_looks_like_ball_by_ball(df: pd.DataFrame) -> bool:
    cols = {c.lower() for c in df.columns}
    has_batter = bool({"batter", "batsman", "striker", "batter_id"} & cols)
    has_bowler = "bowler" in cols or "bowler_id" in cols
    return has_batter and has_bowler


def parse_csv_file(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return None

    if not csv_looks_like_ball_by_ball(df):
        return None

    df = standardize_dataframe(df)

    missing = [c for c in ("match_id", "batter", "bowler") if c not in df.columns]
    if missing:
        return None

    out = pd.DataFrame()
    out["match_id"] = df["match_id"]
    if "date" in df.columns:
        out["date"] = df["date"].astype(str)
    else:
        out["date"] = ""
    out["venue"] = df["venue"] if "venue" in df.columns else ""
    out["batting_team"] = df["batting_team"] if "batting_team" in df.columns else ""
    out["bowling_team"] = df["bowling_team"] if "bowling_team" in df.columns else ""
    out["batter"] = df["batter"]
    out["bowler"] = df["bowler"]
    out["non_striker"] = df["non_striker"] if "non_striker" in df.columns else ""
    out["runs_off_bat"] = df["runs_off_bat"]
    out["extras"] = df["extras"]
    out["is_wide"] = df["is_wide"]
    out["is_noball"] = df["is_noball"]
    out["dismissal_kind"] = df["dismissal_kind"] if "dismissal_kind" in df.columns else ""
    out["player_dismissed"] = df["player_dismissed"] if "player_dismissed" in df.columns else ""

    out = out[out["batter"].astype(str).str.len() > 0]
    out = out[out["bowler"].astype(str).str.len() > 0]

    out["extras"] = pd.to_numeric(out["extras"], errors="coerce").fillna(0.0)
    out["runs_off_bat"] = pd.to_numeric(out["runs_off_bat"], errors="coerce").fillna(0.0)
    out["is_wide"] = out["is_wide"].astype(int)
    out["is_noball"] = out["is_noball"].astype(int)

    return out[FINAL_COLUMNS]


def _json_match_date(info: dict) -> pd.Timestamp:
    dates = info.get("dates") or []
    if isinstance(dates, str):
        dates = [dates]
    parsed = []
    for d in dates:
        t = pd.to_datetime(str(d), errors="coerce")
        if pd.notna(t):
            parsed.append(t)
    if not parsed:
        return pd.NaT
    return pd.Timestamp(min(parsed))


def _bowling_team(info: dict, batting_team: str) -> str:
    teams = info.get("teams") or []
    if len(teams) == 2:
        for t in teams:
            if t != batting_team:
                return t
    return ""


def parse_json_cricsheet(path: Path) -> pd.DataFrame | None:
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:
        return None

    info = doc.get("info") or {}
    innings_list = doc.get("innings") or []
    if not innings_list:
        return None

    match_id = path.stem
    match_date = _json_match_date(info)
    venue = _strip_str(info.get("venue", ""))

    rows: list[dict] = []

    for inning in innings_list:
        batting_team = inning.get("team") or ""
        bowling_team = _bowling_team(info, batting_team)
        overs_list = inning.get("overs") or []
        for over in overs_list:
            deliveries = over.get("deliveries") or []
            for delivery in deliveries:
                batter = delivery.get("batter") or delivery.get("batsman") or ""
                bowler = delivery.get("bowler") or ""
                non_striker = delivery.get("non_striker") or ""
                batter = _strip_str(batter)
                bowler = _strip_str(bowler)
                non_striker = _strip_str(non_striker)

                runs = delivery.get("runs") or {}
                runs_off_bat = _safe_float(runs.get("batter", 0))

                ex_detail = delivery.get("extras") or {}
                if isinstance(ex_detail, dict):
                    wides = _safe_float(ex_detail.get("wides", 0))
                    noballs = _safe_float(ex_detail.get("noballs", 0))
                    byes = _safe_float(ex_detail.get("byes", 0))
                    legbyes = _safe_float(ex_detail.get("legbyes", 0))
                    penalty = _safe_float(ex_detail.get("penalty", 0))
                    extras_total = wides + noballs + byes + legbyes + penalty
                else:
                    extras_total = _safe_float(runs.get("extras", 0))
                    wides = noballs = 0.0

                if extras_total == 0.0 and "extras" in runs:
                    extras_total = _safe_float(runs.get("extras", 0))

                is_wide = 1 if (isinstance(ex_detail, dict) and _safe_float(ex_detail.get("wides", 0)) > 0) else 0
                is_noball = 1 if (isinstance(ex_detail, dict) and _safe_float(ex_detail.get("noballs", 0)) > 0) else 0

                dismissal_kind = ""
                player_dismissed = ""
                wk = delivery.get("wickets")
                if wk and isinstance(wk, list) and len(wk) > 0:
                    w0 = wk[0]
                    if isinstance(w0, dict):
                        dismissal_kind = _strip_str(w0.get("kind", ""))
                        player_dismissed = _strip_str(w0.get("player_out", ""))

                rows.append(
                    {
                        "match_id": match_id,
                        "date": match_date,
                        "venue": venue,
                        "batting_team": batting_team,
                        "bowling_team": bowling_team,
                        "batter": batter,
                        "bowler": bowler,
                        "non_striker": non_striker,
                        "runs_off_bat": runs_off_bat,
                        "extras": extras_total,
                        "is_wide": is_wide,
                        "is_noball": is_noball,
                        "dismissal_kind": dismissal_kind,
                        "player_dismissed": player_dismissed,
                    }
                )

    if not rows:
        return None

    out = pd.DataFrame(rows)
    out = out[out["batter"].str.len() > 0]
    out = out[out["bowler"].str.len() > 0]
    if out.empty:
        return None

    out["extras"] = pd.to_numeric(out["extras"], errors="coerce").fillna(0.0)
    out["runs_off_bat"] = pd.to_numeric(out["runs_off_bat"], errors="coerce").fillna(0.0)
    out["is_wide"] = out["is_wide"].astype(int)
    out["is_noball"] = out["is_noball"].astype(int)

    return out[FINAL_COLUMNS]


def _walk_inputs(base: Path) -> list[Path]:
    if not base.is_dir():
        return []
    out: list[Path] = []
    for root, _dirs, files in os.walk(base, followlinks=True):
        for name in files:
            if name.lower().endswith((".csv", ".json")):
                out.append(Path(root) / name)
    return out


def collect_raw_paths() -> list[Path]:
    """All CSV/JSON ball-by-ball sources from data/raw and matches/."""
    paths: list[Path] = []
    paths.extend(_walk_inputs(RAW_DIR))
    paths.extend(_walk_inputs(MATCHES_DIR))
    return sorted(paths)


def normalize_match_ids_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """
    Numeric match_id values are kept. String keys (e.g. match1-rcb-vs-srh) map to
    stable ints starting at 910_000_000 to avoid clashing with Cricsheet/ICC ids.
    """
    out = df.copy()
    seen: dict[str, int] = {}
    next_id = 910_000_000

    def map_one(x: object) -> int:
        nonlocal next_id
        if pd.isna(x):
            return -1
        s = str(x).strip()
        if s.isdigit():
            return int(s)
        if s not in seen:
            seen[s] = next_id
            next_id += 1
        return seen[s]

    out["match_id"] = out["match_id"].map(map_one)
    return out


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for path in collect_raw_paths():
        try:
            if path.suffix.lower() == ".csv":
                df = parse_csv_file(path)
            else:
                df = parse_json_cricsheet(path)
            if df is not None and not df.empty:
                frames.append(df)
        except Exception:
            errors.append(f"{path}: {traceback.format_exc()}")

    if not frames:
        print("No ball-by-ball rows produced.", file=sys.stderr)
        if errors:
            print("Errors:", *errors, sep="\n", file=sys.stderr)
        sys.exit(1)

    unified = pd.concat(frames, ignore_index=True)
    unified = normalize_match_ids_to_int(unified)

    for c in FINAL_COLUMNS:
        if c not in unified.columns:
            unified[c] = pd.NA

    unified = unified[FINAL_COLUMNS]

    unified["date"] = pd.to_datetime(unified["date"], errors="coerce")

    unified = unified.drop_duplicates(
        subset=[
            "match_id",
            "batter",
            "bowler",
            "non_striker",
            "runs_off_bat",
            "extras",
        ]
    )
    print("Rows after deduplication:", len(unified))

    unified.to_csv(OUTPUT_PATH, index=False)

    n_rows = len(unified)
    players = pd.unique(
        pd.concat(
            [
                unified["batter"].astype(str),
                unified["bowler"].astype(str),
                unified["non_striker"].astype(str),
                unified["player_dismissed"].astype(str),
            ],
            ignore_index=True,
        )
    )
    n_players = len([p for p in players if p and p != "nan"])
    n_matches = unified["match_id"].nunique()

    print(f"total rows: {n_rows}")
    print(f"number of unique players: {n_players}")
    print(f"number of matches: {n_matches}")
    print("Unified dataset created successfully")


if __name__ == "__main__":
    main()
