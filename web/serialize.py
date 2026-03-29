"""Convert predict_match_outcomes / simulation output to JSON-serializable dicts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _clean_scalar(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (np.floating, np.integer)):
        return float(x) if isinstance(x, np.floating) else int(x)
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    return x


def series_or_row_to_dict(obj: Any) -> dict | None:
    if obj is None:
        return None
    if isinstance(obj, pd.Series):
        d = obj.to_dict()
        return {k: _clean_scalar(v) for k, v in d.items()}
    if isinstance(obj, dict):
        return {k: _clean_scalar(v) for k, v in obj.items()}
    return None


def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    if df is None or len(df) == 0:
        return []
    records = df.replace({np.nan: None}).to_dict(orient="records")
    return [{k: _clean_scalar(v) for k, v in r.items()} for r in records]


def slim_simulation(sim: dict) -> dict:
    """Drop heavy / non-JSON parts of run_simulation output."""
    drop = {"t1_df", "t2_df", "player_confidence"}
    out = {k: v for k, v in sim.items() if k not in drop}
    pm = out.get("pitch_multipliers")
    if pm is not None and hasattr(pm, "__dict__"):
        out["pitch_multipliers"] = {
            "first_innings_runs": getattr(pm, "first_innings_runs", None),
            "second_innings_runs": getattr(pm, "second_innings_runs", None),
            "first_innings_wickets": getattr(pm, "first_innings_wickets", None),
            "second_innings_wickets": getattr(pm, "second_innings_wickets", None),
        }
    for k in list(out.keys()):
        v = out[k]
        if isinstance(v, pd.DataFrame):
            out[k] = dataframe_to_records(v)
        elif isinstance(v, pd.Series):
            out[k] = series_or_row_to_dict(v)
    return out


def make_json_safe(obj: Any) -> Any:
    """Recursively convert numpy types and NaN for JSON."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, str):
        return obj
    if isinstance(obj, int) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, pd.DataFrame):
        return dataframe_to_records(obj)
    if isinstance(obj, pd.Series):
        return series_or_row_to_dict(obj)
    return obj


def six_to_payload(six: dict) -> dict:
    out: dict[str, Any] = {}
    out["winning_team"] = six.get("winning_team")
    out["best_batsman"] = series_or_row_to_dict(six.get("best_batsman"))
    out["best_bowler"] = series_or_row_to_dict(six.get("best_bowler"))
    out["top5_batsmen"] = dataframe_to_records(six.get("top5_batsmen"))
    out["top3_bowlers"] = dataframe_to_records(six.get("top3_bowlers"))
    out["player_of_the_match"] = series_or_row_to_dict(six.get("player_of_the_match"))
    return out
