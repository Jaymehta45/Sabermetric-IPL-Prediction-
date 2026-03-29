"""
Train RandomForest regressors for player runs and wickets per match.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from iplpred.paths import MODELS_DIR, PROCESSED_DIR

PLAYER_FEATURES_PATH = PROCESSED_DIR / "player_features.csv"
PLAYER_MATCH_STATS_PATH = PROCESSED_DIR / "player_match_stats.csv"
RUNS_MODEL_PATH = MODELS_DIR / "player_runs_model.pkl"
WICKETS_MODEL_PATH = MODELS_DIR / "player_wickets_model.pkl"

FEATURE_COLS = [
    "form_runs",
    "form_wickets",
    "form_runs_ipl",
    "form_wickets_ipl",
    "days_since_prev_match",
    "momentum_runs",
    "momentum_wickets",
    "strike_rate",
    "economy",
    "venue_avg_runs",
    "venue_avg_wickets",
    "batting_matchup_sr",
    "bowling_matchup_sr",
    "opponent_strength",
    "std_runs",
    "std_wickets",
    "role_encoded",
]

OPTIONAL_FROM_CSV = [
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
    "opponent_strength",
]

ROLE_MAP = {"batter": 0, "bowler": 1, "allrounder": 2}

MOMENTUM_WEIGHTS = (0.5, 0.3, 0.2)


def _weighted_lag3(series: pd.Series) -> pd.Series:
    s1, s2, s3 = series.shift(1), series.shift(2), series.shift(3)
    w = MOMENTUM_WEIGHTS
    return w[0] * s1 + w[1] * s2 + w[2] * s3


def fill_new_features_with_player_global(df: pd.DataFrame, cols: list[str]) -> None:
    """Fill NaN with per-player mean, then overall mean."""
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
        gm = df.groupby("player_id")[col].transform("mean")
        df[col] = df[col].fillna(gm)
        df[col] = df[col].fillna(df[col].mean())


def load_and_prepare() -> pd.DataFrame:
    df = pd.read_csv(PLAYER_FEATURES_PATH, low_memory=False)
    if not {"runs", "wickets"}.issubset(df.columns):
        raise ValueError("player_features.csv must include runs and wickets")

    for c in OPTIONAL_FROM_CSV:
        if c not in df.columns:
            df[c] = np.nan

    if PLAYER_MATCH_STATS_PATH.exists():
        meta = pd.read_csv(
            PLAYER_MATCH_STATS_PATH,
            usecols=["match_id", "player_id", "date", "role"],
            low_memory=False,
        )
        df = df.merge(meta, on=["match_id", "player_id"], how="left")
    else:
        df["date"] = pd.NaT
        df["role"] = "allrounder"

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["role"] = df["role"].fillna("allrounder").astype(str).str.strip().str.lower()
    df["role_encoded"] = df["role"].map(lambda r: ROLE_MAP.get(r, 2))

    df = df.sort_values(["player_id", "date", "match_id"])

    df["momentum_runs"] = df.groupby("player_id")["runs"].transform(
        lambda s: _weighted_lag3(s)
    )
    df["momentum_wickets"] = df.groupby("player_id")["wickets"].transform(
        lambda s: _weighted_lag3(s)
    )

    if "std_runs" not in df.columns:
        df["std_runs"] = (
            df.groupby("player_id")["runs"]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).std())
            .fillna(0.0)
        )
    else:
        df["std_runs"] = pd.to_numeric(df["std_runs"], errors="coerce").fillna(0.0)

    if "std_wickets" not in df.columns:
        df["std_wickets"] = (
            df.groupby("player_id")["wickets"]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).std())
            .fillna(0.0)
        )
    else:
        df["std_wickets"] = pd.to_numeric(df["std_wickets"], errors="coerce").fillna(0.0)

    for c in ("form_runs", "form_wickets", "strike_rate", "economy"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ("form_runs_ipl", "form_wickets_ipl", "days_since_prev_match"):
        if c not in df.columns:
            df[c] = np.nan

    new_like = [
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
        "opponent_strength",
        "momentum_runs",
        "momentum_wickets",
        "form_runs_ipl",
        "form_wickets_ipl",
        "days_since_prev_match",
    ]
    fill_new_features_with_player_global(df, new_like)

    df["form_runs_ipl"] = df["form_runs_ipl"].fillna(df["form_runs"])
    df["form_wickets_ipl"] = df["form_wickets_ipl"].fillna(df["form_wickets"])
    dsm = pd.to_numeric(df["days_since_prev_match"], errors="coerce")
    med = float(dsm.median()) if dsm.notna().any() else 30.0
    if np.isnan(med):
        med = 30.0
    df["days_since_prev_match"] = dsm.fillna(med)

    for c in ("form_runs", "form_wickets", "strike_rate", "economy"):
        df[c] = df[c].fillna(df.groupby("player_id")[c].transform("mean"))
        df[c] = df[c].fillna(df[c].mean())

    df = df.dropna(subset=["runs", "wickets"])
    return df


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["date", "match_id", "player_id"]).reset_index(drop=True)
    n = len(df)
    cut = int(np.floor(n * (1.0 - test_frac)))
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()
    return train, test


def print_top_importances(
    label: str, rf: RandomForestRegressor, features: list[str], n: int = 10
) -> None:
    pairs = sorted(zip(features, rf.feature_importances_), key=lambda x: -x[1])[:n]
    print(f"{label} (top {n}):")
    for name, imp in pairs:
        print(f"  {name}: {imp:.4f}")


def main() -> None:
    df = load_and_prepare()

    train_df, test_df = time_based_split(df, test_frac=0.2)

    X_train = train_df[FEATURE_COLS].values
    X_test = test_df[FEATURE_COLS].values
    y_runs_train = train_df["runs"].values
    y_runs_test = test_df["runs"].values
    y_wk_train = train_df["wickets"].values
    y_wk_test = test_df["wickets"].values

    rf_runs = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    rf_runs.fit(X_train, y_runs_train)
    pred_runs = rf_runs.predict(X_test)
    r2_runs = r2_score(y_runs_test, pred_runs)

    rf_wk = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    rf_wk.fit(X_train, y_wk_train)
    pred_wk = rf_wk.predict(X_test)
    r2_wk = r2_score(y_wk_test, pred_wk)

    print(f"R2 score (runs): {r2_runs:.4f}")
    print(f"R2 score (wickets): {r2_wk:.4f}")
    print()
    print_top_importances("Feature importance (runs model)", rf_runs, FEATURE_COLS)
    print()
    print_top_importances("Feature importance (wickets model)", rf_wk, FEATURE_COLS)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_runs, RUNS_MODEL_PATH)
    joblib.dump(rf_wk, WICKETS_MODEL_PATH)

    print("Upgraded player model trained")


if __name__ == "__main__":
    main()
