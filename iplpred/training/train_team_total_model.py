"""
Train Ridge regressors for team innings totals from match-level features (no leakage).

Requires match_training_dataset.csv with team1_total_runs, team2_total_runs from
build_training_dataset.py after full player_features build.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from iplpred.match.match_winner_model import WINNER_FEATURE_COLS, build_winner_feature_matrix
from iplpred.training.train_match_winner_model import recency_sample_weights
from iplpred.paths import MODELS_DIR, PROCESSED_DIR

MATCH_TRAINING_PATH = PROCESSED_DIR / "match_training_dataset.csv"
PLAYER_MATCH_STATS_PATH = PROCESSED_DIR / "player_match_stats.csv"
MODEL_PATH = MODELS_DIR / "team_total_regressor.pkl"


def load_with_dates() -> pd.DataFrame:
    df = pd.read_csv(MATCH_TRAINING_PATH, low_memory=False)
    if not PLAYER_MATCH_STATS_PATH.exists():
        df["date"] = pd.NaT
        return df
    pms = pd.read_csv(PLAYER_MATCH_STATS_PATH, usecols=["match_id", "date"], low_memory=False)
    pms["date"] = pd.to_datetime(pms["date"], errors="coerce")
    md = pms.groupby("match_id", as_index=False)["date"].min()
    return df.merge(md, on="match_id", how="left")


def time_based_split(
    df: pd.DataFrame, test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    df = df[
        df["team1_total_runs"].notna()
        & df["team2_total_runs"].notna()
        & (df["team1_total_runs"] >= 0)
        & (df["team2_total_runs"] >= 0)
    ].copy()
    n = len(df)
    if n < 20:
        raise ValueError(f"Too few rows with team totals: {n}")
    cut = int(np.floor(n * (1.0 - test_frac)))
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()
    if len(test) == 0:
        test = train.iloc[-max(1, n // 5) :].copy()
        train = train.iloc[: -len(test)].copy()
    return train, test


def main() -> None:
    df = load_with_dates()
    train, test = time_based_split(df, test_frac=0.2)

    X_train = build_winner_feature_matrix(train)
    X_test = build_winner_feature_matrix(test)
    y1_train = train["team1_total_runs"].astype(float).values
    y1_test = test["team1_total_runs"].astype(float).values
    y2_train = train["team2_total_runs"].astype(float).values
    y2_test = test["team2_total_runs"].astype(float).values

    sw_train = recency_sample_weights(train["date"])

    r1 = Ridge(alpha=2.0)
    r2 = Ridge(alpha=2.0)
    r1.fit(X_train, y1_train, sample_weight=sw_train)
    r2.fit(X_train, y2_train, sample_weight=sw_train)

    p1 = r1.predict(X_test)
    p2 = r2.predict(X_test)
    mae1 = mean_absolute_error(y1_test, p1)
    mae2 = mean_absolute_error(y2_test, p2)
    r2_1 = r2_score(y1_test, p1)
    r2_2 = r2_score(y2_test, p2)

    print(f"Train rows: {len(train)}  Test rows: {len(test)}")
    print(f"Team1 total — MAE: {mae1:.2f}  R2: {r2_1:.4f}")
    print(f"Team2 total — MAE: {mae2:.2f}  R2: {r2_2:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_team1": r1,
            "model_team2": r2,
            "feature_names": WINNER_FEATURE_COLS,
            "test_mae_team1": mae1,
            "test_mae_team2": mae2,
        },
        MODEL_PATH,
    )
    print(f"Saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
