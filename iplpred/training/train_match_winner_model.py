"""
Train a team-level classifier: P(team1 wins) from pre-match aggregate team features.

Labels come from match_training_dataset.csv (winner column): **official** IPL results
when ``data/ipl/matches_updated_mens_ipl.csv`` matches ``match_id``, else **proxy**
from higher total team runs (see ``resolve_winner_label``). Ties are excluded from
training rows. Sample weights down-weight proxy vs official labels.

Usage:
  python -m iplpred.training.train_match_winner_model [--official-only] [--model rf|logistic|gbm]
"""

from __future__ import annotations

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from iplpred.match.match_winner_model import (
    MODELS_DIR,
    WINNER_FEATURE_COLS,
    WINNER_MODEL_PATH,
    build_winner_feature_matrix,
)
from iplpred.paths import PROCESSED_DIR

MATCH_TRAINING_PATH = PROCESSED_DIR / "match_training_dataset.csv"
PLAYER_MATCH_STATS_PATH = PROCESSED_DIR / "player_match_stats.csv"


def load_match_training_with_dates() -> pd.DataFrame:
    df = pd.read_csv(MATCH_TRAINING_PATH, low_memory=False)
    if not PLAYER_MATCH_STATS_PATH.exists():
        df["date"] = pd.NaT
        return df
    pms = pd.read_csv(PLAYER_MATCH_STATS_PATH, usecols=["match_id", "date"], low_memory=False)
    pms["date"] = pd.to_datetime(pms["date"], errors="coerce")
    md = pms.groupby("match_id", as_index=False)["date"].min()
    return df.merge(md, on="match_id", how="left")


def recency_sample_weights(dates: pd.Series) -> np.ndarray:
    """Higher weight for more recent matches (exponential decay by age in days)."""
    d = pd.to_datetime(dates, errors="coerce")
    ref = d.max()
    if pd.isna(ref):
        return np.ones(len(dates), dtype=np.float64)
    age = (ref - d).dt.days
    age = pd.to_numeric(age, errors="coerce").fillna(0.0).clip(lower=0.0)
    return np.exp(-age.astype(np.float64) / 500.0)


def label_weight_from_source(df: pd.DataFrame) -> np.ndarray:
    """
    Down-weight proxy winner labels (runs-based) vs official match results.
    """
    if "winner_source" not in df.columns:
        return np.ones(len(df), dtype=np.float64)
    src = df["winner_source"].fillna("").astype(str).str.strip().str.lower()
    return np.where(src == "official", 1.0, np.where(src == "proxy", 0.55, 0.6))


def time_based_split(
    df: pd.DataFrame, test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    df = df[df["winner"].isin(["team1", "team2"])].copy()
    n = len(df)
    if n < 20:
        raise ValueError(f"Too few labeled matches: {n}")
    cut = int(np.floor(n * (1.0 - test_frac)))
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()
    if len(test) == 0:
        test = train.iloc[-max(1, n // 5) :].copy()
        train = train.iloc[: -len(test)].copy()
    return train, test


def _make_base_estimator(model_name: str):
    m = model_name.lower().strip()
    if m == "logistic":
        return LogisticRegression(
            max_iter=2000,
            C=0.15,
            random_state=42,
            class_weight="balanced",
        )
    if m == "gbm":
        return HistGradientBoostingClassifier(
            max_depth=5,
            max_iter=180,
            learning_rate=0.06,
            min_samples_leaf=8,
            l2_regularization=0.2,
            random_state=42,
        )
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train match winner classifier.")
    ap.add_argument(
        "--official-only",
        action="store_true",
        help="Train only on rows with official IPL winner labels.",
    )
    ap.add_argument(
        "--model",
        choices=("rf", "logistic", "gbm"),
        default="rf",
        help="Base estimator before isotonic calibration (default: rf).",
    )
    args = ap.parse_args()

    df = load_match_training_with_dates()
    if args.official_only and "winner_source" in df.columns:
        df = df[df["winner_source"].astype(str).str.strip().str.lower() == "official"].copy()
        if len(df) < 30:
            raise SystemExit(
                f"--official-only left only {len(df)} rows; need at least 30 for a stable split."
            )

    train, test = time_based_split(df, test_frac=0.2)

    y_train = (train["winner"] == "team1").astype(int).values
    y_test = (test["winner"] == "team1").astype(int).values

    X_train = build_winner_feature_matrix(train)
    X_test = build_winner_feature_matrix(test)

    sw_train = recency_sample_weights(train["date"]) * label_weight_from_source(train)
    if args.official_only:
        sw_train = recency_sample_weights(train["date"])

    base = _make_base_estimator(args.model)
    cv = min(5, max(2, len(y_train) // 150))
    clf = CalibratedClassifierCV(base, method="isotonic", cv=cv)
    clf.fit(X_train, y_train, sample_weight=sw_train)

    proba_test = clf.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred_test)
    try:
        auc = roc_auc_score(y_test, proba_test)
    except ValueError:
        auc = float("nan")
    try:
        ll = log_loss(y_test, proba_test)
    except ValueError:
        ll = float("nan")
    brier = brier_score_loss(y_test, proba_test)

    print(f"Train rows: {len(train)}  Test rows: {len(test)}")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test ROC-AUC:  {auc:.4f}")
    print(f"Test log loss: {ll:.4f}")
    print(f"Test Brier:    {brier:.4f}")
    print()
    print("Feature importances (mean decrease impurity, base RF in calibration):")
    subs = getattr(clf, "calibrated_classifiers_", None)
    if subs and len(subs) and hasattr(subs[0].estimator, "feature_importances_"):
        imps = subs[0].estimator.feature_importances_
    else:
        imps = np.zeros(len(WINNER_FEATURE_COLS))
    for name, imp in sorted(zip(WINNER_FEATURE_COLS, imps), key=lambda x: -x[1])[:12]:
        print(f"  {name}: {imp:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": clf,
        "calibration": "isotonic",
        "feature_names": WINNER_FEATURE_COLS,
        "base_model_type": args.model,
        "official_only": bool(args.official_only),
        "train_rows": len(train),
        "test_rows": len(test),
        "test_accuracy": acc,
        "test_roc_auc": auc,
        "test_log_loss": ll,
        "test_brier": brier,
    }
    joblib.dump(bundle, WINNER_MODEL_PATH)
    print()
    print(f"Saved: {WINNER_MODEL_PATH}")


if __name__ == "__main__":
    main()
