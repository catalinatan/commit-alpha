"""
Binary directional classifier for weekly stock returns.

Target: direction_5d = 1 if fwd_ret_5d > 0, else 0  (up vs flat/down next week)

Two models are evaluated under identical walk-forward conditions:
  - Logistic Regression  — interpretable linear baseline
  - Gradient Boosting    — captures non-linear signal interactions

Features are normalized by a StandardScaler fitted on the *training fold only*.
Fitting the scaler on all data (including the test fold) would be data leakage.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from models.walk_forward import get_week_splits

TARGET_COL = "fwd_ret_5d"

FEATURE_COLS = [
    "commit_count",
    "commit_count_delta",      # week-over-week momentum
    "contributor_count",
    "contributor_count_delta",
    "star_rank",               # cross-sectional rank within week (1 = fewest stars)
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to the signals DataFrame.

    Transformations:
      - ``*_delta``: week-over-week change (lagged diff, looks back only — no leakage).
      - ``star_rank``: within-week ordinal rank across tickers; preserves cross-sectional
        ordering without distortion from the large absolute scale gap between repos.

    The raw ``commit_count`` and ``contributor_count`` columns are retained so the
    StandardScaler in each fold can normalise them on training data only.

    Args:
        df: Signals DataFrame sorted by (ticker, week_start).

    Returns:
        Copy of ``df`` with the additional feature columns appended.
    """
    out = df.copy().sort_values(["ticker", "week_start"])

    for col in ("commit_count", "contributor_count"):
        out[f"{col}_delta"] = out.groupby("ticker")[col].transform(lambda x: x.diff())

    # Rank tickers by star_count within each week (ties broken by mean rank)
    out["star_rank"] = out.groupby("week_start")["star_count"].rank(method="average")

    return out


def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray, fold: int) -> dict:
    """Accuracy + one-sided binomial p-value testing H1: accuracy > 0.5 (random)."""
    n = len(y_true)
    correct = int((y_true == y_pred).sum())
    acc = correct / n
    p_val = stats.binomtest(correct, n, p=0.5, alternative="greater").pvalue
    return {"fold": fold, "n_test": n, "accuracy": round(acc, 4), "p_value": round(p_val, 4)}


def run_walk_forward_classification(
    signals_df: pd.DataFrame,
    n_splits: int = 5,
) -> tuple:
    """
    Run walk-forward logistic regression and gradient boosting classifiers.

    At every fold:
      1. Fit StandardScaler on *training rows only* to avoid data leakage.
      2. Fit both models on scaled training features.
      3. Predict on the held-out test fold.
      4. Record accuracy and collect out-of-sample probability scores.

    Args:
        signals_df: Joined signals DataFrame from Tier 1 (output of main.py Step 4).
        n_splits:   Number of walk-forward folds (default 5).

    Returns:
        Tuple ``(lr_results, gbm_results, oos_preds)``:
          - ``lr_results`` / ``gbm_results``: DataFrames with per-fold accuracy metrics.
          - ``oos_preds``: combined out-of-sample rows with columns
            [ticker, week_start, fwd_ret_5d, y_true, lr_prob, gbm_prob].
    """
    df = engineer_features(signals_df)

    df["direction"] = (df[TARGET_COL] > 0).astype(int)
    df = df.dropna(subset=FEATURE_COLS + ["direction"]).reset_index(drop=True)

    splits = get_week_splits(df, date_col="week_start", n_splits=n_splits)
    if not splits:
        raise RuntimeError(
            "No valid walk-forward folds — dataset may be too small. "
            "Try reducing min_train_weeks in walk_forward.get_week_splits()."
        )

    lr_rows, gbm_rows, oos_parts = [], [], []

    for fold_i, (train_dates, test_dates) in enumerate(splits, start=1):
        train = df[df["week_start"].isin(train_dates)]
        test  = df[df["week_start"].isin(test_dates)]

        X_tr, y_tr = train[FEATURE_COLS].values, train["direction"].values
        X_te, y_te = test[FEATURE_COLS].values,  test["direction"].values

        # Scaler fitted on training fold only — applying it to test fold is valid
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s  = scaler.transform(X_te)

        # ── Logistic Regression ───────────────────────────────────────────────
        lr = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
        lr.fit(X_tr_s, y_tr)
        lr_pred = lr.predict(X_te_s)
        lr_prob = lr.predict_proba(X_te_s)[:, 1]
        lr_rows.append(_fold_metrics(y_te, lr_pred, fold_i))

        # ── Gradient Boosting ─────────────────────────────────────────────────
        # Shallow trees + few estimators to limit overfitting on this small dataset
        gbm = GradientBoostingClassifier(
            n_estimators=50, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        gbm.fit(X_tr, y_tr)  # GBM builds its own internal scaling
        gbm_pred = gbm.predict(X_te)
        gbm_prob = gbm.predict_proba(X_te)[:, 1]
        gbm_rows.append(_fold_metrics(y_te, gbm_pred, fold_i))

        fold_oos = test[["ticker", "week_start", TARGET_COL]].copy()
        fold_oos["y_true"]   = y_te
        fold_oos["lr_prob"]  = lr_prob
        fold_oos["gbm_prob"] = gbm_prob
        oos_parts.append(fold_oos)

    lr_results  = pd.DataFrame(lr_rows)
    gbm_results = pd.DataFrame(gbm_rows)
    oos_preds   = pd.concat(oos_parts, ignore_index=True)

    return lr_results, gbm_results, oos_preds
