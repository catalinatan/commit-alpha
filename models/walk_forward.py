"""
Walk-forward validation utilities for time-series panel data.

Why TimeSeriesSplit and not random shuffle?
    A random train/test split on a time series lets training examples from the
    future leak into the fit. The model memorises outcomes it could never have
    known at prediction time, and reported accuracy is artificially inflated —
    a classic look-ahead bias. Walk-forward validation keeps every prediction
    strictly out-of-sample: the model is only ever trained on data that pre-dates
    the week it is predicting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def get_week_splits(
    df: pd.DataFrame,
    date_col: str = "week_start",
    n_splits: int = 5,
    min_train_weeks: int = 10,
) -> list:
    """
    Generate walk-forward train/test splits over unique calendar weeks.

    Splits are performed on the sorted sequence of unique week timestamps so
    that every ticker belonging to a given week is always in the same fold.
    This avoids the subtle data-leakage that would occur if splits were made
    on row indices directly (which could put week W of ticker A in train while
    week W of ticker B is in test).

    Args:
        df:               Signals DataFrame containing the date column.
        date_col:         Column holding the week timestamp.
        n_splits:         Number of walk-forward folds.
        min_train_weeks:  Minimum number of training weeks required per fold;
                          early folds with too little history are skipped.

    Returns:
        List of ``(train_dates, test_dates)`` tuples. Each element is a NumPy
        array of week timestamps that belong to that partition.
    """
    unique_weeks = np.sort(df[date_col].unique())

    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_idx, test_idx in tscv.split(unique_weeks):
        if len(train_idx) < min_train_weeks:
            continue
        splits.append((unique_weeks[train_idx], unique_weeks[test_idx]))

    return splits
