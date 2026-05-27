"""
Computes forward log returns at multiple horizons from daily OHLCV price data.

Forward log return at horizon h:
    fwd_ret_h[t] = log(close[t + h]) - log(close[t])

This is the continuously-compounded h-period holding-period return.
Rows near the end of each ticker's series will produce NaN (no future price).
"""

import numpy as np
import pandas as pd


def compute_forward_log_returns(
    price_df: pd.DataFrame,
    horizons: list = [1, 5, 20],
) -> pd.DataFrame:
    """
    Compute forward log returns for each ticker at multiple horizons.

    Args:
        price_df: Long-format DataFrame with at minimum columns [ticker, date, close].
                  Must be sorted by date within each ticker group (the function sorts
                  internally, but providing pre-sorted data is faster).
        horizons: Forward horizons in *trading days* (e.g. [1, 5, 20]).

    Returns:
        DataFrame with columns [ticker, date, fwd_ret_1d, fwd_ret_5d, fwd_ret_20d]
        (column names derived from the horizon values). Rows at the end of each
        ticker's series where no forward price exists will contain NaN.
    """
    result_frames = []

    for ticker, grp in price_df.groupby("ticker"):
        grp = grp.sort_values("date").copy().set_index("date")
        log_close = np.log(grp["close"])

        ret_df = pd.DataFrame({"date": grp.index, "ticker": ticker})

        for h in horizons:
            # shift(-h) aligns the price h steps ahead with the current row
            ret_df[f"fwd_ret_{h}d"] = (log_close.shift(-h) - log_close).values

        result_frames.append(ret_df)

    return pd.concat(result_frames, ignore_index=True)
