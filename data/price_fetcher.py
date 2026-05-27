"""
Fetches historical daily OHLCV price data for a list of tickers using yfinance.

Uses Ticker.history() per ticker rather than yf.download() so that column shapes
are consistent regardless of whether one or many tickers are requested.
"""

import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_price_data(
    tickers: list,
    start_date: str,
    end_date: str,
    extra_trading_days: int = 25,
) -> pd.DataFrame:
    """
    Download daily adjusted OHLCV data for all tickers.

    The end date is extended by ``extra_trading_days`` so that forward returns at
    the longest horizon (default 20d) can be computed for dates near ``end_date``.

    Args:
        tickers:            list of ticker symbols, e.g. ``["MSFT", "GOOGL"]``
        start_date:         ISO date string, e.g. ``"2024-06-01"``
        end_date:           ISO date string for the *signal* end (not the data end)
        extra_trading_days: buffer beyond ``end_date`` for forward return coverage

    Returns:
        Long-format DataFrame with columns
        [ticker, date, open, high, low, close, volume].
        Dates are tz-naive UTC. Prices are split- and dividend-adjusted (auto_adjust=True).
    """
    # Extend the window so we have enough prices to compute the longest forward return
    end_extended = (
        pd.Timestamp(end_date) + pd.offsets.BDay(extra_trading_days)
    ).strftime("%Y-%m-%d")

    frames = []
    for ticker in tickers:
        try:
            df = yf.Ticker(ticker).history(
                start=start_date,
                end=end_extended,
                auto_adjust=True,   # adjusts for splits and dividends
                actions=False,      # skip dividend/split event columns
            )
        except Exception as exc:
            logger.error(f"yfinance error for {ticker}: {exc}")
            continue

        if df.empty:
            logger.warning(f"No price data returned for {ticker}; skipping.")
            continue

        df.index.name = "date"
        df.columns = [c.lower() for c in df.columns]

        # Drop any extra columns yfinance might add (e.g. "capital gains")
        keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep_cols].reset_index()

        # Strip timezone info — stock prices are already in exchange local time
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df.insert(0, "ticker", ticker)

        frames.append(df)
        logger.info(f"  {ticker}: {len(df)} trading days ({df['date'].min().date()} – {df['date'].max().date()})")

    if not frames:
        raise ValueError(
            f"yfinance returned no data for any ticker in {tickers} "
            f"over {start_date} – {end_extended}."
        )

    return pd.concat(frames, ignore_index=True)
