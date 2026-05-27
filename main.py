"""
commit-alpha — Tier 1: GitHub activity → stock returns alpha signal pipeline.

Entry point. Runs the full pipeline:
  1. Fetch weekly GitHub metrics (commits, contributors, stars)
  2. Fetch daily stock OHLCV prices
  3. Compute forward log returns at 1d / 5d / 20d horizons
  4. Join GitHub signals to stock return dates via merge_asof
  5. Save joined signals to data/signals.csv
  6. Run Pearson + Spearman correlation study
  7. Save heatmap to outputs/correlation_heatmap.png

Usage:
    export GITHUB_TOKEN=ghp_...
    python main.py
"""

import os
import logging
import pandas as pd

import config
from data.github_fetcher import fetch_all_github_signals
from data.price_fetcher import fetch_price_data
from features.returns import compute_forward_log_returns
from analysis.correlation import compute_correlations, plot_correlation_heatmap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SIGNALS_PATH = os.path.join("data", "signals.csv")
HEATMAP_PATH = os.path.join("outputs", "correlation_heatmap.png")

SIGNAL_COLS = ["commit_count", "contributor_count", "star_count"]
RETURN_COLS = [f"fwd_ret_{h}d" for h in config.RETURNS_HORIZONS]


# ── Join helper ───────────────────────────────────────────────────────────────

def join_signals_and_returns(
    github_df: pd.DataFrame,
    returns_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align weekly GitHub signals with stock return dates using a forward merge.

    GitHub's stats API reports weeks starting on Sunday. merge_asof with
    direction="forward" finds the nearest *later* trading day for each week_start,
    within a 5-day tolerance (skips missing data across weekends / holidays cleanly).

    Args:
        github_df:  Long-format DataFrame with [ticker, week_start, …signal cols…].
        returns_df: Long-format DataFrame with [ticker, date, …return cols…].

    Returns:
        Merged DataFrame with both signal and return columns.
    """
    github_df = github_df.copy()
    returns_df = returns_df.copy()

    github_df["week_start"] = pd.to_datetime(github_df["week_start"])
    returns_df["date"] = pd.to_datetime(returns_df["date"])

    merged_frames = []
    for ticker in config.TICKERS:
        gh = github_df[github_df["ticker"] == ticker].sort_values("week_start")
        ret = returns_df[returns_df["ticker"] == ticker].sort_values("date")

        if gh.empty or ret.empty:
            logger.warning(f"  {ticker}: missing GitHub or price data — skipped.")
            continue

        merged = pd.merge_asof(
            gh,
            ret[["ticker", "date"] + RETURN_COLS],
            left_on="week_start",
            right_on="date",
            by="ticker",
            direction="forward",
            tolerance=pd.Timedelta("5 days"),
        )
        merged_frames.append(merged)

    if not merged_frames:
        raise RuntimeError("No tickers survived the join step.")

    return pd.concat(merged_frames, ignore_index=True)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # ── 1. GitHub signals ─────────────────────────────────────────────────────
    logger.info("━━━ Step 1 / 5  Fetching GitHub signals ━━━")
    github_df = fetch_all_github_signals(
        ticker_to_repo=config.TICKER_TO_REPO,
        github_token=config.GITHUB_TOKEN,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
    )
    logger.info(f"GitHub signals: {len(github_df):,} rows | {github_df['ticker'].nunique()} tickers")

    # ── 2. Stock prices ───────────────────────────────────────────────────────
    logger.info("━━━ Step 2 / 5  Fetching stock prices ━━━")
    price_df = fetch_price_data(
        tickers=config.TICKERS,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
    )
    logger.info(f"Price data: {len(price_df):,} rows | {price_df['ticker'].nunique()} tickers")

    # ── 3. Forward returns ────────────────────────────────────────────────────
    logger.info("━━━ Step 3 / 5  Computing forward log returns ━━━")
    returns_df = compute_forward_log_returns(price_df, horizons=config.RETURNS_HORIZONS)
    logger.info(f"Returns: {len(returns_df):,} rows | horizons: {config.RETURNS_HORIZONS}d")

    # ── 4. Join signals + returns ─────────────────────────────────────────────
    logger.info("━━━ Step 4 / 5  Joining signals and returns ━━━")
    signals_df = join_signals_and_returns(github_df, returns_df)

    # Drop rows where any signal or return is missing
    n_before = len(signals_df)
    signals_df = signals_df.dropna(subset=SIGNAL_COLS + RETURN_COLS).reset_index(drop=True)
    logger.info(
        f"Joined: {n_before:,} rows → {len(signals_df):,} after dropping NaN  "
        f"| columns: {list(signals_df.columns)}"
    )

    signals_df.to_csv(SIGNALS_PATH, index=False)
    logger.info(f"Signals saved → {SIGNALS_PATH}")

    # ── 5. Correlation study ──────────────────────────────────────────────────
    logger.info("━━━ Step 5 / 5  Running correlation study ━━━")
    pearson_df, spearman_df = compute_correlations(signals_df, SIGNAL_COLS, RETURN_COLS)

    print("\n" + "=" * 55)
    print("  Pearson correlations (GitHub signal vs forward return)")
    print("=" * 55)
    print(pearson_df.to_string())
    print("\n" + "=" * 55)
    print("  Spearman correlations")
    print("=" * 55)
    print(spearman_df.to_string())
    print()

    plot_correlation_heatmap(pearson_df, spearman_df, HEATMAP_PATH)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
