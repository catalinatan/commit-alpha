"""
Long/short strategy backtester and performance analytics.

Construction rules (applied weekly, out-of-sample only):
  - Rank all tickers by predicted probability of going up (gbm_prob by default).
  - Long the top-N tickers (equal-weighted).
  - Short the bottom-N tickers (equal-weighted).
  - Weekly P&L = mean(long actual returns) − mean(short actual returns).

Sharpe ratio is annualised assuming 52 trading weeks per year.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_long_short_returns(
    oos_preds: pd.DataFrame,
    signal_col: str = "gbm_prob",
    return_col: str = "fwd_ret_5d",
    n_long: int = 2,
    n_short: int = 2,
) -> pd.Series:
    """
    Simulate an equal-weighted long/short portfolio on out-of-sample predictions.

    At each week tickers are ranked by ``signal_col`` (higher = model is more
    confident the stock will rise). The ``n_long`` top-ranked go into the long leg;
    the ``n_short`` bottom-ranked go into the short leg.

    Args:
        oos_preds:  Out-of-sample predictions DataFrame with columns
                    [ticker, week_start, fwd_ret_5d, lr_prob, gbm_prob].
        signal_col: Column used to rank tickers each week.
        return_col: Actual return column used for P&L.
        n_long:     Number of tickers held long each week.
        n_short:    Number of tickers sold short each week.

    Returns:
        pd.Series of weekly portfolio log-returns indexed by week_start.
        Weeks where fewer than (n_long + n_short) tickers have valid data are skipped.
    """
    rows = []
    for week, grp in oos_preds.groupby("week_start"):
        grp = grp.dropna(subset=[signal_col, return_col])
        if len(grp) < n_long + n_short:
            continue

        ranked = grp.sort_values(signal_col, ascending=False)
        long_ret  = ranked.head(n_long)[return_col].mean()
        short_ret = ranked.tail(n_short)[return_col].mean()
        rows.append({"week_start": week, "portfolio_return": long_ret - short_ret})

    if not rows:
        raise RuntimeError("No valid weeks found for long/short strategy.")

    return pd.DataFrame(rows).set_index("week_start")["portfolio_return"]


def compute_sharpe(
    returns: pd.Series,
    periods_per_year: int = 52,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Annualised Sharpe ratio: (mean excess return / std) × sqrt(periods_per_year).

    Args:
        returns:          Series of periodic (e.g. weekly) returns.
        periods_per_year: 52 for weekly data.
        risk_free_rate:   Per-period risk-free rate (default 0).

    Returns:
        Float. Returns NaN if return series has zero variance.
    """
    excess = returns - risk_free_rate
    if excess.std() == 0:
        return float("nan")
    return (excess.mean() / excess.std()) * np.sqrt(periods_per_year)


def summarise_strategy(returns: pd.Series, periods_per_year: int = 52) -> pd.Series:
    """
    Compute a concise performance tear-sheet for a return series.

    Args:
        returns:          Weekly portfolio return series.
        periods_per_year: 52 for weekly.

    Returns:
        pd.Series with keys: n_weeks, total_return, ann_return, ann_vol,
        sharpe, win_rate, max_drawdown.
    """
    cum = returns.cumsum()
    rolling_max = cum.cummax()
    max_dd = (cum - rolling_max).min()

    return pd.Series({
        "n_weeks":       len(returns),
        "total_return":  round(returns.sum(), 4),
        "ann_return":    round(returns.mean() * periods_per_year, 4),
        "ann_vol":       round(returns.std() * np.sqrt(periods_per_year), 4),
        "sharpe":        round(compute_sharpe(returns, periods_per_year), 3),
        "win_rate":      round((returns > 0).mean(), 3),
        "max_drawdown":  round(max_dd, 4),
    })


def plot_strategy_returns(
    returns: pd.Series,
    output_path: str,
    title: str = "Long/Short Portfolio — Cumulative Returns (OOS)",
) -> None:
    """
    Plot cumulative log-returns and a zero benchmark, saving to ``output_path``.

    Args:
        returns:     Weekly portfolio return series indexed by week_start.
        output_path: Path for the saved PNG.
        title:       Chart title.
    """
    cum = returns.cumsum()
    sharpe = compute_sharpe(returns)

    fig, (ax_cum, ax_wk) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # ── Cumulative return ──────────────────────────────────────────────────────
    ax_cum.plot(cum.index, cum.values, color="#2196F3", linewidth=2, label="L/S portfolio")
    ax_cum.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_cum.fill_between(cum.index, cum.values, 0, where=(cum.values >= 0),
                        alpha=0.15, color="#4CAF50")
    ax_cum.fill_between(cum.index, cum.values, 0, where=(cum.values < 0),
                        alpha=0.15, color="#F44336")
    ax_cum.set_ylabel("Cumulative log-return")
    ax_cum.legend(loc="upper left", fontsize=9)
    ax_cum.text(
        0.98, 0.05,
        f"Annualised Sharpe: {sharpe:.2f}",
        transform=ax_cum.transAxes,
        ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
    )

    # ── Weekly bar chart ───────────────────────────────────────────────────────
    colors = ["#4CAF50" if r > 0 else "#F44336" for r in returns.values]
    ax_wk.bar(returns.index, returns.values, color=colors, width=5)
    ax_wk.axhline(0, color="black", linewidth=0.6)
    ax_wk.set_ylabel("Weekly return")
    ax_wk.set_xlabel("Week")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
