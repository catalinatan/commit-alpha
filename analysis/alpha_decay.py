"""
Alpha decay analysis — how fast does the signal's predictive power degrade?

For each forward-return horizon h ∈ {1, 3, 5, 10, 20} trading days we compute:

  - Information coefficient (IC):     pooled Spearman rank-correlation between the
                                       model score and the realised h-day return.
  - Mean weekly rank-IC (IC_t):       same correlation computed within each week,
                                       then averaged. Standard quant-research IC.
  - IC information ratio (ICIR):      mean(IC_t) / std(IC_t). Stability of the signal.
  - Hit rate:                          P(sign(score - 0.5) == sign(return)).
  - Long/short Sharpe (annualised):    re-running the strategy with fwd_ret_h as P&L.

A signal that decays slowly will hold IC and Sharpe across longer horizons; one that
decays fast collapses to zero (or flips sign) by the 20-day mark.

Interview-ready answer to "how long does your alpha live?" → the plot from this module.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from strategy.sharpe import compute_sharpe

logger = logging.getLogger(__name__)


def _pooled_ic(scores: np.ndarray, returns: np.ndarray) -> float:
    """Spearman rank-correlation across the full pooled sample."""
    if len(scores) < 5 or np.std(returns) == 0:
        return float("nan")
    rho, _ = stats.spearmanr(scores, returns)
    return float(rho)


def _weekly_rank_ic(df: pd.DataFrame, score_col: str, ret_col: str) -> pd.Series:
    """Compute Spearman IC within each week, returning the time-series of weekly ICs."""
    weekly = []
    for week, grp in df.groupby("week_start"):
        valid = grp[[score_col, ret_col]].dropna()
        if len(valid) < 3 or valid[ret_col].std() == 0:
            continue
        rho, _ = stats.spearmanr(valid[score_col].values, valid[ret_col].values)
        weekly.append({"week_start": week, "ic": rho})
    return pd.DataFrame(weekly).set_index("week_start")["ic"] if weekly else pd.Series(dtype=float)


def _hit_rate(scores: np.ndarray, returns: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of obs where (score > 0.5) matches (return > 0)."""
    pred_up = scores > threshold
    actual_up = returns > 0
    return float((pred_up == actual_up).mean()) if len(scores) else float("nan")


def _horizon_sharpe(
    df: pd.DataFrame,
    score_col: str,
    ret_col: str,
    n_long: int = 2,
    n_short: int = 2,
    periods_per_year: int = 52,
) -> float:
    """Long/short Sharpe computed against an arbitrary forward-return horizon."""
    rows = []
    for _, grp in df.groupby("week_start"):
        grp = grp.dropna(subset=[score_col, ret_col])
        if len(grp) < n_long + n_short:
            continue
        ranked = grp.sort_values(score_col, ascending=False)
        long_r  = ranked.head(n_long)[ret_col].mean()
        short_r = ranked.tail(n_short)[ret_col].mean()
        rows.append(long_r - short_r)
    if not rows:
        return float("nan")
    s = pd.Series(rows)
    return float(compute_sharpe(s, periods_per_year=periods_per_year))


def compute_alpha_decay(
    oos_preds: pd.DataFrame,
    signals_df: pd.DataFrame,
    horizons: list,
    score_col: str = "gbm_prob",
) -> pd.DataFrame:
    """
    Build a tear-sheet of predictive-power metrics across forward-return horizons.

    Args:
        oos_preds:  Out-of-sample predictions from the walk-forward classifier
                    (cols [ticker, week_start, fwd_ret_5d, lr_prob, gbm_prob]).
        signals_df: Full joined signals+returns DataFrame (must contain fwd_ret_{h}d
                    columns for every h in ``horizons``).
        horizons:   Forward-return horizons in trading days, e.g. [1, 3, 5, 10, 20].
        score_col:  Which model probability to evaluate ("gbm_prob" or "lr_prob").

    Returns:
        DataFrame indexed by horizon with columns:
        [pooled_ic, mean_weekly_ic, ic_ir, hit_rate, sharpe, n_obs].
    """
    needed = [f"fwd_ret_{h}d" for h in horizons]
    missing = [c for c in needed if c not in signals_df.columns]
    if missing:
        raise ValueError(
            f"signals_df is missing required return columns: {missing}. "
            f"Add the corresponding horizons to config.RETURNS_HORIZONS and rerun the pipeline."
        )

    # Bring all horizon returns onto the OOS prediction rows
    keys = ["ticker", "week_start"]
    enriched = oos_preds.merge(
        signals_df[keys + needed].drop_duplicates(subset=keys),
        on=keys,
        how="left",
        suffixes=("", "_dup"),
    )
    # If oos_preds already had fwd_ret_5d, prefer the version from signals_df for consistency
    for h in horizons:
        col = f"fwd_ret_{h}d"
        dup = f"{col}_dup"
        if dup in enriched.columns:
            enriched[col] = enriched[dup].combine_first(enriched[col])
            enriched = enriched.drop(columns=[dup])

    rows = []
    for h in horizons:
        ret_col = f"fwd_ret_{h}d"
        valid = enriched.dropna(subset=[score_col, ret_col])
        scores  = valid[score_col].values
        returns = valid[ret_col].values

        weekly_ic = _weekly_rank_ic(valid, score_col, ret_col)
        ic_mean = float(weekly_ic.mean()) if not weekly_ic.empty else float("nan")
        ic_std  = float(weekly_ic.std())  if not weekly_ic.empty else float("nan")
        ic_ir   = ic_mean / ic_std if ic_std and not np.isnan(ic_std) and ic_std > 0 else float("nan")

        rows.append({
            "horizon_days":   h,
            "pooled_ic":      round(_pooled_ic(scores, returns), 4),
            "mean_weekly_ic": round(ic_mean, 4) if not np.isnan(ic_mean) else np.nan,
            "ic_ir":          round(ic_ir, 4)   if not np.isnan(ic_ir) else np.nan,
            "hit_rate":       round(_hit_rate(scores, returns), 4),
            "sharpe":         round(_horizon_sharpe(valid, score_col, ret_col), 3),
            "n_obs":          int(len(valid)),
        })

    return pd.DataFrame(rows).set_index("horizon_days")


def plot_alpha_decay(decay_df: pd.DataFrame, output_path: str) -> None:
    """
    Render a two-panel decay chart: IC vs horizon (top) and Sharpe vs horizon (bottom).

    A signal with "fast decay" peaks at short horizons then collapses; a robust signal
    holds its IC and Sharpe across the 1d → 20d window.

    Args:
        decay_df:    Output of :func:`compute_alpha_decay`.
        output_path: PNG file to write.
    """
    fig, (ax_ic, ax_sr) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("Alpha Decay — Signal Predictive Power vs Forward Horizon",
                 fontsize=13, fontweight="bold")

    horizons = decay_df.index.values

    # ── IC panel ─────────────────────────────────────────────────────────────
    ax_ic.plot(horizons, decay_df["pooled_ic"], marker="o", linewidth=2,
               color="#1976D2", label="Pooled Spearman IC")
    ax_ic.plot(horizons, decay_df["mean_weekly_ic"], marker="s", linewidth=2,
               color="#43A047", label="Mean weekly rank-IC")
    ax_ic.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax_ic.set_ylabel("Information coefficient")
    ax_ic.legend(loc="best", fontsize=9)
    ax_ic.grid(alpha=0.3)

    # ── Sharpe panel ─────────────────────────────────────────────────────────
    colors = ["#4CAF50" if s > 0 else "#F44336" for s in decay_df["sharpe"].fillna(0)]
    ax_sr.bar(horizons, decay_df["sharpe"], color=colors, width=horizons.max() * 0.05)
    ax_sr.axhline(0, color="black", linewidth=0.7)
    ax_sr.set_xlabel("Forward return horizon (trading days)")
    ax_sr.set_ylabel("Annualised Sharpe")
    ax_sr.set_xticks(horizons)
    ax_sr.grid(axis="y", alpha=0.3)

    # Annotate the Sharpe values above/below each bar
    for h, s in zip(horizons, decay_df["sharpe"]):
        if pd.isna(s):
            continue
        offset = 0.05 if s >= 0 else -0.15
        ax_sr.text(h, s + offset, f"{s:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Alpha-decay chart saved → {output_path}")
