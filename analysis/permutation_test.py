"""
Permutation test for statistical significance of the long/short Sharpe.

Why this exists
---------------
A strategy with a Sharpe of (say) 1.4 on 40 weeks of out-of-sample data sounds great
until you ask: how often would a *random* signal produce a Sharpe at least that high
on the same dataset? Without that null distribution, you cannot distinguish real edge
from p-hacking on a small sample.

Method
------
1. Hold the OOS predictions fixed (model output is treated as the signal).
2. For each of N permutations:
     - Shuffle the realised forward returns *within each week* (preserves the
       cross-sectional return distribution but destroys any signal-return link).
     - Re-run the long/short strategy and record the resulting annualised Sharpe.
3. Compare the *observed* Sharpe to the null distribution:
     p-value = P(null Sharpe ≥ observed Sharpe) under H0 (no skill).

Maps directly onto what Citadel/Two Sigma describe as "risk/overfit controls":
shows that the reported Sharpe is unlikely under the null of zero predictive power.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from strategy.sharpe import compute_long_short_returns, compute_sharpe

logger = logging.getLogger(__name__)


def _shuffle_returns_within_week(
    df: pd.DataFrame,
    return_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Permute the return column independently within each week.

    This preserves the marginal distribution of weekly cross-sectional returns
    (so the null is "same returns, but ticker labels scrambled within the week"),
    breaking only the association between model score and realised return.
    """
    out = df.copy()
    for week, idx in out.groupby("week_start").groups.items():
        vals = out.loc[idx, return_col].values.copy()
        rng.shuffle(vals)
        out.loc[idx, return_col] = vals
    return out


def run_permutation_test(
    oos_preds: pd.DataFrame,
    signal_col: str = "gbm_prob",
    return_col: str = "fwd_ret_5d",
    n_long: int = 2,
    n_short: int = 2,
    n_permutations: int = 1000,
    periods_per_year: int = 52,
    seed: int = 42,
) -> dict:
    """
    Run a within-week return-shuffle permutation test on the long/short Sharpe.

    Args:
        oos_preds:        Out-of-sample prediction DataFrame.
        signal_col:       Ranking signal (default "gbm_prob").
        return_col:       Realised forward return column to permute.
        n_long, n_short:  Strategy leg sizes.
        n_permutations:   Number of label shuffles (1000 is the standard default).
        periods_per_year: 52 for weekly.
        seed:             RNG seed for reproducibility.

    Returns:
        dict with keys:
          observed_sharpe   — Sharpe on the un-permuted data
          null_sharpes      — np.ndarray of length n_permutations
          p_value           — P(null Sharpe ≥ observed) ∈ [1/(N+1), 1]
          null_mean         — mean of null distribution
          null_std          — std  of null distribution
          z_score           — (observed - null_mean) / null_std
          n_permutations    — N
    """
    rng = np.random.default_rng(seed)

    observed_returns = compute_long_short_returns(
        oos_preds, signal_col=signal_col, return_col=return_col,
        n_long=n_long, n_short=n_short,
    )
    observed_sharpe = compute_sharpe(observed_returns, periods_per_year=periods_per_year)

    null_sharpes = np.empty(n_permutations, dtype=float)
    log_every = max(1, n_permutations // 10)

    for i in range(n_permutations):
        shuffled = _shuffle_returns_within_week(oos_preds, return_col, rng)
        try:
            sim_returns = compute_long_short_returns(
                shuffled, signal_col=signal_col, return_col=return_col,
                n_long=n_long, n_short=n_short,
            )
            null_sharpes[i] = compute_sharpe(sim_returns, periods_per_year=periods_per_year)
        except RuntimeError:
            null_sharpes[i] = np.nan

        if (i + 1) % log_every == 0:
            logger.info(f"  permutation {i + 1:>5} / {n_permutations}")

    null_sharpes = null_sharpes[~np.isnan(null_sharpes)]

    # One-sided test: are we to the right of the null?
    # +1 numerator / denominator is the standard small-sample correction (avoids p=0).
    n_at_or_above = int((null_sharpes >= observed_sharpe).sum())
    p_value = (n_at_or_above + 1) / (len(null_sharpes) + 1)

    null_mean = float(np.mean(null_sharpes))
    null_std  = float(np.std(null_sharpes, ddof=1))
    z_score   = (observed_sharpe - null_mean) / null_std if null_std > 0 else float("nan")

    return {
        "observed_sharpe": float(observed_sharpe),
        "null_sharpes":    null_sharpes,
        "p_value":         float(p_value),
        "null_mean":       null_mean,
        "null_std":        null_std,
        "z_score":         float(z_score),
        "n_permutations":  int(len(null_sharpes)),
    }


def summarise_permutation_test(result: dict) -> pd.Series:
    """Concise tear-sheet for printing alongside the other strategy metrics."""
    return pd.Series({
        "observed_sharpe": round(result["observed_sharpe"], 3),
        "null_mean":       round(result["null_mean"], 3),
        "null_std":        round(result["null_std"], 3),
        "z_score":         round(result["z_score"], 3),
        "p_value":         round(result["p_value"], 4),
        "n_permutations":  result["n_permutations"],
    })


def plot_permutation_distribution(result: dict, output_path: str) -> None:
    """
    Plot the null Sharpe distribution with the observed Sharpe overlaid.

    A well-behaved real signal should sit far in the right tail. If the observed
    bar lands inside the bulk of the null, the reported Sharpe is plausibly luck.
    """
    null = result["null_sharpes"]
    observed = result["observed_sharpe"]
    p_val = result["p_value"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(null, bins=40, color="#90CAF9", edgecolor="white", alpha=0.85,
            label=f"Null distribution (N={len(null)})")
    ax.axvline(observed, color="#D32F2F", linewidth=2.2,
               label=f"Observed Sharpe = {observed:.2f}")
    ax.axvline(np.mean(null), color="#1976D2", linewidth=1.2, linestyle="--",
               label=f"Null mean = {np.mean(null):.2f}")

    ax.set_title("Permutation Test — Strategy Sharpe vs Null Distribution",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Annualised Sharpe ratio")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper left", fontsize=9)
    ax.text(
        0.98, 0.95,
        f"p-value = {p_val:.4f}\nz-score = {result['z_score']:.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow", edgecolor="gray"),
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Permutation-test plot saved → {output_path}")
