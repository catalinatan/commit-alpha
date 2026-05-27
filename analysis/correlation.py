"""
Computes Pearson and Spearman correlations between GitHub signals and forward returns,
then renders a side-by-side heatmap saved as a PNG.
"""

import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

logger = logging.getLogger(__name__)


def compute_correlations(
    df: pd.DataFrame,
    signal_cols: list,
    return_cols: list,
) -> tuple:
    """
    Compute pairwise Pearson and Spearman correlations between signal and return columns.

    Uses pairwise-complete observations (rows where both columns are non-NaN) so
    different return horizons can have different effective sample sizes.

    Args:
        df:           DataFrame containing both signal and return columns.
        signal_cols:  Column names for GitHub signals (x-axis in heatmap).
        return_cols:  Column names for forward returns (y-axis in heatmap).

    Returns:
        Tuple ``(pearson_df, spearman_df)`` — DataFrames indexed by return_cols,
        with signal_cols as columns.
    """
    pearson_vals: dict = {}
    spearman_vals: dict = {}

    for ret_col in return_cols:
        pearson_row: dict = {}
        spearman_row: dict = {}

        for sig_col in signal_cols:
            valid = df[[sig_col, ret_col]].dropna()
            n = len(valid)

            if n < 10:
                logger.warning(
                    f"Only {n} complete observations for ({sig_col}, {ret_col}); "
                    "correlation will be NaN."
                )
                pearson_row[sig_col] = np.nan
                spearman_row[sig_col] = np.nan
                continue

            x = valid[sig_col].values
            y = valid[ret_col].values

            pearson_r, _ = stats.pearsonr(x, y)
            spearman_r, _ = stats.spearmanr(x, y)

            pearson_row[sig_col] = round(pearson_r, 4)
            spearman_row[sig_col] = round(spearman_r, 4)

        pearson_vals[ret_col] = pearson_row
        spearman_vals[ret_col] = spearman_row

    pearson_df = pd.DataFrame(pearson_vals).T   # shape: (len(return_cols), len(signal_cols))
    spearman_df = pd.DataFrame(spearman_vals).T

    return pearson_df, spearman_df


def plot_correlation_heatmap(
    pearson_df: pd.DataFrame,
    spearman_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Render and save a side-by-side Pearson / Spearman correlation heatmap.

    Colour scale is symmetric around zero (red = negative, green = positive)
    with a fixed range of [-0.5, 0.5] — widen if signals are stronger than expected.

    Args:
        pearson_df:   DataFrame from :func:`compute_correlations` (Pearson).
        spearman_df:  DataFrame from :func:`compute_correlations` (Spearman).
        output_path:  Absolute or relative path for the saved PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("GitHub Signals → Forward Return Correlations", fontsize=14, fontweight="bold")

    cmap = "RdYlGn"
    vmin, vmax = -0.5, 0.5

    for ax, corr_df, method in zip(axes, [pearson_df, spearman_df], ["Pearson", "Spearman"]):
        sns.heatmap(
            corr_df,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"shrink": 0.75, "label": "correlation"},
        )
        ax.set_title(f"{method} r", fontsize=12)
        ax.set_xlabel("GitHub signal", fontsize=10)
        ax.set_ylabel("Forward return horizon", fontsize=10)
        ax.tick_params(axis="x", rotation=20)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Heatmap saved → {output_path}")
