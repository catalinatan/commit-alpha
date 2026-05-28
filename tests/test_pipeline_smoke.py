"""
Smoke tests that exercise the Tier 2/3 modules end-to-end on a synthetic fixture.

These run without GITHUB_TOKEN or network access, so they're safe for CI. They check
that the modules import cleanly, accept the documented inputs, and produce outputs
of the expected shape — not the numerical quality of the signal.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from analysis.alpha_decay import compute_alpha_decay, plot_alpha_decay
from analysis.permutation_test import run_permutation_test
from models.classifier import run_walk_forward_classification
from strategy.sharpe import compute_long_short_returns, summarise_strategy


@pytest.fixture(scope="module")
def synthetic_signals() -> pd.DataFrame:
    """A 6-ticker × 40-week panel with a weak true signal in commit_count."""
    rng = np.random.default_rng(0)
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    weeks = pd.date_range("2024-01-07", periods=40, freq="W-SUN")

    rows = []
    for t_i, ticker in enumerate(tickers):
        commits = rng.integers(50, 500, size=len(weeks))
        contribs = rng.integers(5, 50, size=len(weeks))
        stars = 10_000 + 1_000 * t_i
        for i, week in enumerate(weeks):
            # Weak embedded signal: high commits → mildly positive 5d return
            base = (commits[i] - 250) / 5_000
            rows.append({
                "ticker": ticker,
                "week_start": week,
                "commit_count": int(commits[i]),
                "contributor_count": int(contribs[i]),
                "star_count": stars,
                "fwd_ret_1d":  base * 0.4 + rng.normal(0, 0.02),
                "fwd_ret_3d":  base * 0.6 + rng.normal(0, 0.025),
                "fwd_ret_5d":  base       + rng.normal(0, 0.03),
                "fwd_ret_10d": base * 0.5 + rng.normal(0, 0.04),
                "fwd_ret_20d": base * 0.2 + rng.normal(0, 0.05),
            })
    return pd.DataFrame(rows)


def test_walk_forward_classification_shape(synthetic_signals):
    lr_res, gbm_res, oos = run_walk_forward_classification(synthetic_signals, n_splits=4)
    assert {"fold", "n_test", "accuracy", "p_value"} <= set(lr_res.columns)
    assert {"fold", "n_test", "accuracy", "p_value"} <= set(gbm_res.columns)
    assert {"ticker", "week_start", "lr_prob", "gbm_prob", "fwd_ret_5d"} <= set(oos.columns)
    assert len(oos) > 0


def test_long_short_strategy_runs(synthetic_signals):
    _, _, oos = run_walk_forward_classification(synthetic_signals, n_splits=4)
    rets = compute_long_short_returns(oos)
    summary = summarise_strategy(rets)
    for key in ("n_weeks", "sharpe", "ann_return", "max_drawdown"):
        assert key in summary.index


def test_alpha_decay_emits_one_row_per_horizon(synthetic_signals):
    _, _, oos = run_walk_forward_classification(synthetic_signals, n_splits=4)
    decay = compute_alpha_decay(oos, synthetic_signals, horizons=[1, 3, 5, 10, 20])
    assert list(decay.index) == [1, 3, 5, 10, 20]
    assert {"pooled_ic", "mean_weekly_ic", "ic_ir", "hit_rate", "sharpe", "n_obs"} <= set(decay.columns)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "decay.png")
        plot_alpha_decay(decay, path)
        assert os.path.exists(path) and os.path.getsize(path) > 0


def test_permutation_test_returns_valid_p_value(synthetic_signals):
    _, _, oos = run_walk_forward_classification(synthetic_signals, n_splits=4)
    # 50 perms keeps the test fast; 1000 is the production default.
    result = run_permutation_test(oos, n_permutations=50, seed=1)
    assert 0.0 < result["p_value"] <= 1.0
    assert result["n_permutations"] > 0
    assert len(result["null_sharpes"]) > 0
    # Observed Sharpe should be a real number even if not significant on synthetic data
    assert np.isfinite(result["observed_sharpe"])
