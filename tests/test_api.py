"""FastAPI route smoke tests using a synthetic signals/predictions fixture."""

import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """
    Build a tiny signals + predictions fixture, then import the API with cwd
    pointing at that tmp dir so the lazy loaders pick up our files.
    """
    tmp = tmp_path_factory.mktemp("api-fixture")
    (tmp / "data").mkdir()

    rng = np.random.default_rng(7)
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    weeks = pd.date_range("2024-01-07", periods=12, freq="W-SUN")
    rows = []
    for ticker in tickers:
        for week in weeks:
            r5 = rng.normal(0, 0.03)
            rows.append({
                "ticker": ticker,
                "week_start": week,
                "commit_count": int(rng.integers(50, 500)),
                "contributor_count": int(rng.integers(5, 50)),
                "star_count": 10_000,
                "fwd_ret_1d":  rng.normal(0, 0.02),
                "fwd_ret_3d":  rng.normal(0, 0.025),
                "fwd_ret_5d":  r5,
                "fwd_ret_10d": rng.normal(0, 0.04),
                "fwd_ret_20d": rng.normal(0, 0.05),
                "lr_prob":  rng.uniform(),
                "gbm_prob": rng.uniform(),
                "y_true":   int(r5 > 0),
            })
    df = pd.DataFrame(rows)

    signal_cols = ["ticker", "week_start", "commit_count", "contributor_count", "star_count",
                   "fwd_ret_1d", "fwd_ret_3d", "fwd_ret_5d", "fwd_ret_10d", "fwd_ret_20d"]
    pred_cols   = ["ticker", "week_start", "fwd_ret_5d", "y_true", "lr_prob", "gbm_prob"]
    df[signal_cols].to_csv(tmp / "data" / "signals.csv", index=False)
    df[pred_cols].to_csv(tmp / "data" / "oos_predictions.csv", index=False)

    prev_cwd = os.getcwd()
    os.chdir(tmp)
    try:
        from api.main import app  # imported after chdir so its relative paths resolve
        yield TestClient(app)
    finally:
        os.chdir(prev_cwd)


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "endpoints" in r.json()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["signals_loaded"] and body["predictions_loaded"]


def test_signals_latest(client):
    r = client.get("/signals/latest")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) >= 1
    assert "week_start" in rows[0]


def test_predictions_latest_ranks_baskets(client):
    r = client.get("/predictions/latest")
    assert r.status_code == 200
    body = r.json()
    assert len(body["long_basket"])  == 2
    assert len(body["short_basket"]) == 2
    assert body["long_basket"][0] != body["short_basket"][-1]


def test_backtest(client):
    r = client.get("/backtest")
    assert r.status_code == 200
    body = r.json()
    for key in ("sharpe", "ann_return", "max_drawdown", "n_weeks"):
        assert key in body


def test_alpha_decay(client):
    r = client.get("/alpha-decay")
    assert r.status_code == 200
    rows = r.json()
    horizons = sorted(row["horizon_days"] for row in rows)
    assert horizons == [1, 3, 5, 10, 20]
