"""
FastAPI service wrapping the commit-alpha signal pipeline.

Endpoints
---------
GET  /                  – metadata + endpoint catalogue
GET  /health            – liveness probe (used by CI / Docker healthcheck)
GET  /signals/latest    – most recent week of joined GitHub signals + forward returns
GET  /predictions       – OOS model predictions (filterable by ticker)
GET  /predictions/latest– latest week's model scores ranked long → short
GET  /backtest          – strategy tear-sheet (Sharpe, drawdown, win-rate, …)
GET  /alpha-decay       – IC / Sharpe by forward-return horizon
POST /retrain           – re-run the full pipeline (long; requires GITHUB_TOKEN)

Run locally:
    uvicorn api.main:app --reload --port 8000

The service loads its predictions from the artefacts produced by ``python main.py``
(``data/signals.csv`` and ``data/oos_predictions.csv``). Hitting /retrain regenerates
them on disk; everything else reads them with no live GitHub or yfinance calls,
keeping per-request latency at sub-100 ms.
"""

import os
import logging
import subprocess
import sys
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

import config
from analysis.alpha_decay import compute_alpha_decay
from strategy.sharpe import compute_long_short_returns, summarise_strategy

logger = logging.getLogger("commit-alpha.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

SIGNALS_PATH   = os.path.join("data", "signals.csv")
OOS_PREDS_PATH = os.path.join("data", "oos_predictions.csv")

app = FastAPI(
    title="commit-alpha API",
    description="GitHub developer activity → equity alpha signals.",
    version="1.0.0",
)


# ── Models ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    signals_loaded: bool
    predictions_loaded: bool
    n_signals: int
    n_predictions: int


class BacktestResponse(BaseModel):
    n_weeks: int
    total_return: float
    ann_return: float
    ann_vol: float
    sharpe: float
    win_rate: float
    max_drawdown: float
    signal_col: str
    return_col: str
    n_long: int
    n_short: int


# ── Lazy loaders (re-read on each call so /retrain takes effect immediately) ─

def _load_signals() -> pd.DataFrame:
    if not os.path.exists(SIGNALS_PATH):
        raise HTTPException(
            status_code=503,
            detail=f"signals.csv not found at {SIGNALS_PATH}. Run `python main.py` or POST /retrain first.",
        )
    df = pd.read_csv(SIGNALS_PATH, parse_dates=["week_start"])
    return df


def _load_predictions() -> pd.DataFrame:
    if not os.path.exists(OOS_PREDS_PATH):
        raise HTTPException(
            status_code=503,
            detail=f"oos_predictions.csv not found at {OOS_PREDS_PATH}. Run `python main.py` or POST /retrain first.",
        )
    df = pd.read_csv(OOS_PREDS_PATH, parse_dates=["week_start"])
    return df


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root() -> dict:
    return {
        "service": "commit-alpha",
        "version": app.version,
        "endpoints": [
            "/health", "/signals/latest", "/predictions",
            "/predictions/latest", "/backtest", "/alpha-decay", "/retrain",
        ],
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    signals_ok = os.path.exists(SIGNALS_PATH)
    preds_ok   = os.path.exists(OOS_PREDS_PATH)
    n_sig = len(pd.read_csv(SIGNALS_PATH))   if signals_ok else 0
    n_pr  = len(pd.read_csv(OOS_PREDS_PATH)) if preds_ok   else 0
    return HealthResponse(
        status="ok" if signals_ok and preds_ok else "degraded",
        signals_loaded=signals_ok,
        predictions_loaded=preds_ok,
        n_signals=n_sig,
        n_predictions=n_pr,
    )


@app.get("/signals/latest")
def signals_latest(ticker: Optional[str] = None) -> list:
    df = _load_signals()
    if ticker:
        df = df[df["ticker"].str.upper() == ticker.upper()]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No signals for ticker {ticker!r}.")
    latest_week = df["week_start"].max()
    out = df[df["week_start"] == latest_week].copy()
    out["week_start"] = out["week_start"].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


@app.get("/predictions")
def predictions(
    ticker: Optional[str] = None,
    limit: int = Query(100, ge=1, le=10_000),
) -> list:
    df = _load_predictions().sort_values("week_start", ascending=False)
    if ticker:
        df = df[df["ticker"].str.upper() == ticker.upper()]
    df = df.head(limit).copy()
    df["week_start"] = df["week_start"].dt.strftime("%Y-%m-%d")
    return df.to_dict(orient="records")


@app.get("/predictions/latest")
def predictions_latest() -> dict:
    """
    Latest-week predictions ranked from long (top score) to short (bottom score),
    plus the recommended long/short basket for the configured leg sizes.
    """
    df = _load_predictions()
    latest_week = df["week_start"].max()
    latest = df[df["week_start"] == latest_week].sort_values("gbm_prob", ascending=False).copy()
    latest["week_start"] = latest["week_start"].dt.strftime("%Y-%m-%d")
    return {
        "week_start": latest["week_start"].iloc[0],
        "long_basket":  latest.head(2)["ticker"].tolist(),
        "short_basket": latest.tail(2)["ticker"].tolist(),
        "ranked": latest.to_dict(orient="records"),
    }


@app.get("/backtest", response_model=BacktestResponse)
def backtest(
    signal_col: str = "gbm_prob",
    return_col: str = "fwd_ret_5d",
    n_long: int = Query(2, ge=1, le=10),
    n_short: int = Query(2, ge=1, le=10),
) -> BacktestResponse:
    preds = _load_predictions()
    if signal_col not in preds.columns:
        raise HTTPException(status_code=400,
                            detail=f"signal_col {signal_col!r} not in predictions; available: {list(preds.columns)}")
    if return_col not in preds.columns:
        raise HTTPException(status_code=400,
                            detail=f"return_col {return_col!r} not in predictions; available: {list(preds.columns)}")
    ls = compute_long_short_returns(preds, signal_col=signal_col, return_col=return_col,
                                    n_long=n_long, n_short=n_short)
    summary = summarise_strategy(ls)
    return BacktestResponse(
        n_weeks=int(summary["n_weeks"]),
        total_return=float(summary["total_return"]),
        ann_return=float(summary["ann_return"]),
        ann_vol=float(summary["ann_vol"]),
        sharpe=float(summary["sharpe"]),
        win_rate=float(summary["win_rate"]),
        max_drawdown=float(summary["max_drawdown"]),
        signal_col=signal_col,
        return_col=return_col,
        n_long=n_long,
        n_short=n_short,
    )


@app.get("/alpha-decay")
def alpha_decay() -> list:
    preds = _load_predictions()
    signals = _load_signals()
    decay = compute_alpha_decay(preds, signals, horizons=config.RETURNS_HORIZONS,
                                score_col="gbm_prob")
    decay = decay.reset_index()
    return decay.to_dict(orient="records")


@app.post("/retrain")
def retrain() -> dict:
    """
    Re-run the full pipeline (`python main.py`) and refresh signals/predictions on disk.

    Long-running (minutes), and requires GITHUB_TOKEN in the env. In production this
    would be triggered via a scheduler / worker, not a web request — exposed here for
    convenience during development.
    """
    if not os.getenv("GITHUB_TOKEN"):
        raise HTTPException(
            status_code=400,
            detail="GITHUB_TOKEN env var is not set — pipeline would hit rate limits.",
        )
    try:
        proc = subprocess.run(
            [sys.executable, "main.py"],
            check=True, capture_output=True, text=True, timeout=900,
        )
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline failed:\n{exc.stderr[-2000:]}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Pipeline exceeded 15-minute timeout.")
    return {"status": "ok", "stdout_tail": proc.stdout[-1000:]}
