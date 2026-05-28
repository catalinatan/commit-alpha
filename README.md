# commit-alpha

A quantitative research pipeline that turns GitHub developer activity into stock return alpha signals. Built in two tiers: data ingestion + correlation study (Tier 1), and walk-forward ML classification + long/short backtest (Tier 2).

## What it does

1. Fetches weekly GitHub metrics (commit count, contributor count, star count) for a configurable list of public tech repos mapped to listed equities
2. Pulls historical daily OHLCV prices via yfinance
3. Computes forward log returns at 1-day, 5-day, and 20-day horizons
4. Joins GitHub signals to stock return dates and runs a Pearson + Spearman correlation study
5. Trains logistic regression and gradient boosting classifiers under strict walk-forward validation (no look-ahead bias) to predict weekly return direction
6. Simulates a long/short portfolio (long top-2 tickers by model score, short bottom-2) and reports annualised Sharpe ratio, max drawdown, and win rate

## Project structure

```
commit-alpha/
├── config.py                  # tickers, repo map, date range, return horizons
├── main.py                    # 8-step pipeline entry point
├── requirements.txt
├── data/
│   ├── github_fetcher.py      # PyGitHub stats API → weekly signals
│   ├── price_fetcher.py       # yfinance → daily OHLCV
│   └── signals.csv            # output: joined signals + returns
├── features/
│   └── returns.py             # forward log return computation
├── models/
│   ├── walk_forward.py        # TimeSeriesSplit on unique weeks
│   └── classifier.py          # LR + GBM with per-fold StandardScaler
├── strategy/
│   └── sharpe.py              # long/short backtest, Sharpe, drawdown
├── analysis/
│   └── correlation.py         # Pearson/Spearman heatmap
└── outputs/
    ├── correlation_heatmap.png
    └── strategy_returns.png
```

## Quickstart

```bash
pip install -r requirements.txt

export GITHUB_TOKEN=ghp_...   # required — unauthenticated limit is 60 req/hr
python main.py
```

On Windows use `$env:GITHUB_TOKEN = "ghp_..."` in PowerShell.

A GitHub personal access token with default (read-only) scopes is sufficient. Generate one at Settings → Developer settings → Personal access tokens.

## Configuration

Edit `config.py` to swap tickers, repos, or date range:

```python
TICKERS = ["MSFT", "GOOGL", "META", "AAPL", "NVDA", "AMZN"]

TICKER_TO_REPO = {
    "MSFT": "microsoft/vscode",
    "GOOGL": "google/jax",
    "META":  "facebook/react",
    "AAPL":  "apple/swift",
    "NVDA":  "NVIDIA/TensorRT-LLM",
    "AMZN":  "aws/aws-cli",
}

START_DATE = "2024-06-01"
END_DATE   = "2025-05-01"
RETURNS_HORIZONS = [1, 5, 20]   # trading days
```

The GitHub stats API covers the last ~52 weeks from the time of the call. Keep `START_DATE` within that window.

## Outputs

| File | Description |
|---|---|
| `data/signals.csv` | Joined panel: (ticker, week) × (signals + forward returns) |
| `outputs/correlation_heatmap.png` | Pearson and Spearman heatmaps, signals vs return horizons |
| `outputs/strategy_returns.png` | Cumulative long/short returns with annotated Sharpe ratio |

## Design notes

**No look-ahead bias** — the walk-forward split (`TimeSeriesSplit`) is applied to the sorted sequence of unique week timestamps, not raw row indices. This ensures all tickers for a given week land in the same fold and the model is never trained on data from after its prediction date. The `StandardScaler` is fit on the training fold only and applied to the test fold.

**Why log returns** — log returns are time-additive and approximately normally distributed, which satisfies the assumptions of both the correlation tests and the Sharpe ratio calculation.

**Star count as a cross-sectional factor** — star count does not vary week-to-week for a given ticker (it is a snapshot). It acts as a size/popularity proxy in cross-sectional regressions, not a time-series signal. The ML model uses a within-week star rank to capture relative popularity without scale distortion.

**Small-sample caveat** — with 6 tickers and ~52 weeks of data, each walk-forward test fold contains roughly 40–60 observations. Model accuracy should be interpreted alongside the binomial p-value reported per fold, not in isolation.
