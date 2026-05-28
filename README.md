# commit-alpha

A quantitative research pipeline that turns GitHub developer activity into stock return alpha signals. Built in three tiers: data ingestion + correlation study (Tier 1), walk-forward ML classification + long/short backtest (Tier 2), and production-grade research controls — alpha-decay analysis, permutation-test overfit check, FastAPI service, and a CI Sharpe gate (Tier 3).

## What it does

1. Fetches weekly GitHub metrics (commit count, contributor count, star count) for a configurable list of public tech repos mapped to listed equities
2. Pulls historical daily OHLCV prices via yfinance
3. Computes forward log returns at 1-, 3-, 5-, 10-, and 20-day horizons
4. Joins GitHub signals to stock return dates and runs a Pearson + Spearman correlation study
5. Trains logistic regression and gradient boosting classifiers under strict walk-forward validation (no look-ahead bias) to predict weekly return direction
6. Simulates a long/short portfolio (long top-2 tickers by model score, short bottom-2) and reports annualised Sharpe ratio, max drawdown, and win rate
7. **Alpha-decay analysis** — measures how quickly the signal's information coefficient (IC) and Sharpe degrade as the forward horizon stretches from 1 to 20 trading days
8. **Permutation overfit check** — shuffles forward returns 1 000× within each week and computes a p-value for the observed Sharpe against the null of zero predictive power
9. **FastAPI service** + **GitHub Actions Sharpe gate** — exposes the trained signal as REST endpoints; CI blocks merges that drop the validation-set Sharpe below `config.CI_SHARPE_THRESHOLD`

## Project structure

```
commit-alpha/
├── config.py                  # tickers, repo map, date range, return horizons, CI threshold
├── main.py                    # 10-step pipeline entry point
├── requirements.txt
├── data/
│   ├── github_fetcher.py      # PyGitHub stats API → weekly signals
│   ├── price_fetcher.py       # yfinance → daily OHLCV
│   ├── signals.csv            # output: joined signals + returns
│   └── oos_predictions.csv    # output: walk-forward OOS model scores
├── features/
│   └── returns.py             # forward log return computation
├── models/
│   ├── walk_forward.py        # TimeSeriesSplit on unique weeks
│   └── classifier.py          # LR + GBM with per-fold StandardScaler
├── strategy/
│   └── sharpe.py              # long/short backtest, Sharpe, drawdown
├── analysis/
│   ├── correlation.py         # Pearson/Spearman heatmap
│   ├── alpha_decay.py         # IC + Sharpe vs forward horizon  (Tier 3)
│   └── permutation_test.py    # 1 000-shuffle null distribution  (Tier 3)
├── api/
│   └── main.py                # FastAPI service (signals, predictions, backtest) (Tier 3)
├── scripts/
│   └── validate_sharpe.py     # CI gate — fails build below Sharpe threshold (Tier 3)
├── tests/
│   ├── test_pipeline_smoke.py
│   └── test_api.py
├── .github/workflows/ci.yml   # pytest + Sharpe gate on every PR  (Tier 3)
└── outputs/
    ├── correlation_heatmap.png
    ├── strategy_returns.png
    ├── alpha_decay.png        # (Tier 3)
    └── permutation_test.png   # (Tier 3)
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
RETURNS_HORIZONS = [1, 3, 5, 10, 20]   # trading days — extra horizons drive the alpha-decay curve
PERMUTATION_N    = 1000                # null-distribution shuffles for the overfit check
CI_SHARPE_THRESHOLD = 0.30             # GitHub Actions fails below this OOS Sharpe
```

The GitHub stats API covers the last ~52 weeks from the time of the call. Keep `START_DATE` within that window.

## Outputs

| File | Description |
|---|---|
| `data/signals.csv` | Joined panel: (ticker, week) × (signals + forward returns) |
| `data/oos_predictions.csv` | Walk-forward out-of-sample model scores (consumed by FastAPI + CI) |
| `outputs/correlation_heatmap.png` | Pearson and Spearman heatmaps, signals vs return horizons |
| `outputs/strategy_returns.png` | Cumulative long/short returns with annotated Sharpe ratio |
| `outputs/alpha_decay.png` | IC and Sharpe at each forward horizon — how fast the alpha decays |
| `outputs/permutation_test.png` | Null Sharpe distribution from 1 000 within-week return shuffles |

## Tier 3 — production-grade research pipeline

### Alpha decay
`analysis/alpha_decay.py` answers the question every quant interviewer asks — *"how long does your signal live?"* — by re-scoring the OOS predictions against forward returns at 1-, 3-, 5-, 10-, and 20-day horizons. It reports the pooled Spearman IC, the mean weekly rank-IC, the IC information ratio, the directional hit rate, and the long/short Sharpe at each horizon, then renders a two-panel decay chart.

### Permutation overfit check
`analysis/permutation_test.py` runs a within-week return shuffle 1 000× (configurable via `config.PERMUTATION_N`), recomputes the long/short Sharpe each time, and reports the p-value of the observed Sharpe against that null distribution. This is the kind of risk/overfit control Citadel and Two Sigma describe explicitly; it costs ~30 s of CPU and almost no other candidates ship it.

### FastAPI service
```bash
uvicorn api.main:app --reload --port 8000
# then visit http://localhost:8000/docs for the interactive Swagger UI
```

| Endpoint | Returns |
|---|---|
| `GET  /health` | Liveness + counts of cached signals/predictions |
| `GET  /signals/latest` | Most recent week of joined signals (optional `?ticker=`) |
| `GET  /predictions` | OOS model scores (optional `?ticker=`, `?limit=`) |
| `GET  /predictions/latest` | Latest-week ranking + recommended long/short basket |
| `GET  /backtest` | Strategy tear-sheet (Sharpe, drawdown, win-rate, …) |
| `GET  /alpha-decay` | IC + Sharpe per horizon, JSON form of the decay chart |
| `POST /retrain` | Re-runs `python main.py` (needs `GITHUB_TOKEN`) |

All read endpoints serve the cached artefacts produced by `python main.py`, so per-request latency stays sub-100 ms.

### GitHub Actions Sharpe gate
`.github/workflows/ci.yml` runs on every push and PR:

1. `pytest tests/` — synthetic-fixture smoke tests for the model, strategy, alpha-decay, permutation test, and FastAPI routes (no GitHub token, no network).
2. `python scripts/validate_sharpe.py` — re-trains the walk-forward model on `data/signals.csv` (committed to the repo as a fixture) and **fails the build** if the OOS Sharpe drops below `config.CI_SHARPE_THRESHOLD`. This catches silent regressions in feature engineering, the model, or the strategy code before they reach `main`.

Adjust the threshold in `config.py` as your signal improves — start conservatively and ratchet it up alongside genuine, reproducible gains.

## Design notes

**No look-ahead bias** — the walk-forward split (`TimeSeriesSplit`) is applied to the sorted sequence of unique week timestamps, not raw row indices. This ensures all tickers for a given week land in the same fold and the model is never trained on data from after its prediction date. The `StandardScaler` is fit on the training fold only and applied to the test fold.

**Why log returns** — log returns are time-additive and approximately normally distributed, which satisfies the assumptions of both the correlation tests and the Sharpe ratio calculation.

**Star count as a cross-sectional factor** — star count does not vary week-to-week for a given ticker (it is a snapshot). It acts as a size/popularity proxy in cross-sectional regressions, not a time-series signal. The ML model uses a within-week star rank to capture relative popularity without scale distortion.

**Small-sample caveat** — with 6 tickers and ~52 weeks of data, each walk-forward test fold contains roughly 40–60 observations. Model accuracy should be interpreted alongside the binomial p-value reported per fold, not in isolation.
