"""
CI gate: re-train the walk-forward model on the cached signals fixture and fail the
build if the out-of-sample Sharpe drops below ``config.CI_SHARPE_THRESHOLD``.

This is the production-research safety net. If a refactor silently breaks feature
engineering, the model, or the strategy code, the validation Sharpe will collapse
and CI blocks the merge before the regression reaches main.

Usage (used by .github/workflows/ci.yml):

    python scripts/validate_sharpe.py
    python scripts/validate_sharpe.py --threshold 0.50          # tighter override
    python scripts/validate_sharpe.py --signals path/to.csv     # alternative fixture
    python scripts/validate_sharpe.py --json                    # machine-readable output
"""

import argparse
import json
import logging
import os
import sys

import pandas as pd

import config
from models.classifier import run_walk_forward_classification
from strategy.sharpe import compute_long_short_returns, summarise_strategy

logger = logging.getLogger("commit-alpha.ci")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")


def validate(signals_path: str, threshold: float, n_splits: int = 5) -> dict:
    """Run the model + backtest on a signals fixture and return the tear-sheet."""
    if not os.path.exists(signals_path):
        raise FileNotFoundError(
            f"Signals fixture not found at {signals_path}. "
            f"Commit a cached data/signals.csv so CI has something to validate against."
        )

    signals_df = pd.read_csv(signals_path, parse_dates=["week_start"])

    _, _, oos_preds = run_walk_forward_classification(signals_df, n_splits=n_splits)
    ls_returns = compute_long_short_returns(oos_preds, signal_col="gbm_prob")
    summary = summarise_strategy(ls_returns)

    passed = float(summary["sharpe"]) >= threshold
    return {
        "passed":    passed,
        "threshold": threshold,
        "metrics":   {k: float(v) if k != "n_weeks" else int(v) for k, v in summary.items()},
        "n_signal_rows": int(len(signals_df)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", default=os.path.join("data", "signals.csv"),
                        help="Path to cached signals CSV (default: data/signals.csv).")
    parser.add_argument("--threshold", type=float, default=config.CI_SHARPE_THRESHOLD,
                        help=f"Minimum acceptable Sharpe (default: {config.CI_SHARPE_THRESHOLD}).")
    parser.add_argument("--n-splits", type=int, default=5, help="Walk-forward folds (default 5).")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable.")
    args = parser.parse_args()

    try:
        result = validate(args.signals, args.threshold, n_splits=args.n_splits)
    except Exception as exc:
        logger.error(f"CI gate crashed: {exc}")
        if args.json:
            print(json.dumps({"passed": False, "error": str(exc)}))
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 55)
        print(f"  CI Sharpe Gate — threshold = {args.threshold:.3f}")
        print("=" * 55)
        for k, v in result["metrics"].items():
            print(f"  {k:<14} {v}")
        verdict = "PASS ✓" if result["passed"] else "FAIL ✗"
        print(f"\n  Verdict: {verdict}  "
              f"(observed Sharpe = {result['metrics']['sharpe']:.3f})\n")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
