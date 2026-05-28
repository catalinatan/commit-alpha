"""Central configuration for the commit-alpha pipeline. Edit here to swap tickers, repos, or date range."""
import os

# ── Tickers and their representative public GitHub repos ──────────────────────
TICKERS = ["MSFT", "GOOGL", "META", "AAPL", "NVDA", "AMZN"]

TICKER_TO_REPO = {
    "MSFT": "microsoft/vscode",          # Microsoft's highest-activity OSS project
    "GOOGL": "google/jax",               # Google-owned ML framework, very active
    "META": "facebook/react",            # Meta's flagship OSS project
    "AAPL": "apple/swift",               # Apple's primary OSS language project
    "NVDA": "NVIDIA/TensorRT-LLM",       # NVIDIA's high-growth AI inference repo
    "AMZN": "aws/aws-cli",               # AWS's core CLI tool
}

# ── Date range ────────────────────────────────────────────────────────────────
# GitHub's stats API (get_stats_commit_activity / get_stats_contributors) returns
# the last ~52 weeks from the time of the call. Set dates within that window.
# Price data will be extended by 25 trading days automatically to cover forward returns.
START_DATE = "2024-06-01"
END_DATE   = "2025-05-01"

# ── GitHub auth ────────────────────────────────────────────────────────────────
# Unauthenticated limit: 60 req/hr — far too low for this pipeline.
# Set via: export GITHUB_TOKEN=ghp_...
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# ── Return horizons (trading days) ────────────────────────────────────────────
# 5d is the ML target; 1d / 3d / 10d / 20d feed the alpha-decay analysis.
RETURNS_HORIZONS = [1, 3, 5, 10, 20]

# ── Tier 3 ─ overfit / permutation test ──────────────────────────────────────
PERMUTATION_N = 1000        # standard quant-research default
PERMUTATION_SEED = 42

# ── Tier 3 ─ CI gate ─────────────────────────────────────────────────────────
# The CI workflow fails if the OOS Sharpe falls below this threshold. Set it
# conservatively so genuine improvements pass and regressions get caught.
CI_SHARPE_THRESHOLD = 0.30
