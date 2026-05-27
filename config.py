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
RETURNS_HORIZONS = [1, 5, 20]
