"""
Fetches weekly GitHub activity metrics for a list of repos using PyGitHub.

Signals produced (weekly granularity, covering the last ~52 weeks from call time):
  commit_count      — total commits merged in the week
  contributor_count — unique authors with ≥1 commit that week
  star_count        — snapshot of total stars at fetch time (static per repo;
                      used as a cross-sectional size/popularity proxy)

Rate limits:
  Authenticated:   5 000 req/hr   ← required for this pipeline
  Unauthenticated:    60 req/hr   ← will exhaust quickly; always set GITHUB_TOKEN
"""

import time
import logging
import pandas as pd
from github import Github, GithubException

logger = logging.getLogger(__name__)


def _unix_to_ts(unix: int) -> pd.Timestamp:
    """Convert GitHub's Unix week timestamp to a tz-naive UTC date."""
    return pd.Timestamp(unix, unit="s", tz="UTC").normalize().tz_localize(None)


def _fetch_weekly_stats(repo, max_retries: int = 5) -> tuple:
    """
    Retrieve commit activity and contributor stats, retrying on 202 (GitHub computes
    these asynchronously and may not have them ready on the first request).

    Returns (commit_activity, contributor_stats) — either may be None on failure.
    """
    commit_activity = contributor_stats = None
    for attempt in range(max_retries):
        commit_activity = repo.get_stats_commit_activity()
        contributor_stats = repo.get_stats_contributors()
        if commit_activity is not None and contributor_stats is not None:
            return commit_activity, contributor_stats
        wait = 2 ** attempt
        logger.info(
            f"Stats not ready for {repo.full_name} (attempt {attempt + 1}/{max_retries}), "
            f"retrying in {wait}s…"
        )
        time.sleep(wait)
    return commit_activity, contributor_stats


def fetch_repo_weekly_metrics(
    repo_full_name: str,
    github_client: Github,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Fetch weekly GitHub metrics for a single repo, filtered to [start_date, end_date].

    Args:
        repo_full_name: e.g. ``"microsoft/vscode"``
        github_client:  authenticated ``Github`` instance
        start_date:     inclusive lower bound (tz-naive)
        end_date:       inclusive upper bound (tz-naive)

    Returns:
        DataFrame with columns [week_start, commit_count, contributor_count, star_count].
        Empty DataFrame if GitHub stats are unavailable after retries.
    """
    try:
        repo = github_client.get_repo(repo_full_name)
    except GithubException as exc:
        logger.error(f"Could not retrieve repo {repo_full_name}: {exc}")
        return pd.DataFrame(columns=["week_start", "commit_count", "contributor_count", "star_count"])

    star_count = repo.stargazers_count

    commit_activity, contributor_stats = _fetch_weekly_stats(repo)

    if commit_activity is None or contributor_stats is None:
        logger.warning(f"Stats permanently unavailable for {repo_full_name}; skipping.")
        return pd.DataFrame(columns=["week_start", "commit_count", "contributor_count", "star_count"])

    # ── Weekly commit totals ──────────────────────────────────────────────────
    weekly_commits: dict[pd.Timestamp, int] = {
        _unix_to_ts(w.week): w.total for w in commit_activity
    }

    # ── Weekly unique contributor counts ─────────────────────────────────────
    # get_stats_contributors() returns one StatsContributor per author, each
    # carrying a list of weekly (w, a, d, c) tuples. We aggregate across authors.
    weekly_contributors: dict[pd.Timestamp, int] = {}
    for contributor in contributor_stats:
        for week_stat in contributor.weeks:
            if week_stat.c > 0:
                ts = _unix_to_ts(week_stat.w)
                weekly_contributors[ts] = weekly_contributors.get(ts, 0) + 1

    # ── Merge into a single DataFrame ─────────────────────────────────────────
    all_weeks = sorted(set(weekly_commits) | set(weekly_contributors))
    records = [
        {
            "week_start": w,
            "commit_count": weekly_commits.get(w, 0),
            "contributor_count": weekly_contributors.get(w, 0),
            "star_count": star_count,
        }
        for w in all_weeks
    ]

    df = pd.DataFrame(records)
    df["week_start"] = pd.to_datetime(df["week_start"])

    mask = (df["week_start"] >= start_date) & (df["week_start"] <= end_date)
    return df.loc[mask].reset_index(drop=True)


def fetch_all_github_signals(
    ticker_to_repo: dict,
    github_token: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch weekly GitHub signals for every ticker in ``ticker_to_repo``.

    Args:
        ticker_to_repo: mapping from ticker symbol to ``"owner/repo"`` string
        github_token:   personal access token (required for 5 000 req/hr limit)
        start_date:     ISO date string, e.g. ``"2024-06-01"``
        end_date:       ISO date string, e.g. ``"2025-05-01"``

    Returns:
        Long-format DataFrame with columns
        [ticker, week_start, commit_count, contributor_count, star_count].
    """
    if not github_token:
        logger.warning(
            "GITHUB_TOKEN is not set — unauthenticated rate limit is 60 req/hr. "
            "The pipeline will likely hit rate limits. Set GITHUB_TOKEN in your environment."
        )

    g = Github(github_token) if github_token else Github()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    frames = []
    for ticker, repo_name in ticker_to_repo.items():
        logger.info(f"  Fetching GitHub metrics: {ticker} → {repo_name}")
        df = fetch_repo_weekly_metrics(repo_name, g, start, end)
        if df.empty:
            logger.warning(f"  No data returned for {ticker}; it will be absent from signals.")
            continue
        df.insert(0, "ticker", ticker)
        frames.append(df)
        logger.info(f"  {ticker}: {len(df)} weeks")

    if not frames:
        raise RuntimeError("GitHub fetcher returned no data for any ticker.")

    return pd.concat(frames, ignore_index=True)
