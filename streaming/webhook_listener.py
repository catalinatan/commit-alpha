"""
GitHub webhook listener — produces normalised activity events to Kafka.

Receives raw GitHub webhook deliveries on POST /webhook, verifies the HMAC-SHA256
signature, normalises ``push`` / ``watch`` / ``pull_request`` events to the
project schema, and publishes one JSON message per event to the
``github.activity`` Kafka topic.

Message schema produced to Kafka
--------------------------------
    {
      "ticker":        "MSFT",                       # mapped from repo
      "repo":          "microsoft/vscode",           # GitHub repo full name
      "event_type":    "push" | "watch" | "pull_request",
      "actor_count":   12,                           # commits in push, else 1
      "timestamp_utc": "2026-05-29T14:32:11+00:00"   # event time (UTC, ISO-8601)
    }

The Kafka message key is the ticker, so all events for the same equity land in the
same partition and the downstream signal-consumer sees them in order per ticker.

Configure GitHub:
    Repo → Settings → Webhooks → Add webhook
      Payload URL:  https://<your-host>:8001/webhook
      Content type: application/json
      Secret:       same value as WEBHOOK_SECRET below
      Events:       Pushes, Stars, Pull requests
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import uvicorn
from confluent_kafka import KafkaException, Producer
from fastapi import FastAPI, Header, HTTPException, Request, status

import config

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("webhook-listener")

# ── Configuration ────────────────────────────────────────────────────────────
WEBHOOK_SECRET   = os.getenv("WEBHOOK_SECRET", "")
KAFKA_BOOTSTRAP  = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC      = os.getenv("KAFKA_TOPIC_GITHUB_ACTIVITY", "github.activity")
LISTEN_HOST      = os.getenv("WEBHOOK_HOST", "0.0.0.0")
LISTEN_PORT      = int(os.getenv("WEBHOOK_PORT", "8001"))

# Invert TICKER_TO_REPO so we can resolve incoming repo → ticker in O(1).
# Repo names normalised to lower-case for case-insensitive matching.
REPO_TO_TICKER: dict[str, str] = {
    repo.lower(): ticker for ticker, repo in config.TICKER_TO_REPO.items()
}

SUPPORTED_EVENTS = {"push", "watch", "pull_request"}


# ── Kafka producer (one instance, reused) ────────────────────────────────────
def _delivery_report(err, msg) -> None:
    """Called once per message to surface async produce failures."""
    if err is not None:
        logger.error(f"Kafka delivery failed: {err} (topic={msg.topic()})")
    else:
        logger.debug(
            f"Kafka delivered → {msg.topic()}[{msg.partition()}]@{msg.offset()}"
        )


producer = Producer({
    "bootstrap.servers": KAFKA_BOOTSTRAP,
    "client.id":         "webhook-listener",
    # Wait for leader-acked persistence — safest for low-volume webhook traffic.
    "acks":              "all",
    "enable.idempotence": True,
    "linger.ms":         5,
})


# ── Signature verification ───────────────────────────────────────────────────
def _verify_signature(secret: str, body: bytes, signature_header: str | None) -> bool:
    """
    Constant-time HMAC-SHA256 check on the raw request body.

    GitHub sends ``X-Hub-Signature-256: sha256=<hexdigest>``. Returns True iff
    the digest matches. Constant-time comparison via ``hmac.compare_digest``
    avoids leaking the secret through response-time differences.
    """
    if not secret:
        # Dev escape hatch — surfaced loudly so prod misconfig is obvious.
        logger.warning(
            "WEBHOOK_SECRET is unset — accepting all deliveries unsigned. "
            "Set WEBHOOK_SECRET before exposing this service publicly."
        )
        return True
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    sent_digest = signature_header.removeprefix("sha256=")
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(sent_digest, expected)


# ── Event normalisation ──────────────────────────────────────────────────────
def _extract_timestamp(event_type: str, payload: dict[str, Any]) -> str:
    """
    Best-effort event timestamp in UTC ISO-8601.

    ``push`` payloads carry ``head_commit.timestamp``; ``pull_request`` payloads
    carry ``pull_request.updated_at``/``created_at``. ``watch`` payloads have
    no event time, so we fall back to server-side ``now()``.
    """
    if event_type == "push":
        head = payload.get("head_commit") or {}
        if head.get("timestamp"):
            return head["timestamp"]
    if event_type == "pull_request":
        pr = payload.get("pull_request") or {}
        ts = pr.get("updated_at") or pr.get("created_at")
        if ts:
            return ts
    return datetime.now(timezone.utc).isoformat()


def _extract_actor_count(event_type: str, payload: dict[str, Any]) -> int:
    """
    Per the spec: commits-in-push for ``push``, otherwise 1 (one PR, one star).
    """
    if event_type == "push":
        return len(payload.get("commits") or [])
    return 1


def _build_message(
    event_type: str,
    payload: dict[str, Any],
    ticker: str,
    repo_full_name: str,
) -> dict[str, Any]:
    return {
        "ticker":        ticker,
        "repo":          repo_full_name,
        "event_type":    event_type,
        "actor_count":   _extract_actor_count(event_type, payload),
        "timestamp_utc": _extract_timestamp(event_type, payload),
    }


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="commit-alpha webhook-listener",
    description="Normalises GitHub webhook events and forwards them to Kafka.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status":           "ok",
        "kafka_topic":      KAFKA_TOPIC,
        "kafka_bootstrap":  KAFKA_BOOTSTRAP,
        "mapped_tickers":   len(REPO_TO_TICKER),
        "secret_configured": bool(WEBHOOK_SECRET),
    }


@app.post("/webhook")
async def webhook(
    request: Request,
    x_hub_signature_256: str | None = Header(default=None),
    x_github_event:      str | None = Header(default=None),
    x_github_delivery:   str | None = Header(default=None),
) -> dict[str, Any]:
    """
    Receive one GitHub webhook delivery.

    Flow:
      1. Read the raw body (needed for HMAC — JSON re-encoded body would differ).
      2. Verify ``X-Hub-Signature-256``; reject with 400 on mismatch.
      3. Ignore event types outside {push, watch, pull_request} with 200 + skip.
      4. Resolve repo → ticker via ``config.TICKER_TO_REPO``; silently ack 200
         and skip if the repo is not tracked by this project.
      5. Normalise the payload to the project schema and produce to Kafka.
    """
    raw_body = await request.body()

    # ── 1+2. Signature check ────────────────────────────────────────────────
    if not _verify_signature(WEBHOOK_SECRET, raw_body, x_hub_signature_256):
        # Spec: 400 on signature failure (not 401/403).
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid X-Hub-Signature-256",
        )

    # ── 3. Event-type filter ────────────────────────────────────────────────
    event_type = (x_github_event or "").lower()
    if event_type not in SUPPORTED_EVENTS:
        logger.info(f"Ignored event type {event_type!r} (delivery={x_github_delivery})")
        return {"status": "ignored", "reason": f"unsupported event type {event_type!r}"}

    # ── Parse JSON only after signature verification ────────────────────────
    try:
        payload = json.loads(raw_body) if raw_body else {}
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="payload is not valid JSON",
        )

    repo = payload.get("repository") or {}
    repo_full_name = repo.get("full_name")
    if not repo_full_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="payload missing repository.full_name",
        )

    # ── 4. Ticker resolution (silently skip unknown repos) ──────────────────
    ticker = REPO_TO_TICKER.get(repo_full_name.lower())
    if ticker is None:
        logger.info(f"Skipped untracked repo {repo_full_name!r} (event={event_type})")
        return {"status": "skipped", "reason": "repo not in TICKER_TO_REPO"}

    # ── 5. Publish to Kafka ─────────────────────────────────────────────────
    message = _build_message(event_type, payload, ticker, repo_full_name)
    try:
        producer.produce(
            topic=KAFKA_TOPIC,
            key=ticker.encode("utf-8"),       # partition-by-ticker → per-ticker ordering
            value=json.dumps(message).encode("utf-8"),
            on_delivery=_delivery_report,
        )
        # Non-blocking poll to serve delivery callbacks queued from earlier produces.
        producer.poll(0)
    except (BufferError, KafkaException) as exc:
        logger.error(f"Producer rejected message for {ticker}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="kafka producer unavailable",
        )

    logger.info(
        f"Published {event_type:<13} {ticker} ({repo_full_name}) "
        f"actor_count={message['actor_count']}"
    )
    return {"status": "ok", "ticker": ticker, "event_type": event_type}


@app.on_event("shutdown")
def _flush_producer_on_shutdown() -> None:
    """Make sure in-flight messages reach the broker before the process exits."""
    logger.info("Flushing Kafka producer…")
    remaining = producer.flush(timeout=5.0)
    if remaining:
        logger.warning(f"{remaining} messages still in queue after flush timeout")


if __name__ == "__main__":
    logger.info(
        f"Starting webhook-listener on {LISTEN_HOST}:{LISTEN_PORT} "
        f"→ kafka={KAFKA_BOOTSTRAP} topic={KAFKA_TOPIC} "
        f"tracked_repos={len(REPO_TO_TICKER)}"
    )
    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT, log_level="info")
