"""
Discord webhook notifier.

Fires a message whenever the agent executes a trade on Alpaca, so you can
see (and optionally manually mirror) the same signal on Robinhood or
anywhere else. This module NEVER places orders and has zero effect on
trading logic — it's a one-way, fire-and-forget notification, called after
a trade has already been decided and sent to Alpaca.

Setup:
1. In Discord: Server Settings -> Integrations -> Webhooks -> New Webhook
2. Copy the webhook URL, put it in .env as DISCORD_WEBHOOK_URL
3. That's it — no bot, no OAuth, just a POST to a URL
"""

import logging

import httpx

from app.config import settings

logger = logging.getLogger("notifier")


def _send(content: str):
    if not settings.discord_webhook_url:
        logger.warning("DISCORD_WEBHOOK_URL not set, skipping notification")
        return
    try:
        httpx.post(settings.discord_webhook_url, json={"content": content}, timeout=10)
    except Exception:
        logger.exception("Failed to send Discord notification")


def notify_trade_signal(
    pair_a: str,
    pair_b: str,
    action: str,
    side_a: str,
    side_b: str,
    qty_a: float,
    qty_b: float,
    z_score: float,
    hedge_ratio: float,
):
    """Called right after the agent places an order on Alpaca. Informational only."""
    direction = "LONG spread" if action == "ENTER_LONG" else "SHORT spread"
    content = (
        f"**Trade Signal Fired** ({direction})\n"
        f"Pair: `{pair_a}` / `{pair_b}`\n"
        f"Z-score: `{z_score:.2f}` | Hedge ratio: `{hedge_ratio:.3f}`\n"
        f"Executed on Alpaca:\n"
        f"  • {side_a} {qty_a:.2f} {pair_a}\n"
        f"  • {side_b} {qty_b:.2f} {pair_b}\n"
        f"_If mirroring on Robinhood, match side/ratio above — not exact qty._"
    )
    _send(content)


def notify_exit(pair_a: str, pair_b: str, reason: str, z_score: float | None = None):
    z_str = f" | Z-score: `{z_score:.2f}`" if z_score is not None else ""
    content = f"**Position Closed**\nPair: `{pair_a}` / `{pair_b}`{z_str}\nReason: {reason}"
    _send(content)


def notify_halt(reason: str):
    content = f"**⚠️ Agent Halted**\n{reason}"
    _send(content)
