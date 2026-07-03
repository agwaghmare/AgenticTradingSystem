"""
Discord webhook notifier.

Fires messages when the agent trades, completes a cycle, or halts.
One-way fire-and-forget — never affects trading logic.
"""

import logging

import httpx

from app.config import settings

logger = logging.getLogger("notifier")

DISCORD_MAX = 1900


def _send(content: str | None = None, embeds: list[dict] | None = None):
    if not settings.discord_webhook_url:
        logger.warning("DISCORD_WEBHOOK_URL not set, skipping notification")
        return
    payload: dict = {}
    if content:
        payload["content"] = content[:DISCORD_MAX]
    if embeds:
        payload["embeds"] = embeds[:10]
    if not payload:
        return
    try:
        httpx.post(settings.discord_webhook_url, json=payload, timeout=15)
    except Exception:
        logger.exception("Failed to send Discord notification")


def notify_startup(nav: float, mode: str, pairs: int, singles: int):
    _send(
        embeds=[
            {
                "title": "🤖 Agentic Trading System — Online",
                "description": (
                    f"Paper agent is live and cycling every **{settings.cycle_interval_minutes} min**.\n"
                    f"Mode: **{mode}** · NAV: **${nav:,.2f}**\n"
                    f"Universe: **{pairs}** pairs + **{singles}** single-stock names\n"
                    f"You'll get a Discord update each cycle (trades highlighted)."
                ),
                "color": 5763719,
            }
        ]
    )


def notify_agent_cycle(summary: str, narration: str | None = None, had_trades: bool = False):
    color = 15844367 if had_trades else 3447003  # gold if trades, blue otherwise
    embeds = [
        {
            "title": "📊 Agent Cycle" + (" — TRADE" if had_trades else ""),
            "description": summary[:DISCORD_MAX],
            "color": color,
        }
    ]
    if narration and narration.strip() and "unavailable" not in narration.lower():
        embeds.append(
            {
                "title": "🧠 Agent Narration",
                "description": narration[:DISCORD_MAX],
                "color": 10181046,
            }
        )
    _send(embeds=embeds)


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
    is_single = pair_b == "SINGLE"
    if is_single:
        direction = "LONG" if action == "ENTER_LONG" else "SHORT"
        desc = (
            f"**Single-stock mean reversion** ({direction})\n"
            f"Ticker: `{pair_a}` · Z: `{z_score:.2f}`\n"
            f"Alpaca: **{side_a}** {qty_a:.2f} shares"
        )
    else:
        direction = "LONG spread" if action == "ENTER_LONG" else "SHORT spread"
        desc = (
            f"**Pairs trade** ({direction})\n"
            f"`{pair_a}` / `{pair_b}` · Z: `{z_score:.2f}` · β: `{hedge_ratio:.3f}`\n"
            f"• {side_a} {qty_a:.2f} {pair_a}\n"
            f"• {side_b} {qty_b:.2f} {pair_b}"
        )
    _send(
        embeds=[
            {
                "title": "⚡ Trade Executed",
                "description": desc,
                "color": 3066993,
            }
        ]
    )


def notify_exit(
    pair_a: str,
    pair_b: str,
    reason: str,
    z_score: float | None = None,
    realized_pnl: float | None = None,
):
    is_single = pair_b == "SINGLE"
    sym = f"`{pair_a}`" if is_single else f"`{pair_a}` / `{pair_b}`"
    z_str = f" · Z: `{z_score:.2f}`" if z_score is not None else ""
    pnl_str = ""
    if realized_pnl is not None:
        sign = "+" if realized_pnl >= 0 else ""
        pnl_str = f"\nRealized P&L: **{sign}${realized_pnl:,.2f}**"
    _send(
        embeds=[
            {
                "title": "🔒 Position Closed",
                "description": f"{sym}{z_str}\nReason: {reason}{pnl_str}",
                "color": 15158332 if realized_pnl and realized_pnl < 0 else 5763719,
            }
        ]
    )


def notify_halt(reason: str):
    _send(
        embeds=[
            {
                "title": "⚠️ Agent Halted",
                "description": reason,
                "color": 15158332,
            }
        ]
    )


def notify_error(context: str, detail: str):
    _send(
        embeds=[
            {
                "title": f"❌ Agent Error — {context}",
                "description": detail[:DISCORD_MAX],
                "color": 10038562,
            }
        ]
    )
