"""
Agentic cycle reporter — posts structured + LLM-narrated summaries to Discord.

Runs after each trading cycle. Trading logic is unchanged; this is read-only reporting.
"""

import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.orm import DecisionLog, RiskSnapshot
from app.services import notifier, live_performance
from app.services.llm_explainer import narrate_cycle
from app.strategy_params import MODEL_VERSION

logger = logging.getLogger("cycle_reporter")

TRADE_ACTIONS = {"ENTER_LONG", "ENTER_SHORT", "EXIT"}


async def report_cycle(db: AsyncSession, cycle_id: UUID) -> None:
    if not settings.discord_webhook_url:
        return

    dec_result = await db.execute(select(DecisionLog).where(DecisionLog.cycle_id == cycle_id))
    decisions = dec_result.scalars().all()

    risk_result = await db.execute(
        select(RiskSnapshot).where(RiskSnapshot.cycle_id == cycle_id).limit(1)
    )
    risk = risk_result.scalars().first()

    actions = [d.action for d in decisions]
    trades = [d for d in decisions if d.action in TRADE_ACTIONS]
    rejects = [d for d in decisions if d.action == "REJECT"]

    if not settings.discord_notify_every_cycle and not trades:
        return

    nav = float(risk.nav) if risk else None
    regime = risk.regime if risk else "unknown"
    halt = risk.halt_status if risk else "NORMAL"
    dd = float(risk.drawdown_mtd_pct) if risk and risk.drawdown_mtd_pct is not None else 0.0

    lines = [
        f"**Cycle** `{str(cycle_id)[:8]}…` · model `{MODEL_VERSION}`",
        f"Regime: `{regime}` · Halt: `{halt}` · MTD DD: `{dd:.1%}`",
    ]
    if nav is not None:
        lines.append(f"NAV: `${nav:,.2f}`")

    if trades:
        lines.append(f"\n**Trades ({len(trades)})**")
        for d in trades:
            sym = f"{d.pair_a}/{d.pair_b}" if d.pair_b != "SINGLE" else f"{d.pair_a} (single)"
            z = f" z={float(d.z_score):.2f}" if d.z_score is not None else ""
            lines.append(f"• **{d.action}** `{sym}`{z}")
            if d.reasoning and "realized_pnl" in (d.reasoning or ""):
                lines.append(f"  _{d.reasoning}_")
    else:
        lines.append("\n_No entries or exits this cycle._")

    if rejects and settings.discord_include_reject_summary:
        by_reason: dict[str, int] = {}
        for d in rejects:
            key = (d.rejection_reason or d.reasoning or "unknown")[:60]
            by_reason[key] = by_reason.get(key, 0) + 1
        top = sorted(by_reason.items(), key=lambda x: -x[1])[:3]
        lines.append(f"\n**Top rejections** ({len(rejects)} total)")
        for reason, count in top:
            lines.append(f"• ({count}×) {reason}")

    perf = await live_performance.get_live_performance(db)
    if perf["trade_count"] > 0:
        sharpe = perf.get("sharpe_ann")
        sharpe_str = f"{sharpe}" if sharpe is not None else "n/a"
        reliable = "✓" if perf.get("sharpe_reliable") else f"need {perf['min_trades_for_sharpe']}+ trades"
        lines.append(
            f"\n**OOS paper track:** {perf['trade_count']} closed · "
            f"P&L `${perf['total_pnl']:,.2f}` · Sharpe `{sharpe_str}` ({reliable})"
        )

    summary = "\n".join(lines)
    narration = None
    if settings.enable_agentic_narration and (trades or settings.discord_notify_every_cycle):
        decision_dicts = [_decision_to_dict(d) for d in decisions]
        risk_dict = _risk_to_dict(risk) if risk else {}
        try:
            narration = narrate_cycle(decision_dicts, risk_dict)
        except Exception:
            logger.exception("LLM narration failed")

    notifier.notify_agent_cycle(summary, narration=narration, had_trades=bool(trades))


def _decision_to_dict(d: DecisionLog) -> dict:
    return {
        "pair_a": d.pair_a,
        "pair_b": d.pair_b,
        "action": d.action,
        "z_score": float(d.z_score) if d.z_score is not None else None,
        "regime": d.regime,
        "reasoning": d.reasoning,
        "rejection_reason": d.rejection_reason,
    }


def _risk_to_dict(r: RiskSnapshot) -> dict:
    return {
        "nav": float(r.nav),
        "gross_leverage": float(r.gross_leverage) if r.gross_leverage else None,
        "net_leverage": float(r.net_leverage) if r.net_leverage else None,
        "drawdown_mtd_pct": float(r.drawdown_mtd_pct) if r.drawdown_mtd_pct else None,
        "regime": r.regime,
        "halt_status": r.halt_status,
        "open_positions": r.open_positions,
    }
