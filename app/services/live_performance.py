"""Persist closed trades and compute out-of-sample Sharpe from live paper results."""

import math
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.orm import LiveTrade
from app.strategy_params import OOS_MIN_TRADES_FOR_SHARPE, OOS_TARGET_SHARPE, FROZEN_AS_OF


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


async def record_closed_trade(
    db: AsyncSession,
    *,
    strategy_type: str,
    symbol_a: str,
    symbol_b: str | None,
    direction: str,
    entry_price: float,
    exit_price: float,
    shares: float,
    realized_pnl: float,
    exit_reason: str,
    z_entry: float | None,
    z_exit: float | None,
    model_version: str,
) -> LiveTrade:
    row = LiveTrade(
        closed_at=datetime.now(timezone.utc),
        strategy_type=strategy_type,
        symbol_a=symbol_a,
        symbol_b=symbol_b,
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        shares=shares,
        realized_pnl=realized_pnl,
        exit_reason=exit_reason,
        z_entry=_finite_or_none(z_entry),
        z_exit=_finite_or_none(z_exit),
        model_version=model_version,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def get_live_performance(db: AsyncSession, initial_capital: float = 100_000.0) -> dict:
    result = await db.execute(select(LiveTrade).order_by(LiveTrade.closed_at))
    trades = result.scalars().all()

    if not trades:
        return {
            "trade_count": 0,
            "total_pnl": 0.0,
            "win_rate_pct": 0.0,
            "sharpe_ann": None,
            "sharpe_reliable": False,
            "oos_target_sharpe": OOS_TARGET_SHARPE,
            "min_trades_for_sharpe": OOS_MIN_TRADES_FOR_SHARPE,
            "frozen_params_as_of": str(FROZEN_AS_OF),
            "note": "No closed live trades yet — paper trade to build OOS track record.",
        }

    pnls = [float(t.realized_pnl) for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    returns = [p / initial_capital for p in pnls]
    std_r = float(np.std(returns))
    mean_r = float(np.mean(returns))
    sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 0 and len(pnls) >= 2 else None

    by_strategy: dict[str, list[float]] = {}
    for t in trades:
        by_strategy.setdefault(t.strategy_type, []).append(float(t.realized_pnl))

    return {
        "trade_count": len(trades),
        "total_pnl": round(sum(pnls), 2),
        "win_rate_pct": round(wins / len(pnls) * 100, 1),
        "sharpe_ann": round(sharpe, 3) if sharpe is not None else None,
        "sharpe_reliable": len(pnls) >= OOS_MIN_TRADES_FOR_SHARPE,
        "oos_target_sharpe": OOS_TARGET_SHARPE,
        "min_trades_for_sharpe": OOS_MIN_TRADES_FOR_SHARPE,
        "frozen_params_as_of": str(FROZEN_AS_OF),
        "by_strategy": {k: {"trades": len(v), "pnl": round(sum(v), 2)} for k, v in by_strategy.items()},
        "recent_trades": [
            {
                "closed_at": t.closed_at.isoformat(),
                "strategy": t.strategy_type,
                "symbols": f"{t.symbol_a}/{t.symbol_b}" if t.symbol_b else t.symbol_a,
                "pnl": float(t.realized_pnl),
                "reason": t.exit_reason,
            }
            for t in trades[-10:]
        ],
    }


async def count_live_trades(db: AsyncSession) -> int:
    result = await db.execute(select(func.count()).select_from(LiveTrade))
    return result.scalar() or 0
