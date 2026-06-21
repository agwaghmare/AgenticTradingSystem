import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.orm import RiskSnapshot


async def log_risk_snapshot(
    db: AsyncSession,
    *,
    cycle_id: uuid.UUID,
    nav: float,
    gross_leverage: float | None,
    net_leverage: float | None,
    daily_var_95: float | None,
    drawdown_mtd_pct: float,
    open_positions: int,
    regime: str,
    halt_status: str,
) -> RiskSnapshot:
    snapshot = RiskSnapshot(
        cycle_id=cycle_id,
        timestamp=datetime.now(timezone.utc),
        nav=nav,
        gross_leverage=gross_leverage,
        net_leverage=net_leverage,
        daily_var_95=daily_var_95,
        drawdown_mtd_pct=drawdown_mtd_pct,
        open_positions=open_positions,
        regime=regime,
        halt_status=halt_status,
    )
    db.add(snapshot)
    await db.commit()
    await db.refresh(snapshot)
    return snapshot
