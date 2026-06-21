from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.orm import RiskSnapshot


async def calc_drawdown_mtd(db: AsyncSession, current_nav: float) -> float:
    """Peak-to-trough drawdown for the current calendar month."""
    if current_nav <= 0:
        return 0.0

    start_of_month = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    result = await db.execute(
        select(func.max(RiskSnapshot.nav)).where(RiskSnapshot.timestamp >= start_of_month)
    )
    peak_nav = result.scalar()

    if peak_nav is None:
        return 0.0

    peak_nav = float(peak_nav)
    if peak_nav <= 0:
        return 0.0

    return (current_nav - peak_nav) / peak_nav
