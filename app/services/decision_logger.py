from sqlalchemy.ext.asyncio import AsyncSession
from app.models.orm import DecisionLog
from app.models.schemas import DecisionLogSchema


async def log_decision(db: AsyncSession, decision: DecisionLogSchema) -> DecisionLog:
    entry = DecisionLog(**decision.model_dump())
    db.add(entry)
    await db.commit()
    await db.refresh(entry)
    return entry
