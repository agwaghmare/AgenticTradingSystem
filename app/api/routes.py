from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.db.session import get_db
from app.models.orm import DecisionLog, RiskSnapshot, OrderLog
from app.services.llm_explainer import narrate_cycle, review_decisions

router = APIRouter(prefix="/api", tags=["monitoring"])


@router.get("/decisions/recent")
async def recent_decisions(limit: int = 50, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(DecisionLog).order_by(desc(DecisionLog.timestamp)).limit(limit))
    return result.scalars().all()


@router.get("/risk/latest")
async def latest_risk(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(RiskSnapshot).order_by(desc(RiskSnapshot.timestamp)).limit(1))
    return result.scalars().first()

@router.get("/orders/recent")
async def recent_orders(limit: int = 50, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(OrderLog).order_by(desc(OrderLog.timestamp)).limit(limit))
    return result.scalars().all()


@router.get("/decisions/narrate-latest-cycle")
async def narrate_latest_cycle(db: AsyncSession = Depends(get_db)):
    """LLM-generated plain-English summary of the most recent cycle. Read-only, no trading impact."""
    latest = await db.execute(select(DecisionLog).order_by(desc(DecisionLog.timestamp)).limit(1))
    latest_row = latest.scalars().first()
    if not latest_row:
        return {"summary": "No decisions logged yet."}

    cycle_result = await db.execute(
        select(DecisionLog).where(DecisionLog.cycle_id == latest_row.cycle_id)
    )
    decisions = [d.__dict__ for d in cycle_result.scalars().all()]
    for d in decisions:
        d.pop("_sa_instance_state", None)

    risk_result = await db.execute(
        select(RiskSnapshot).where(RiskSnapshot.cycle_id == latest_row.cycle_id).limit(1)
    )
    risk_row = risk_result.scalars().first()
    risk_dict = risk_row.__dict__ if risk_row else {}
    risk_dict.pop("_sa_instance_state", None)

    summary = narrate_cycle(decisions, risk_dict)
    return {"cycle_id": str(latest_row.cycle_id), "summary": summary}


@router.get("/decisions/review-recent")
async def review_recent(limit: int = 200, db: AsyncSession = Depends(get_db)):
    """LLM-generated anomaly review over recent decisions. Read-only, no trading impact."""
    result = await db.execute(select(DecisionLog).order_by(desc(DecisionLog.timestamp)).limit(limit))
    decisions = [d.__dict__ for d in result.scalars().all()]
    for d in decisions:
        d.pop("_sa_instance_state", None)

    if not decisions:
        return {"review": "No decisions logged yet."}

    review = review_decisions(decisions, window_label=f"last {limit} decisions")
    return {"review": review}


@router.get("/health")
async def health():
    return {"status": "ok"}

