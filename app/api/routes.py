from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func

from app.db.session import get_db
from app.models.orm import DecisionLog, RiskSnapshot, OrderLog
from app.services.llm_explainer import narrate_cycle, review_decisions
from app.services.broker import BrokerClient

from app.pairs_universe import CANDIDATE_PAIRS

router = APIRouter(prefix="/api", tags=["monitoring"])

_broker = BrokerClient()


@router.get("/", include_in_schema=False, response_class=HTMLResponse)
async def dashboard(db: AsyncSession = Depends(get_db)):
    nav = _broker.get_account_nav()
    mode = "Alpaca Paper" if _broker.client else "Simulation"

    risk_result = await db.execute(select(RiskSnapshot).order_by(desc(RiskSnapshot.timestamp)).limit(1))
    risk = risk_result.scalars().first()

    count_result = await db.execute(select(func.count()).select_from(DecisionLog))
    decision_count = count_result.scalar() or 0

    dec_result = await db.execute(select(DecisionLog).order_by(desc(DecisionLog.timestamp)).limit(20))
    decisions = dec_result.scalars().all()

    rows = ""
    for d in decisions:
        ts = d.timestamp.strftime("%Y-%m-%d %H:%M UTC") if d.timestamp else "—"
        z = f"{float(d.z_score):.2f}" if d.z_score is not None else "—"
        reason = d.reasoning or "—"
        if len(reason) > 80:
            reason = reason[:80] + "…"
        rows += (
            f"<tr><td>{ts}</td><td>{d.pair_a}/{d.pair_b}</td>"
            f"<td><b>{d.action}</b></td><td>{d.regime or '—'}</td>"
            f"<td>{z}</td><td>{reason}</td></tr>"
        )

    risk_block = "No risk snapshots yet."
    if risk:
        risk_block = (
            f"NAV ${float(risk.nav):,.2f} · Gross {float(risk.gross_leverage or 0):.2f}x · "
            f"Net {float(risk.net_leverage or 0):.2f}x · VaR95 ${float(risk.daily_var_95 or 0):,.2f} · "
            f"MTD DD {float(risk.drawdown_mtd_pct or 0):.1%} · Regime {risk.regime} · {risk.halt_status}"
        )

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><meta http-equiv="refresh" content="30">
<title>AgenticTradingSystem</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #0f1117; color: #e6e6e6; }}
  h1 {{ color: #6ee7b7; }} .card {{ background: #1a1d27; border-radius: 8px; padding: 1rem 1.5rem; margin: 1rem 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th, td {{ text-align: left; padding: 0.5rem; border-bottom: 1px solid #333; }}
  th {{ color: #94a3b8; }} a {{ color: #6ee7b7; }}
  .badge {{ background: #065f46; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; }}
</style></head><body>
<h1>AgenticTradingSystem <span class="badge">{mode}</span></h1>
<div class="card">
  <strong>Account NAV:</strong> ${nav:,.2f} &nbsp;|&nbsp;
  <strong>Decisions logged:</strong> {decision_count} &nbsp;|&nbsp;
  <strong>Cycle interval:</strong> 5 min &nbsp;|&nbsp;
  <a href="/docs">API Docs</a> · <a href="/api/decisions/recent">JSON decisions</a> ·
  <a href="/api/risk/latest">JSON risk</a>
</div>
<div class="card"><strong>Latest risk snapshot</strong><br>{risk_block}</div>
<div class="card"><strong>Recent decisions</strong>
<table><tr><th>Time</th><th>Pair</th><th>Action</th><th>Regime</th><th>Z</th><th>Reason</th></tr>
{rows if rows else '<tr><td colspan="6">Waiting for first cycle…</td></tr>'}
</table></div>
<p style="color:#64748b;font-size:0.85rem">Auto-refreshes every 30s. Monitoring {len(CANDIDATE_PAIRS)} high-conviction pairs ({len(set(t for p in CANDIDATE_PAIRS for t in p))} tickers) — filtered from 375-pair backtest.</p>
</body></html>"""


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

