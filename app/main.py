import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.db.session import init_db, AsyncSessionLocal
from app.api.routes import router
from app.agents.trading_agent import TradingAgent
from app.services.broker import BrokerClient
from app.services.regime import RegimeDetector
from app.services import data_pipeline, notifier, cycle_reporter
from app.config import settings

from app.pairs_universe import CANDIDATE_PAIRS, SINGLE_STOCK_UNIVERSE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

scheduler = AsyncIOScheduler()
broker = BrokerClient()
regime_detector = RegimeDetector()
agent = TradingAgent(broker, regime_detector)

logger.info(
    "Loaded %d pairs + %d single-stock tickers (singles %s)",
    len(CANDIDATE_PAIRS),
    len(SINGLE_STOCK_UNIVERSE),
    "enabled" if settings.enable_single_stock else "disabled",
)


async def scheduled_cycle():
    async with AsyncSessionLocal() as db:
        try:
            singles = SINGLE_STOCK_UNIVERSE if settings.enable_single_stock else None
            price_data = await data_pipeline.get_latest_prices(db, CANDIDATE_PAIRS, singles)
            market_returns = await data_pipeline.get_market_returns(db)
            cycle_id = await agent.run_cycle(db, CANDIDATE_PAIRS, price_data, market_returns, singles)
            await cycle_reporter.report_cycle(db, cycle_id)
        except Exception as exc:
            logger.exception("Agent cycle failed")
            notifier.notify_error("scheduled_cycle", str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    try:
        import subprocess
        subprocess.run(["python", "-m", "alembic", "upgrade", "head"], check=False)
    except Exception:
        logger.warning("Alembic upgrade skipped (may already be current)")

    interval = settings.cycle_interval_minutes
    scheduler.add_job(scheduled_cycle, "interval", minutes=interval, id="agent_cycle")
    scheduler.start()
    logger.info("Agent scheduler started — interval %d min", interval)

    mode = "Alpaca Paper" if broker.client else "Simulation"
    nav = broker.get_account_nav()
    notifier.notify_startup(
        nav,
        mode,
        len(CANDIDATE_PAIRS),
        len(SINGLE_STOCK_UNIVERSE) if settings.enable_single_stock else 0,
    )

    asyncio.create_task(scheduled_cycle())
    yield
    scheduler.shutdown()


app = FastAPI(title="AgenticTradingSystem", lifespan=lifespan)
app.include_router(router)


@app.get("/", include_in_schema=False)
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/api/")
