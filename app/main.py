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
from app.services import data_pipeline

from app.pairs_universe import CANDIDATE_PAIRS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

scheduler = AsyncIOScheduler()
broker = BrokerClient()
regime_detector = RegimeDetector()
agent = TradingAgent(broker, regime_detector)

logger.info("Loaded %d candidate pairs", len(CANDIDATE_PAIRS))


async def scheduled_cycle():
    async with AsyncSessionLocal() as db:
        try:
            price_data = await data_pipeline.get_latest_prices(db, CANDIDATE_PAIRS)
            market_returns = await data_pipeline.get_market_returns(db)
            await agent.run_cycle(db, CANDIDATE_PAIRS, price_data, market_returns)
        except Exception:
            logger.exception("Agent cycle failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    scheduler.add_job(scheduled_cycle, "interval", minutes=5, id="agent_cycle")
    scheduler.start()
    logger.info("Agent scheduler started — running first cycle now")
    asyncio.create_task(scheduled_cycle())
    yield
    scheduler.shutdown()


app = FastAPI(title="AgenticTradingSystem", lifespan=lifespan)
app.include_router(router)


@app.get("/", include_in_schema=False)
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/api/")
