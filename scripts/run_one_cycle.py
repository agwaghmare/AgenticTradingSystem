"""Run a single agent cycle (for local verification)."""
import asyncio
import logging

from app.db.session import AsyncSessionLocal, init_db
from app.pairs_universe import CANDIDATE_PAIRS
from app.main import agent
from app.services import data_pipeline

logging.basicConfig(level=logging.INFO)


async def main():
    await init_db()
    async with AsyncSessionLocal() as db:
        price_data = await data_pipeline.get_latest_prices(db, CANDIDATE_PAIRS)
        market_returns = await data_pipeline.get_market_returns(db)
        await agent.run_cycle(db, CANDIDATE_PAIRS, price_data, market_returns)
    print("Cycle complete.")


if __name__ == "__main__":
    asyncio.run(main())
