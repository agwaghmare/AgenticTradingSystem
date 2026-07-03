"""Print out-of-sample live Sharpe from closed paper trades."""
import asyncio
import json

from app.db.session import AsyncSessionLocal, init_db
from app.services import live_performance


async def main():
    await init_db()
    async with AsyncSessionLocal() as db:
        stats = await live_performance.get_live_performance(db)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
