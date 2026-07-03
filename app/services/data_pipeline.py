import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pandas as pd
import yfinance as yf
from sqlalchemy import select, delete
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.orm import PriceHistory, TickerMetadata

logger = logging.getLogger("data_pipeline")

_sector_cache: dict[str, str | None] = {}
_industry_cache: dict[str, str | None] = {}
_FETCH_RETRIES = 3
_FETCH_RETRY_DELAY_SEC = 2.0
_TICKER_FETCH_DELAY_SEC = 1.25


def _unique_tickers(
    candidate_pairs: list[tuple[str, str]], extra_tickers: list[str] | None = None
) -> list[str]:
    tickers: set[str] = set()
    for a, b in candidate_pairs:
        tickers.add(a)
        tickers.add(b)
    if extra_tickers:
        tickers.update(extra_tickers)
    tickers.add(settings.market_benchmark_ticker)
    return sorted(tickers)


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def _fetch_ohlcv_via_chart(ticker: str, days: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 5)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    response = httpx.get(
        url,
        params={
            "period1": int(start.timestamp()),
            "period2": int(end.timestamp()),
            "interval": "1d",
            "events": "history",
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()["chart"]["result"][0]
    quote = payload["indicators"]["quote"][0]
    rows = []
    for ts, o, h, l, c, v in zip(
        payload["timestamp"],
        quote.get("open", []),
        quote.get("high", []),
        quote.get("low", []),
        quote.get("close", []),
        quote.get("volume", []),
    ):
        if c is None:
            continue
        idx = pd.Timestamp(ts, unit="s", tz="UTC")
        rows.append(
            {
                "open": float(o or c),
                "high": float(h or c),
                "low": float(l or c),
                "close": float(c),
                "volume": float(v or 0),
                "_ts": idx,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("_ts")
    df.index.name = None
    return df


def _fetch_ohlcv(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    days = settings.price_history_days
    for attempt in range(_FETCH_RETRIES):
        try:
            df = _fetch_ohlcv_via_chart(ticker, days)
            if not df.empty:
                return df
        except Exception as exc:
            logger.warning("Chart API attempt %s/%s failed for %s: %s", attempt + 1, _FETCH_RETRIES, ticker, exc)
            time.sleep(_FETCH_RETRY_DELAY_SEC * (attempt + 1))

    last_error: Exception | None = None
    for attempt in range(_FETCH_RETRIES):
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
            if not df.empty:
                return _normalize_ohlcv(df)
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            return _normalize_ohlcv(df)
        except Exception as exc:
            last_error = exc
            logger.warning("yfinance attempt %s/%s failed for %s: %s", attempt + 1, _FETCH_RETRIES, ticker, exc)
            time.sleep(_FETCH_RETRY_DELAY_SEC * (attempt + 1))
    if last_error:
        logger.error("All fetch attempts failed for %s", ticker)
    return pd.DataFrame()


def _fetch_intraday_last_timestamp(ticker: str) -> datetime | None:
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        last_ts = pd.to_datetime(df.index[-1])
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        else:
            last_ts = last_ts.tz_convert("UTC")
        return last_ts.to_pydatetime()
    except Exception:
        logger.exception("Failed to fetch intraday timestamp for %s", ticker)
        return None


def _fetch_metadata(ticker: str) -> tuple[str | None, str | None]:
    if ticker in _sector_cache:
        return _sector_cache[ticker], _industry_cache.get(ticker)

    time.sleep(_TICKER_FETCH_DELAY_SEC)
    try:
        info = yf.Ticker(ticker).info or {}
        sector = info.get("sector")
        industry = info.get("industry")
        _sector_cache[ticker] = sector
        _industry_cache[ticker] = industry
        return sector, industry
    except Exception:
        logger.exception("Failed to fetch metadata for %s", ticker)
        _sector_cache[ticker] = None
        _industry_cache[ticker] = None
        return None, None


def _fetch_sector(ticker: str) -> str | None:
    sector, _ = _fetch_metadata(ticker)
    return sector


async def sync_ticker_metadata(db: AsyncSession, tickers: list[str]) -> None:
    for ticker in tickers:
        sector, industry = await asyncio.to_thread(_fetch_metadata, ticker)
        stmt = insert(TickerMetadata).values(
            ticker=ticker,
            sector=sector,
            industry=industry,
            updated_at=datetime.now(timezone.utc),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker"],
            set_={"sector": stmt.excluded.sector, "industry": stmt.excluded.industry, "updated_at": stmt.excluded.updated_at},
        )
        await db.execute(stmt)
    await db.commit()


async def sync_price_history(db: AsyncSession, tickers: list[str]) -> None:
    for ticker in tickers:
        await asyncio.sleep(_TICKER_FETCH_DELAY_SEC)
        df = await asyncio.to_thread(_fetch_ohlcv, ticker, f"{settings.price_history_days}d", "1d")
        if df.empty:
            logger.warning("No price history returned for %s", ticker)
            continue

        await db.execute(
            delete(PriceHistory).where(PriceHistory.ticker == ticker, PriceHistory.interval == "1d")
        )

        rows = []
        for ts, row in df.iterrows():
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": ts.to_pydatetime(),
                    "interval": "1d",
                    "open": float(row.get("open", row["close"])),
                    "high": float(row.get("high", row["close"])),
                    "low": float(row.get("low", row["close"])),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0) or 0),
                }
            )
        if rows:
            for row in rows:
                db.add(PriceHistory(**row))
    await db.commit()


async def _load_price_series_from_db(db: AsyncSession, ticker: str) -> pd.DataFrame:
    result = await db.execute(
        select(PriceHistory)
        .where(PriceHistory.ticker == ticker, PriceHistory.interval == "1d")
        .order_by(PriceHistory.timestamp)
    )
    rows = result.scalars().all()
    if not rows:
        return pd.DataFrame()

    data = {
        "open": [float(r.open or r.close) for r in rows],
        "high": [float(r.high or r.close) for r in rows],
        "low": [float(r.low or r.close) for r in rows],
        "close": [float(r.close) for r in rows],
        "volume": [float(r.volume or 0) for r in rows],
    }
    index = pd.DatetimeIndex([r.timestamp for r in rows], tz="UTC")
    return pd.DataFrame(data, index=index)


async def get_latest_prices(
    db: AsyncSession,
    candidate_pairs: list[tuple[str, str]],
    extra_tickers: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return {ticker: {close, high, low, open, last_timestamp}} for the agent."""
    tickers = _unique_tickers(candidate_pairs, extra_tickers)
    await sync_price_history(db, tickers)

    price_data: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        df = await _load_price_series_from_db(db, ticker)
        if df.empty:
            df = await asyncio.to_thread(_fetch_ohlcv, ticker, f"{settings.price_history_days}d", "1d")
            if df.empty:
                continue

        last_ts = await asyncio.to_thread(_fetch_intraday_last_timestamp, ticker)
        if last_ts is None:
            last_daily = df.index[-1]
            last_ts = last_daily.to_pydatetime() if hasattr(last_daily, "to_pydatetime") else last_daily

        price_data[ticker] = {
            "close": df["close"],
            "high": df["high"],
            "low": df["low"],
            "open": df["open"],
            "last_timestamp": last_ts,
        }
    return price_data


async def get_market_returns(db: AsyncSession) -> pd.Series:
    benchmark = settings.market_benchmark_ticker
    df = await _load_price_series_from_db(db, benchmark)
    if df.empty or len(df) < 2:
        df = await asyncio.to_thread(_fetch_ohlcv, benchmark, f"{settings.price_history_days}d", "1d")
    if df.empty or len(df) < 2:
        return pd.Series(dtype=float)
    return df["close"].pct_change().dropna()


async def get_sector_map(db: AsyncSession, tickers: list[str]) -> dict[str, str]:
    result = await db.execute(select(TickerMetadata).where(TickerMetadata.ticker.in_(tickers)))
    rows = {r.ticker: r.sector for r in result.scalars().all()}
    missing = [t for t in tickers if t not in rows or rows[t] is None]
    if missing:
        await sync_ticker_metadata(db, missing)
        result = await db.execute(select(TickerMetadata).where(TickerMetadata.ticker.in_(tickers)))
        rows = {r.ticker: r.sector for r in result.scalars().all()}

    return {t: (rows.get(t) or "Unknown") for t in tickers}


def compute_sector_exposure(
    open_positions: list[dict],
    sector_map: dict[str, str],
    additional_notionals: dict[str, float],
) -> dict[str, float]:
    exposure: dict[str, float] = {}
    for pos in open_positions:
        sector = sector_map.get(pos["ticker"], "Unknown")
        exposure[sector] = exposure.get(sector, 0.0) + abs(pos["market_value"])
    for ticker, notional in additional_notionals.items():
        sector = sector_map.get(ticker, "Unknown")
        exposure[sector] = exposure.get(sector, 0.0) + abs(notional)
    return exposure
