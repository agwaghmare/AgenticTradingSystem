import logging
from datetime import date, timedelta

import httpx

from app.config import settings

logger = logging.getLogger("earnings")

_earnings_cache: dict[str, list[date]] = {}


async def _fetch_earnings_dates(ticker: str) -> list[date]:
    if ticker in _earnings_cache:
        return _earnings_cache[ticker]

    if not settings.finnhub_api_key:
        logger.debug("Finnhub API key not set; skipping earnings lookup for %s", ticker)
        return []

    today = date.today()
    window_start = today - timedelta(days=settings.earnings_blackout_days + 7)
    window_end = today + timedelta(days=settings.earnings_blackout_days + 7)
    url = "https://finnhub.io/api/v1/calendar/earnings"
    params = {
        "from": window_start.isoformat(),
        "to": window_end.isoformat(),
        "symbol": ticker,
        "token": settings.finnhub_api_key,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
    except Exception:
        logger.exception("Failed to fetch earnings calendar for %s", ticker)
        return []

    dates: list[date] = []
    for item in payload.get("earningsCalendar", []):
        raw = item.get("date")
        if not raw:
            continue
        try:
            dates.append(date.fromisoformat(raw[:10]))
        except ValueError:
            continue

    _earnings_cache[ticker] = dates
    return dates


async def check_earnings_blackout(ticker: str) -> bool:
    """Return True if today falls within the earnings blackout window."""
    if not settings.finnhub_api_key:
        return False

    earnings_dates = await _fetch_earnings_dates(ticker)
    if not earnings_dates:
        return False

    today = date.today()
    blackout = settings.earnings_blackout_days
    for earnings_date in earnings_dates:
        delta = abs((today - earnings_date).days)
        if delta <= blackout:
            return True
    return False

