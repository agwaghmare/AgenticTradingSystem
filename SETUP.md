# AgenticTradingSystem — Setup & Deployment

Follow these phases in order. Do not skip to live trading. Each phase has a
gate you must clear before moving to the next.

---

## Phase 0: Prerequisites

- Python 3.11+
- Docker Desktop (for local Postgres)
- Cursor (or any IDE) with this repo opened
- Alpaca account (paper trading is free, sign up at alpaca.markets)
- Anthropic API key (if using the LLM agent layer)
- Market data source: Polygon.io (paid, better) or yfinance (free, delayed/limited)

---

## Phase 1: Local Environment Setup

```bash
# 1. Clone/open the repo in Cursor
cd AgenticTradingSystem

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start local Postgres via Docker
docker run --name agentictradingsystem-db \
  -e POSTGRES_PASSWORD=localpassword \
  -e POSTGRES_DB=agentictradingsystem \
  -p 5432:5432 \
  -d postgres:16

# 5. Copy env template and fill in real values
cp .env.example .env
```

Edit `.env`:
```
DATABASE_URL=postgresql+asyncpg://postgres:localpassword@localhost:5432/agentictradingsystem
ALPACA_API_KEY=<your paper key>
ALPACA_SECRET_KEY=<your paper secret>
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ANTHROPIC_API_KEY=<your key>
```

**Gate:** `docker ps` shows `agentictradingsystem-db` running. `psql` or any DB client can connect.

---

## Phase 2: Database Migrations

Use Alembic instead of `init_db()`'s `create_all` once you're past prototyping —
`create_all` doesn't handle schema changes cleanly.

```bash
pip install alembic
alembic init alembic
```

Edit `alembic.ini`: set `sqlalchemy.url` to your `DATABASE_URL` (sync driver,
not asyncpg — use `postgresql://` for alembic's own connection).

Edit `alembic/env.py` to import `Base` from `app.db.session` and your models
from `app.models.orm`, then set `target_metadata = Base.metadata`.

```bash
alembic revision --autogenerate -m "initial schema"
alembic upgrade head
```

**Gate:** `\dt` in psql shows `decision_log`, `order_log`, `risk_snapshot`,
`model_performance`.

---

## Phase 3: Data Pipeline (replace stubs)

This is the biggest gap in the skeleton — `main.py` currently passes
`price_data={}`. Build a real ingestion module:

```
app/services/data_pipeline.py
```

Responsibilities:
- Pull daily/intraday OHLCV for your candidate universe (Polygon/yfinance)
- Store to a `price_history` table (add this table via another Alembic migration)
- Expose a function that returns the `pd.DataFrame` shape `trading_agent.py`
  expects: `{ticker: {"close": pd.Series, "last_timestamp": datetime}}`

**Gate:** You can call your pipeline function and get back real, fresh price
series for at least 10 candidate pairs.

---

## Phase 4: Offline Backtest (do this before any live cycle runs)

Do NOT skip this. Test `stats_engine.py` and `risk_engine.py` logic against
1-2 years of historical data in a plain script (not through FastAPI/scheduler):

```python
# scripts/backtest.py
from app.services import stats_engine
import pandas as pd

# load historical data for candidate pairs
# run check_cointegration() across the full universe
# confirm: how many pairs actually pass p<0.05 and half_life<30d?
# run calc_zscore() across history, count how many entry signals fire
# sanity check: do the signals make economic sense (no lookahead bias)?
```

**Gate:** You have empirical confidence the cointegration/half-life/z-score
filters produce a reasonable number of signals (not zero, not hundreds/day).
If almost nothing passes, your universe or thresholds are off — fix here,
not in production.

---

## Phase 5: Paper Trading (minimum 2-4 weeks)

```bash
uvicorn app.main:app --reload --port 8000
```

This starts the FastAPI app, runs `init_db()`, and starts the APScheduler
job (`scheduled_cycle`, every 5 min as configured in `main.py`).

Confirm it's alive:
```bash
curl http://localhost:8000/api/health
curl http://localhost:8000/api/decisions/recent
curl http://localhost:8000/api/risk/latest
```

**Daily routine during paper trading:**
- Check `decision_log` — do rejections make sense? Any silent failures?
- Check `risk_snapshot` — is VaR/leverage tracking as expected?
- Compare paper P&L vs your backtest expectations

**Gate:** 2-4 weeks of paper trading with no unexplained errors, risk limits
holding as designed, and performance roughly matching backtest expectations.

---

## Phase 6: Going Live

1. Swap Alpaca keys/URL in `.env` to live: `https://api.alpaca.markets`
2. **Start with minimal capital** — enough to validate execution, not your
   target allocation
3. Re-run Phase 5's daily routine, now on real money, for another 1-2 weeks
   before scaling up

```
ALPACA_BASE_URL=https://api.alpaca.markets
```

**Never skip from Phase 4 straight to Phase 6.**

---

## Deployment (running it "live" continuously)

You need the FastAPI app + scheduler running 24/7, not just on your laptop.
Two reasonable options:

### Option A: Docker + a cloud VM (simplest for a solo project)

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t agentictradingsystem .
docker run -d --env-file .env -p 8000:8000 --restart unless-stopped agentictradingsystem
```

Deploy to any $5-20/mo VM (DigitalOcean, Hetzner, AWS Lightsail). Use a
managed Postgres (DigitalOcean/Supabase/RDS) instead of a DB container on the
same box once live — separates data from compute.

### Option B: systemd (if running on your own always-on machine)

```ini
# /etc/systemd/system/agentictradingsystem.service
[Unit]
Description=AgenticTradingSystem agent
After=network.target

[Service]
User=youruser
WorkingDirectory=/path/to/AgenticTradingSystem
EnvironmentFile=/path/to/AgenticTradingSystem/.env
ExecStart=/path/to/AgenticTradingSystem/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable agentictradingsystem
sudo systemctl start agentictradingsystem
sudo systemctl status agentictradingsystem
```

**Either way:**
- Set `--restart unless-stopped` (Docker) or `Restart=always` (systemd) so a
  crash doesn't silently kill your risk monitoring
- Set up basic alerting (even a simple cron job hitting `/api/health` and
  texting/emailing you on failure) — you want to know immediately if the
  agent goes down while holding positions
- Never run live trading from a laptop that sleeps/closes

---

## Cursor-specific notes

- Use Cursor's built-in terminal for all commands above — no need to leave the IDE
- Add `.env`, `venv/`, `__pycache__/` to `.gitignore` before your first commit
  (never commit API keys)
- Use Cursor's agent mode to help write `data_pipeline.py` and the Alembic
  migration for `price_history` — those are the two real gaps left in the
  skeleton
