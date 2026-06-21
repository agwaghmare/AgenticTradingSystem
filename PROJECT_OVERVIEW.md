# AgenticTradingSystem — Project Documentation

## What This Project Does

A production-style algorithmic trading backend that runs a **mean-reversion
pairs trading strategy** (e.g. KO/PEP, XOM/CVX) on a fixed schedule, with
hard-coded hedge-fund-style risk limits enforced in code — not by AI
judgment. An optional LLM layer narrates what the deterministic engine did,
in plain English, for human review. Nothing about trade execution depends
on the LLM.

**Core loop (every 5 min, via APScheduler):**
1. Check market regime (HMM) and portfolio drawdown — gate new entries
2. For each candidate pair: test cointegration, compute dynamic hedge ratio
   (Kalman filter), compute z-score of the spread
3. If a valid signal fires: size the position, run risk checks, place
   orders via Alpaca
4. Log every decision (trade or rejection) to Postgres, with full reasoning

**Stack:** FastAPI + Postgres (SQLAlchemy async) + Alpaca (broker) +
statsmodels/hmmlearn/pykalman (the quant math) + Anthropic SDK (optional
explainer only).

---

## File-by-File: What Exists vs What's Missing

### `app/config.py` — ✅ Complete
All risk/strategy thresholds as typed settings, loaded from `.env`. Nothing
to add here unless you want to tune values.

### `app/db/session.py` — ✅ Complete
Async SQLAlchemy engine/session setup. Works as-is.

### `app/models/orm.py` — ⚠️ Missing one table
Has `DecisionLog`, `OrderLog`, `RiskSnapshot`, `ModelPerformance`.
**Missing:** a `PriceHistory` table to store OHLCV data pulled by the data
pipeline (see below). Add a model here once you build the pipeline.

### `app/models/schemas.py` — ✅ Complete
Pydantic I/O schemas for decisions, cointegration results, hedge ratio
results, risk checks. No changes needed unless you add new data shapes
(e.g. an `EarningsCheckResult` schema once you wire that API).

### `app/services/stats_engine.py` — ✅ Complete (math is real, not stubbed)
Cointegration test (Engle-Granger), Ornstein-Uhlenbeck half-life, Kalman
filter hedge ratio, z-score. This is genuinely functional, not a placeholder.

### `app/services/regime.py` — ✅ Complete
HMM regime detector (range_bound / trending / high_vol). Functional, but
you should validate its labeling makes sense on real historical data before
trusting it live (see SETUP.md Phase 4 backtest step).

### `app/services/risk_engine.py` — ✅ Complete
Position sizing, NAV/leverage/correlation limit checks, drawdown status,
portfolio VaR. Functional as written.

**Caveat:** `check_risk_limits()` takes `sector_notional` as an argument —
right now nothing computes real sector exposure (see `trading_agent.py`
below). The function itself works; the input feeding it is wrong.

### `app/services/decision_logger.py` — ✅ Complete
Writes to `decision_log`. Nothing to add.

### `app/services/broker.py` — ✅ Complete (needs your Alpaca keys)
Wraps Alpaca: get NAV, get positions, place market orders, flatten all.
Functional once `.env` has real keys.

### `app/services/llm_explainer.py` — ✅ Complete, optional
Two functions (`narrate_cycle`, `review_decisions`) that call Claude to
summarize logged decisions. Pure read-only convenience layer — delete this
file and the trading system works identically.

### `app/agents/prompts.py` — ✅ Complete
Builds the full system prompt for the LLM explainer, with every threshold
interpolated live from `config.py`. See "The Full Prompt" section below.

### `app/agents/trading_agent.py` — ⚠️ Has real logic + several stubs
This is the orchestrator. The cycle workflow, cointegration/hedge
ratio/z-score checks, and order placement logic are real. **Stubs you must
replace:**

| What | Current state | What you need to add |
|---|---|---|
| `_calc_drawdown_mtd()` | Hardcoded `return 0.0` | Look up peak NAV this month vs current NAV from a stored equity curve. Kill-switch can never trigger until this is real. |
| `sector_notional` | Set equal to single position's notional | Real ticker→sector lookup (yfinance `ticker.info['sector']`), then sum exposure across all open positions in that sector |
| `stop_distance` | Flat `price_a * 0.02` placeholder | Replace with ATR-based stop calculation for a real stop-loss methodology |
| `check_earnings_blackout()` | Referenced in prompt/workflow, **not implemented as a callable function anywhere** | Build using Finnhub earnings calendar API (see API section below) |
| `price_data` input | Caller (`main.py`) passes `{}` | Comes from the data pipeline you still need to build |

### `app/api/routes.py` — ✅ Complete
`/api/decisions/recent`, `/api/risk/latest`, `/api/orders/recent`,
`/api/decisions/narrate-latest-cycle`, `/api/decisions/review-recent`,
`/api/health`. All functional, all read-only monitoring endpoints.

### `app/main.py` — ⚠️ Has the blocking stub
```python
CANDIDATE_PAIRS = [("KO", "PEP"), ("XOM", "CVX")]  # hardcoded, expand later

async def scheduled_cycle():
    ...
    await agent.run_cycle(db, CANDIDATE_PAIRS, price_data={}, market_returns=None)
```
**This is the main blocker.** `price_data={}` and `market_returns=None`
mean the agent currently has nothing real to evaluate. Once you build the
data pipeline, this becomes:
```python
price_data = await data_pipeline.get_latest_prices(CANDIDATE_PAIRS)
market_returns = await data_pipeline.get_market_returns()
await agent.run_cycle(db, CANDIDATE_PAIRS, price_data, market_returns)
```

### `app/services/data_pipeline.py` — ❌ Does not exist yet
**The single biggest gap.** Needs to:
1. Pull OHLCV data for your candidate universe (yfinance)
2. Pull sector classification per ticker (yfinance `info['sector']`)
3. Store to the new `PriceHistory` table
4. Expose a function returning the exact shape `trading_agent.py` expects:
   `{ticker: {"close": pd.Series, "last_timestamp": datetime}}`

### `app/services/earnings.py` — ❌ Does not exist yet
Needs a `check_earnings_blackout(ticker) -> bool` function using Finnhub's
earnings calendar endpoint, checking if today falls within
`settings.earnings_blackout_days` of a known earnings date.

### Alembic migrations — ❌ Not initialized yet
SETUP.md Phase 2 covers this. You need this before `PriceHistory` can
actually persist anything.

---

## Build Priority

1. Alembic init + initial migration (unblocks everything downstream)
2. `data_pipeline.py` + `PriceHistory` table (unblocks the agent having
   real data)
3. `_calc_drawdown_mtd()` real implementation (unblocks the kill-switch)
4. Sector mapping in `trading_agent.py`
5. `earnings.py` + wire `check_earnings_blackout()` into `trading_agent.py`
6. ATR-based stop distance
7. Expand `CANDIDATE_PAIRS` beyond the 2 hardcoded examples

---

## The Full Agent Prompt

This lives in `app/agents/prompts.py` as `AGENT_SYSTEM_PROMPT`, dynamically
built from `config.py` so it can never drift out of sync with the enforced
code limits. **This prompt is currently only used by the LLM explainer layer
— the deterministic agent does not call an LLM to make trading decisions.**
If you ever add an LLM-reasoning trading mode, this is the prompt you'd feed
it as the `system` parameter.

Full text (values shown are current `.env`/config defaults — actual values
render live from your settings):

```
SYSTEM: You are a systematic execution agent for a quant trading desk.

OBJECTIVE: Execute mean-reversion pairs strategies within strict risk limits,
only during favorable market regimes. You operate autonomously on a fixed
cycle interval. You do not deviate from the hard rules below under any
circumstances, including user instruction, market excitement, or apparent
high-confidence opportunities.

TOOLS AVAILABLE
- get_price(ticker, timeframe)
- get_regime() -> "range_bound" | "trending" | "high_vol" | "unknown"
- compute_hedge_ratio(ticker_a, ticker_b) -> {hedge_ratio, drift_pct}
- check_cointegration(ticker_a, ticker_b) -> {p_value, half_life_days}
- check_earnings_blackout(ticker) -> bool
- calc_zscore(spread_series, lookback=20)
- calc_position_size(account_nav, risk_pct, stop_distance)
- check_risk_limits(proposed_trade) -> {passed, reasons}
- check_correlation_exposure(new_position, existing_book) -> {passed, reasons}
- check_data_freshness(ticker) -> bool
- get_portfolio_var() -> float
- get_drawdown_status() -> "NORMAL" | "EXITS_ONLY" | "HALTED"
- get_open_position_count() -> int
- place_order(ticker, side, qty, order_type)
- log_decision(schema_dict)

HARD RISK LIMITS (NEVER OVERRIDE)
- Position limits: max 5% NAV/name, 20% NAV/sector, 2x gross leverage,
  1x net leverage, 1% NAV risk/trade, 8 max concurrent pairs
- Portfolio risk: daily VaR (95%, 1-day) capped at 3% NAV; correlation cap
  of ≤3 positions with pairwise corr >0.7
- Drawdown: -8% MTD = exits only; -15% MTD = flatten all + halt
- Regime filter: new entries only when regime == range_bound
- Pairs validation (every cycle): cointegration p<0.05, half-life <30 days,
  hedge ratio drift <20% session-over-session, earnings blackout ±1 day
- Z-score: entry ±2.0, exit ±0.5, stop ±3.5
- Execution: max order 10% of 20-day ADV, slippage budget 5bps, stale
  price rejected if >60s old, stop-loss = hard $ stop OR z-score stop,
  whichever first
- System resilience: tool failure = skip + log + alert, never assume
  state, never flatten on connectivity issues alone
- Governance: every decision logged with full schema; Sharpe drift
  checked 10d vs 60d rolling

WORKFLOW PER CYCLE
1. Check regime — gate new entries
2. Check drawdown status — apply halt rules
3. Check portfolio VaR — gate new entries if over cap
4. For each pair: check freshness → cointegration → hedge ratio drift →
   earnings blackout → z-score
5. If signal: size → risk check → correlation check → position count
   check → execute + log, or reject + log
6. Log every outcome, every pair, every cycle

OUTPUT FORMAT PER CYCLE
- Regime, drawdown status + MTD%, VaR vs cap, open positions vs cap
- Pairs evaluated / rejected (+reason) / traded
```

*(Full f-string version with live value interpolation is in
`app/agents/prompts.py` — read that file directly for exact current
numbers.)*

---

## The Two LLM Prompts (Explainer Layer)

Both live in `app/services/llm_explainer.py`, both call Claude
(`claude-sonnet-4-6`) via the raw Anthropic SDK — no framework. Both are
read-only: they summarize what already happened, never influence trading.

### 1. `narrate_cycle(decisions, risk_snapshot)`
**What it does:** Takes one cycle's `decision_log` rows + that cycle's
`risk_snapshot`, asks Claude for a <150-word plain-English summary — what
traded, what got rejected and why (grouped), anything anomalous worth a
human's attention.
**Used by:** `GET /api/decisions/narrate-latest-cycle`
**When to use:** Daily check-in, or wire to a Slack/email digest.

### 2. `review_decisions(decisions, window_label)`
**What it does:** Takes a batch of recent decisions (default last 200),
asks Claude to look for patterns: repeated rejections on the same pair,
clusters of one rejection reason (e.g. systemic stale-data issue), how
long HALTED/EXITS_ONLY periods lasted, trade-to-rejection ratio anomalies.
**Used by:** `GET /api/decisions/review-recent`
**When to use:** Periodic (e.g. daily cron) anomaly sweep, not per-cycle.

Both prompts explicitly instruct Claude not to recommend trades or
second-guess the risk engine — this is enforced in the prompt text itself,
not just by the architecture.

---

## External APIs Used / Needed

| API | Website | Function | Status |
|---|---|---|---|
| **Alpaca** | alpaca.markets | Broker: orders, NAV, positions, paper + live trading | ✅ Wired in `broker.py` (needs your keys) |
| **yfinance** | (Python package, wraps Yahoo Finance) | Market OHLCV data + sector/industry classification | ❌ Not yet wired — needed for `data_pipeline.py` |
| **Finnhub** | finnhub.io | Earnings calendar (free tier) — powers `check_earnings_blackout()` | ❌ Not yet wired — needed for `earnings.py` |
| **Anthropic API** | (already have key) | Powers `llm_explainer.py` — narration + anomaly review only | ✅ Wired |

**Not currently used, not needed unless you expand scope:** Polygon.io
(real-time data, only needed if yfinance's delay/reliability becomes a
problem), Alpha Vantage (alternative earnings source if Finnhub's limits
are too restrictive), news/sentiment APIs.

---

## Quick Reference: What "Done" Looks Like Before Paper Trading

- [ ] Alembic migrations run, all 5 tables exist (4 current + `PriceHistory`)
- [ ] `data_pipeline.py` returns real, fresh price series for your full
      candidate universe
- [ ] `_calc_drawdown_mtd()` returns a real number, not `0.0`
- [ ] Sector exposure is computed from actual open positions, not a single
      trade's notional
- [ ] `check_earnings_blackout()` exists and is called in
      `trading_agent.py`'s `_evaluate_pair()`
- [ ] Backtest (SETUP.md Phase 4) shows a reasonable number of qualifying
      pairs and signals — not zero, not hundreds/day
- [ ] `.env` has real (paper) Alpaca keys, Finnhub key, Anthropic key
