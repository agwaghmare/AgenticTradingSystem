"""Single-name mean reversion evaluator (log-price z-score + OU half-life)."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.schemas import DecisionLogSchema, RiskCheckResult
from app.services import stats_engine, risk_engine, decision_logger, earnings
from app.services import data_pipeline, notifier, live_performance
from app.strategy_params import MODEL_VERSION, MIN_CONVICTION_Z

logger = logging.getLogger("agent.single")

SINGLE_SENTINEL = "SINGLE"


class SingleStockEvaluator:
    def __init__(self):
        self._open_singles: dict[str, dict] = {}

    async def evaluate_universe(
        self,
        db: AsyncSession,
        cycle_id: uuid.UUID,
        tickers: list[str],
        price_data: dict[str, dict[str, Any]],
        nav: float,
        open_broker_positions: list[dict],
        regime: str,
        allow_new_entries: bool,
        sector_map: dict[str, str | None],
        pair_leg_tickers: set[str],
    ):
        broker_tickers = {p["ticker"] for p in open_broker_positions}
        self._reconcile(broker_tickers, price_data)

        for ticker in tickers:
            await self._evaluate_one(
                db,
                cycle_id,
                ticker,
                price_data,
                nav,
                open_broker_positions,
                regime,
                allow_new_entries,
                sector_map,
                pair_leg_tickers,
                broker_tickers,
            )

    async def _evaluate_one(
        self,
        db,
        cycle_id,
        ticker,
        price_data,
        nav,
        open_broker_positions,
        regime,
        allow_new_entries,
        sector_map,
        pair_leg_tickers,
        broker_tickers,
    ):
        series = price_data.get(ticker, {}).get("close")
        if series is None or len(series) < settings.price_history_days // 3:
            await self._reject(db, cycle_id, ticker, regime, "Stale or missing price data")
            return

        if ticker in self._open_singles or ticker in broker_tickers:
            if ticker in self._open_singles:
                await self._evaluate_exit(db, cycle_id, ticker, price_data, regime)
            return

        if ticker in pair_leg_tickers:
            await self._reject(
                db, cycle_id, ticker, regime, "Ticker already exposed via pairs book — skip single overlay"
            )
            return

        try:
            mr = stats_engine.check_price_mean_reversion(series, ticker)
        except Exception:
            logger.exception("Mean-reversion check failed for %s", ticker)
            return

        if not mr.is_mean_reverting:
            await self._reject(
                db,
                cycle_id,
                ticker,
                regime,
                f"Not mean-reverting: half_life={mr.half_life_days:.1f}d, z={mr.z_score:.2f}",
                z_score=mr.z_score,
                half_life_days=mr.half_life_days,
            )
            return

        # Entry band: conviction <= |z| < stop — entering beyond the stop means the
        # dislocation is likely structural, not a mean-reversion opportunity.
        if abs(mr.z_score) < MIN_CONVICTION_Z or abs(mr.z_score) >= settings.zscore_stop:
            reason = (
                f"Z-score {mr.z_score:.2f} below high-conviction threshold ({MIN_CONVICTION_Z})"
                if abs(mr.z_score) < MIN_CONVICTION_Z
                else f"Z-score {mr.z_score:.2f} at/beyond stop threshold ({settings.zscore_stop})"
            )
            await self._reject(
                db,
                cycle_id,
                ticker,
                regime,
                reason,
                z_score=mr.z_score,
                half_life_days=mr.half_life_days,
            )
            return

        if not allow_new_entries:
            await self._reject(
                db,
                cycle_id,
                ticker,
                regime,
                "New entries disallowed (regime/drawdown/VaR halt)",
                z_score=mr.z_score,
                half_life_days=mr.half_life_days,
            )
            return

        if await earnings.check_earnings_blackout(ticker):
            await self._reject(
                db, cycle_id, ticker, regime, f"Earnings blackout active for {ticker}", z_score=mr.z_score
            )
            return

        if len(self._open_singles) >= settings.max_single_stock_positions:
            await self._reject(db, cycle_id, ticker, regime, "At max single-stock positions", z_score=mr.z_score)
            return

        price = float(series.iloc[-1])
        high = price_data.get(ticker, {}).get("high", series)
        low = price_data.get(ticker, {}).get("low", series)
        atr = stats_engine.calc_atr(high, low, series, settings.atr_period)
        stop_distance = atr * settings.atr_stop_multiplier

        size = risk_engine.calc_position_size(nav, settings.max_risk_pct_per_trade, stop_distance, price)
        if size <= 0:
            await self._reject(db, cycle_id, ticker, regime, "Position size computed to zero", z_score=mr.z_score)
            return

        action = "ENTER_LONG" if mr.z_score < -MIN_CONVICTION_Z else "ENTER_SHORT"
        side = "BUY" if action == "ENTER_LONG" else "SELL"
        notional = size * price

        sector_exposures = data_pipeline.compute_sector_exposure(
            open_broker_positions, sector_map, {ticker: notional}
        )
        max_sector_notional = max(sector_exposures.values(), default=0.0)
        gross_exposure = (
            sum(abs(p["market_value"]) for p in open_broker_positions) + notional
        ) / nav
        net_delta = notional if action == "ENTER_LONG" else -notional
        net_exposure = (sum(p["market_value"] for p in open_broker_positions) + net_delta) / nav

        risk_check = risk_engine.check_risk_limits(
            notional,
            nav,
            max_sector_notional,
            gross_exposure,
            net_exposure,
            len(self._open_singles) + len(open_broker_positions),
        )
        if not risk_check.passed:
            await self._reject(
                db, cycle_id, ticker, regime, "; ".join(risk_check.reasons), z_score=mr.z_score
            )
            return

        broker = self._broker
        order = broker.place_market_order(ticker, size, side)

        dollar_stop = price - stop_distance if action == "ENTER_LONG" else price + stop_distance
        self._open_singles[ticker] = {
            "direction": "LONG" if action == "ENTER_LONG" else "SHORT",
            "entry_price": price,
            "shares": size,
            "dollar_stop": dollar_stop,
            "entry_z": mr.z_score,
            "entry_date": datetime.now(timezone.utc),
            "half_life_days": mr.half_life_days,
        }

        notifier.notify_trade_signal(
            ticker, SINGLE_SENTINEL, action, side, "—", size, 0, mr.z_score, 0.0
        )

        await decision_logger.log_decision(
            db,
            DecisionLogSchema(
                cycle_id=cycle_id,
                pair_a=ticker,
                pair_b=SINGLE_SENTINEL,
                action=action,
                z_score=mr.z_score,
                half_life_days=mr.half_life_days,
                regime=regime,
                reasoning=f"Single-stock MR: z={mr.z_score:.2f}, half_life={mr.half_life_days:.1f}d, ATR stop={stop_distance:.2f}",
                model_version=MODEL_VERSION,
                nav_at_decision=nav,
                position_size=notional,
            ),
        )
        logger.info("Single-stock %s on %s: %s", action, ticker, order)

    async def _evaluate_exit(self, db, cycle_id, ticker, price_data, regime):
        pos = self._open_singles.get(ticker)
        if pos is None:
            return

        series = price_data.get(ticker, {}).get("close")
        if series is None:
            return

        mr = None
        z = None
        try:
            mr = stats_engine.check_price_mean_reversion(series, ticker)
            z = mr.z_score if np.isfinite(mr.z_score) else None
        except Exception:
            pass

        price_today = float(series.iloc[-1])
        hit_dollar_stop = (
            price_today <= pos["dollar_stop"]
            if pos["direction"] == "LONG"
            else price_today >= pos["dollar_stop"]
        )
        hit_z_stop = z is not None and abs(z) >= settings.zscore_stop
        hit_target = z is not None and abs(z) <= settings.zscore_exit

        entry_date = pos.get("entry_date")
        held_days = (datetime.now(timezone.utc) - entry_date).total_seconds() / 86400 if entry_date else 0.0
        half_life = pos.get("half_life_days")
        if half_life is None or not np.isfinite(half_life) or half_life <= 0:
            hold_limit = float(settings.max_hold_days)
        else:
            hold_limit = min(settings.time_stop_half_lives * half_life, float(settings.max_hold_days))
        hit_time_stop = held_days >= hold_limit

        exit_reason = None
        if hit_dollar_stop:
            exit_reason = "dollar_stop"
        elif hit_z_stop:
            exit_reason = "zscore_stop"
        elif hit_target:
            exit_reason = "target"
        elif mr is not None and not mr.is_mean_reverting:
            exit_reason = "mean_reversion_broke"
        elif hit_time_stop:
            exit_reason = "time_stop"

        if exit_reason is None:
            return

        side = "SELL" if pos["direction"] == "LONG" else "BUY"
        broker = self._broker
        order = broker.place_market_order(ticker, pos["shares"], side)

        if pos["direction"] == "LONG":
            gross = pos["shares"] * (price_today - pos["entry_price"])
        else:
            gross = pos["shares"] * (pos["entry_price"] - price_today)
        commission = pos["shares"] * settings.commission_per_share * 2
        realized_pnl = float(gross - commission)

        await live_performance.record_closed_trade(
            db,
            strategy_type="SINGLE",
            symbol_a=ticker,
            symbol_b=None,
            direction=pos["direction"],
            entry_price=pos["entry_price"],
            exit_price=price_today,
            shares=pos["shares"],
            realized_pnl=realized_pnl,
            exit_reason=exit_reason,
            z_entry=pos["entry_z"],
            z_exit=z,
            model_version=MODEL_VERSION,
        )

        notifier.notify_exit(ticker, SINGLE_SENTINEL, exit_reason, z_score=z, realized_pnl=realized_pnl)

        await decision_logger.log_decision(
            db,
            DecisionLogSchema(
                cycle_id=cycle_id,
                pair_a=ticker,
                pair_b=SINGLE_SENTINEL,
                action="EXIT",
                z_score=z,
                half_life_days=mr.half_life_days if mr is not None else None,
                regime=regime,
                reasoning=f"Single exit on {exit_reason}, realized_pnl=${realized_pnl:.2f}",
                model_version=MODEL_VERSION,
                nav_at_decision=broker.get_account_nav(),
            ),
        )
        logger.info("Single exit %s on %s: %s", exit_reason, ticker, order)
        del self._open_singles[ticker]

    def _reconcile(self, broker_tickers: set[str], price_data: dict):
        for ticker in list(self._open_singles.keys()):
            if ticker not in broker_tickers:
                logger.warning("Dropping stale single-stock position for %s", ticker)
                self._open_singles.pop(ticker, None)

    def bind_broker(self, broker):
        self._broker = broker

    async def _reject(self, db, cycle_id, ticker, regime, reason, **kwargs):
        await decision_logger.log_decision(
            db,
            DecisionLogSchema(
                cycle_id=cycle_id,
                pair_a=ticker,
                pair_b=SINGLE_SENTINEL,
                action="REJECT",
                regime=regime,
                reasoning=reason,
                rejection_reason=reason,
                model_version=MODEL_VERSION,
                **kwargs,
            ),
        )
