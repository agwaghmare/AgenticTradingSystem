import uuid
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.schemas import DecisionLogSchema, RiskCheckResult
from app.services import stats_engine, risk_engine, decision_logger
from app.services.regime import RegimeDetector
from app.services.broker import BrokerClient
from app.services import data_pipeline, earnings, equity_tracker, risk_logger, notifier

logger = logging.getLogger("agent")

MODEL_VERSION = "agentic-trading-v0.1"


class TradingAgent:
    def __init__(self, broker: BrokerClient, regime_detector: RegimeDetector):
        self.broker = broker
        self.regime_detector = regime_detector
        self._prev_hedge_ratios: dict[tuple[str, str], float] = {}

    async def run_cycle(
        self,
        db: AsyncSession,
        candidate_pairs: list[tuple[str, str]],
        price_data: dict[str, dict[str, Any]],
        market_returns: pd.Series,
    ):
        cycle_id = uuid.uuid4()
        nav = self.broker.get_account_nav()
        open_positions = self.broker.get_open_positions()

        regime = self.regime_detector.current_regime(market_returns) if len(market_returns) >= 30 else "unknown"
        drawdown_mtd = await equity_tracker.calc_drawdown_mtd(db, nav)
        halt_status = risk_engine.get_drawdown_status(drawdown_mtd)

        gross_exposure = sum(abs(p["market_value"]) for p in open_positions) / nav if nav else 0.0
        net_exposure = sum(p["market_value"] for p in open_positions) / nav if nav else 0.0
        portfolio_var = self._calc_portfolio_var(nav, open_positions, price_data)

        if halt_status == "HALTED":
            self.broker.flatten_all_positions()
            halt_reason = "Drawdown breached flatten threshold, agent halted"
            notifier.notify_halt(halt_reason)
            await decision_logger.log_decision(
                db,
                DecisionLogSchema(
                    cycle_id=cycle_id,
                    pair_a="ALL",
                    pair_b="ALL",
                    action="EXIT",
                    regime=regime,
                    reasoning=halt_reason,
                    model_version=MODEL_VERSION,
                    nav_at_decision=nav,
                ),
            )
            await risk_logger.log_risk_snapshot(
                db,
                cycle_id=cycle_id,
                nav=nav,
                gross_leverage=gross_exposure,
                net_leverage=net_exposure,
                daily_var_95=portfolio_var,
                drawdown_mtd_pct=drawdown_mtd,
                open_positions=len(open_positions),
                regime=regime,
                halt_status=halt_status,
            )
            return

        allow_new_entries = halt_status == "NORMAL" and regime == "range_bound"
        if allow_new_entries and nav > 0 and portfolio_var / nav > settings.daily_var_95_max_pct:
            allow_new_entries = False
            logger.info("New entries blocked: portfolio VaR %.2f exceeds cap", portfolio_var)

        all_tickers = sorted({t for pair in candidate_pairs for t in pair})
        for pos in open_positions:
            all_tickers.append(pos["ticker"])
        sector_map = await data_pipeline.get_sector_map(db, sorted(set(all_tickers)))

        for pair_a, pair_b in candidate_pairs:
            await self._evaluate_pair(
                db,
                cycle_id,
                pair_a,
                pair_b,
                price_data,
                nav,
                open_positions,
                regime,
                allow_new_entries,
                sector_map,
            )

        await risk_logger.log_risk_snapshot(
            db,
            cycle_id=cycle_id,
            nav=nav,
            gross_leverage=gross_exposure,
            net_leverage=net_exposure,
            daily_var_95=portfolio_var,
            drawdown_mtd_pct=drawdown_mtd,
            open_positions=len(open_positions),
            regime=regime,
            halt_status=halt_status,
        )

    async def _evaluate_pair(
        self,
        db,
        cycle_id,
        pair_a,
        pair_b,
        price_data,
        nav,
        open_positions,
        regime,
        allow_new_entries,
        sector_map,
    ):
        series_a = price_data.get(pair_a, {}).get("close")
        series_b = price_data.get(pair_b, {}).get("close")

        if series_a is None or series_b is None or not self._is_fresh(price_data, pair_a, pair_b):
            await self._reject(db, cycle_id, pair_a, pair_b, regime, "Stale or missing price data")
            return

        coint_result = stats_engine.check_cointegration(series_a, series_b, pair_a, pair_b)
        if not coint_result.is_cointegrated:
            await self._reject(
                db,
                cycle_id,
                pair_a,
                pair_b,
                regime,
                f"Failed cointegration: p={coint_result.p_value:.4f}, half_life={coint_result.half_life_days:.1f}d",
                p_value=coint_result.p_value,
                half_life_days=coint_result.half_life_days,
            )
            return

        prev_ratio = self._prev_hedge_ratios.get((pair_a, pair_b))
        hedge_result = stats_engine.compute_hedge_ratio_kalman(series_a, series_b, pair_a, pair_b, prev_ratio)
        self._prev_hedge_ratios[(pair_a, pair_b)] = hedge_result.hedge_ratio

        if hedge_result.drift_pct > settings.hedge_drift_max_pct:
            await self._reject(
                db,
                cycle_id,
                pair_a,
                pair_b,
                regime,
                f"Hedge ratio drift {hedge_result.drift_pct:.1%} exceeds max",
                hedge_ratio=hedge_result.hedge_ratio,
                hedge_drift_pct=hedge_result.drift_pct,
            )
            return

        if await earnings.check_earnings_blackout(pair_a) or await earnings.check_earnings_blackout(pair_b):
            await self._reject(
                db,
                cycle_id,
                pair_a,
                pair_b,
                regime,
                "Earnings blackout window active",
                hedge_ratio=hedge_result.hedge_ratio,
                p_value=coint_result.p_value,
                half_life_days=coint_result.half_life_days,
            )
            return

        spread = series_a - hedge_result.hedge_ratio * series_b
        z = stats_engine.calc_zscore(spread)

        if not allow_new_entries:
            await self._reject(
                db,
                cycle_id,
                pair_a,
                pair_b,
                regime,
                "New entries disallowed (regime/drawdown/VaR halt) - exits only",
                z_score=z,
                hedge_ratio=hedge_result.hedge_ratio,
                p_value=coint_result.p_value,
                half_life_days=coint_result.half_life_days,
            )
            return

        if abs(z) < settings.zscore_entry:
            await self._reject(
                db,
                cycle_id,
                pair_a,
                pair_b,
                regime,
                f"Z-score {z:.2f} below entry threshold",
                z_score=z,
                hedge_ratio=hedge_result.hedge_ratio,
                p_value=coint_result.p_value,
                half_life_days=coint_result.half_life_days,
            )
            return

        price_a = float(series_a.iloc[-1])
        price_b = float(series_b.iloc[-1])
        high_a = price_data.get(pair_a, {}).get("high", series_a)
        low_a = price_data.get(pair_a, {}).get("low", series_a)
        atr = stats_engine.calc_atr(high_a, low_a, series_a, settings.atr_period)
        stop_distance = atr * settings.atr_stop_multiplier

        size = risk_engine.calc_position_size(nav, settings.max_risk_pct_per_trade, stop_distance, price_a)
        if size <= 0:
            await self._reject(
                db,
                cycle_id,
                pair_a,
                pair_b,
                regime,
                "Position size computed to zero",
                z_score=z,
                hedge_ratio=hedge_result.hedge_ratio,
            )
            return

        notional_a = size * price_a
        notional_b = size * hedge_result.hedge_ratio * price_b
        action = "ENTER_LONG" if z < -settings.zscore_entry else "ENTER_SHORT"

        sector_exposures = data_pipeline.compute_sector_exposure(
            open_positions,
            sector_map,
            {pair_a: notional_a, pair_b: notional_b},
        )
        max_sector_notional = max(sector_exposures.values(), default=0.0)

        gross_exposure = (sum(abs(p["market_value"]) for p in open_positions) + notional_a + notional_b) / nav
        net_delta = (notional_a - notional_b) if action == "ENTER_LONG" else (-notional_a + notional_b)
        net_exposure = (sum(p["market_value"] for p in open_positions) + net_delta) / nav

        risk_check = risk_engine.check_risk_limits(
            notional_a,
            nav,
            max_sector_notional,
            gross_exposure,
            net_exposure,
            len(open_positions),
        )

        if not risk_check.passed:
            await self._reject(
                db,
                cycle_id,
                pair_a,
                pair_b,
                regime,
                "; ".join(risk_check.reasons),
                z_score=z,
                hedge_ratio=hedge_result.hedge_ratio,
            )
            return

        corr_check = self._check_pair_correlation(pair_a, pair_b, price_data, open_positions)
        if not corr_check.passed:
            await self._reject(
                db,
                cycle_id,
                pair_a,
                pair_b,
                regime,
                "; ".join(corr_check.reasons),
                z_score=z,
                hedge_ratio=hedge_result.hedge_ratio,
            )
            return

        side_a = "BUY" if action == "ENTER_LONG" else "SELL"
        side_b = "SELL" if action == "ENTER_LONG" else "BUY"

        qty_b = size * hedge_result.hedge_ratio
        order_a = self.broker.place_market_order(pair_a, size, side_a)
        order_b = self.broker.place_market_order(pair_b, qty_b, side_b)

        await decision_logger.log_decision(
            db,
            DecisionLogSchema(
                cycle_id=cycle_id,
                pair_a=pair_a,
                pair_b=pair_b,
                action=action,
                z_score=z,
                hedge_ratio=hedge_result.hedge_ratio,
                hedge_drift_pct=hedge_result.drift_pct,
                p_value=coint_result.p_value,
                half_life_days=coint_result.half_life_days,
                regime=regime,
                reasoning=f"Z-score {z:.2f} triggered entry, cointegrated p={coint_result.p_value:.4f}",
                model_version=MODEL_VERSION,
                nav_at_decision=nav,
                position_size=notional_a,
            ),
        )
        logger.info("Executed %s on %s/%s: %s, %s", action, pair_a, pair_b, order_a, order_b)
        notifier.notify_trade_signal(
            pair_a,
            pair_b,
            action,
            side_a,
            side_b,
            size,
            qty_b,
            z,
            hedge_result.hedge_ratio,
        )

    def _check_pair_correlation(self, pair_a, pair_b, price_data, open_positions):
        series_a = price_data.get(pair_a, {}).get("close")
        if series_a is None or len(series_a) < 3:
            return RiskCheckResult(passed=True, reasons=[])

        new_returns = series_a.pct_change().dropna().values[-60:]
        existing_returns: dict[str, np.ndarray] = {}
        for pos in open_positions:
            ticker = pos["ticker"]
            if ticker in (pair_a, pair_b):
                continue
            close = price_data.get(ticker, {}).get("close")
            if close is not None and len(close) > 2:
                existing_returns[ticker] = close.pct_change().dropna().values[-60:]

        return risk_engine.check_correlation_exposure(
            new_returns,
            existing_returns,
            settings.correlation_threshold,
        )

    def _calc_portfolio_var(self, nav: float, open_positions: list[dict], price_data: dict) -> float:
        if not open_positions or nav <= 0:
            return 0.0

        position_returns = []
        weights = []
        total_abs = sum(abs(p["market_value"]) for p in open_positions) or 1.0
        for pos in open_positions:
            close = price_data.get(pos["ticker"], {}).get("close")
            if close is None or len(close) < 5:
                continue
            position_returns.append(close.pct_change().dropna().values[-60:])
            weights.append(abs(pos["market_value"]) / total_abs)

        if not position_returns:
            return 0.0

        var_return = risk_engine.calc_portfolio_var(position_returns, weights)
        return float(var_return * nav)

    async def _reject(self, db, cycle_id, pair_a, pair_b, regime, reason, **kwargs):
        await decision_logger.log_decision(
            db,
            DecisionLogSchema(
                cycle_id=cycle_id,
                pair_a=pair_a,
                pair_b=pair_b,
                action="REJECT",
                regime=regime,
                reasoning=reason,
                rejection_reason=reason,
                model_version=MODEL_VERSION,
                **kwargs,
            ),
        )

    def _is_fresh(self, price_data, pair_a, pair_b) -> bool:
        now = datetime.now(timezone.utc)
        for ticker in (pair_a, pair_b):
            data = price_data.get(ticker, {})
            last_ts = data.get("last_timestamp")
            close = data.get("close")
            if last_ts is None or close is None or len(close) == 0:
                return False
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            if (now - last_ts).total_seconds() <= settings.stale_price_seconds:
                continue
            last_daily = close.index[-1]
            if last_daily.tzinfo is None:
                last_daily = last_daily.tz_localize("UTC")
            daily_age = (now - last_daily.to_pydatetime()).total_seconds()
            if daily_age > 4 * 24 * 3600:
                return False
        return True
