import uuid
import logging
from datetime import datetime

import pandas as pd

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.schemas import DecisionLogSchema
from app.services import stats_engine, risk_engine, decision_logger
from app.services.regime import RegimeDetector
from app.services.broker import BrokerClient

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
        price_data: dict[str, pd.DataFrame],
        market_returns: pd.Series,
    ):
        cycle_id = uuid.uuid4()
        nav = self.broker.get_account_nav()
        open_positions = self.broker.get_open_positions()

        # Step 1: regime check
        regime = self.regime_detector.current_regime(market_returns)

        # Step 2: drawdown / halt check
        drawdown_mtd = self._calc_drawdown_mtd(nav)  # implement vs your equity curve store
        halt_status = risk_engine.get_drawdown_status(drawdown_mtd)

        if halt_status == "HALTED":
            self.broker.flatten_all_positions()
            await decision_logger.log_decision(
                db,
                DecisionLogSchema(
                    cycle_id=cycle_id,
                    pair_a="ALL",
                    pair_b="ALL",
                    action="EXIT",
                    regime=regime,
                    reasoning="Drawdown breached flatten threshold, agent halted",
                    model_version=MODEL_VERSION,
                    nav_at_decision=nav,
                ),
            )
            return

        allow_new_entries = halt_status == "NORMAL" and regime == "range_bound"

        for pair_a, pair_b in candidate_pairs:
            await self._evaluate_pair(
                db, cycle_id, pair_a, pair_b, price_data, nav, open_positions, regime, allow_new_entries
            )

    async def _evaluate_pair(
        self, db, cycle_id, pair_a, pair_b, price_data, nav, open_positions, regime, allow_new_entries
    ):
        series_a = price_data.get(pair_a, {}).get("close")
        series_b = price_data.get(pair_b, {}).get("close")

        if series_a is None or series_b is None or not self._is_fresh(price_data, pair_a, pair_b):
            await self._reject(db, cycle_id, pair_a, pair_b, regime, "Stale or missing price data")
            return

        coint_result = stats_engine.check_cointegration(series_a, series_b, pair_a, pair_b)
        if not coint_result.is_cointegrated:
            await self._reject(
                db, cycle_id, pair_a, pair_b, regime,
                f"Failed cointegration: p={coint_result.p_value:.4f}, half_life={coint_result.half_life_days:.1f}d",
                p_value=coint_result.p_value, half_life_days=coint_result.half_life_days,
            )
            return

        prev_ratio = self._prev_hedge_ratios.get((pair_a, pair_b))
        hedge_result = stats_engine.compute_hedge_ratio_kalman(series_a, series_b, pair_a, pair_b, prev_ratio)
        self._prev_hedge_ratios[(pair_a, pair_b)] = hedge_result.hedge_ratio

        if hedge_result.drift_pct > settings.hedge_drift_max_pct:
            await self._reject(
                db, cycle_id, pair_a, pair_b, regime,
                f"Hedge ratio drift {hedge_result.drift_pct:.1%} exceeds max",
                hedge_ratio=hedge_result.hedge_ratio, hedge_drift_pct=hedge_result.drift_pct,
            )
            return

        spread = series_a - hedge_result.hedge_ratio * series_b
        z = stats_engine.calc_zscore(spread)

        if not allow_new_entries:
            await self._reject(
                db, cycle_id, pair_a, pair_b, regime,
                "New entries disallowed (regime/drawdown halt) - exits only",
                z_score=z, hedge_ratio=hedge_result.hedge_ratio,
                p_value=coint_result.p_value, half_life_days=coint_result.half_life_days,
            )
            return

        if abs(z) < settings.zscore_entry:
            await self._reject(
                db, cycle_id, pair_a, pair_b, regime, f"Z-score {z:.2f} below entry threshold",
                z_score=z, hedge_ratio=hedge_result.hedge_ratio,
                p_value=coint_result.p_value, half_life_days=coint_result.half_life_days,
            )
            return

        # Signal triggered - size and risk check
        price_a = float(series_a.iloc[-1])
        stop_distance = price_a * 0.02  # placeholder - tie to ATR or actual stop logic
        size = risk_engine.calc_position_size(nav, settings.max_risk_pct_per_trade, stop_distance, price_a)
        notional = size * price_a

        sector_notional = notional  # TODO: aggregate actual sector exposure from open_positions
        gross_exposure = sum(abs(p["market_value"]) for p in open_positions) / nav
        net_exposure = sum(p["market_value"] for p in open_positions) / nav

        risk_check = risk_engine.check_risk_limits(
            notional, nav, sector_notional, gross_exposure, net_exposure, len(open_positions)
        )

        if not risk_check.passed:
            await self._reject(
                db, cycle_id, pair_a, pair_b, regime, "; ".join(risk_check.reasons),
                z_score=z, hedge_ratio=hedge_result.hedge_ratio,
            )
            return

        action = "ENTER_LONG" if z < -settings.zscore_entry else "ENTER_SHORT"
        side_a = "BUY" if action == "ENTER_LONG" else "SELL"
        side_b = "SELL" if action == "ENTER_LONG" else "BUY"

        order_a = self.broker.place_market_order(pair_a, size, side_a)
        order_b = self.broker.place_market_order(pair_b, size * hedge_result.hedge_ratio, side_b)

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
                position_size=notional,
            ),
        )
        logger.info(f"Executed {action} on {pair_a}/{pair_b}: {order_a}, {order_b}")

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
        for ticker in (pair_a, pair_b):
            last_ts = price_data.get(ticker, {}).get("last_timestamp")
            if last_ts is None:
                return False
            if (datetime.utcnow() - last_ts).total_seconds() > settings.stale_price_seconds:
                return False
        return True

    def _calc_drawdown_mtd(self, nav: float) -> float:
        # TODO: replace with real equity curve lookup (peak NAV this month vs current)
        return 0.0
