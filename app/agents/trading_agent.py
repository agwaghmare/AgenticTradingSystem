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
from app.services import data_pipeline, earnings, equity_tracker, risk_logger, notifier, live_performance
from app.agents.single_stock import SingleStockEvaluator
from app.strategy_params import MODEL_VERSION

logger = logging.getLogger("agent")


class TradingAgent:
    def __init__(self, broker: BrokerClient, regime_detector: RegimeDetector):
        self.broker = broker
        self.regime_detector = regime_detector
        self._prev_hedge_ratios: dict[tuple[str, str], float] = {}
        self._consecutive_breaks: dict[tuple[str, str], int] = {}
        self._open_positions: dict[tuple[str, str], dict] = {}
        self._single_stock = SingleStockEvaluator()
        self._single_stock.bind_broker(broker)

    async def run_cycle(
        self,
        db: AsyncSession,
        candidate_pairs: list[tuple[str, str]],
        price_data: dict[str, dict[str, Any]],
        market_returns: pd.Series,
        single_stock_tickers: list[str] | None = None,
    ) -> uuid.UUID:
        cycle_id = uuid.uuid4()
        nav = self.broker.get_account_nav()
        open_positions = self.broker.get_open_positions()

        regime = self.regime_detector.current_regime(market_returns) if len(market_returns) >= 30 else "unknown"
        drawdown_mtd = await equity_tracker.calc_drawdown_mtd(db, nav)
        halt_status = risk_engine.get_drawdown_status(drawdown_mtd)

        portfolio_var = self._calc_portfolio_var(nav, open_positions, price_data)
        var_breached = nav > 0 and portfolio_var / nav > settings.daily_var_95_max_pct

        gross_exposure = sum(abs(p["market_value"]) for p in open_positions) / nav if nav else 0.0
        net_exposure = sum(p["market_value"] for p in open_positions) / nav if nav else 0.0

        all_tickers = sorted({t for pair in candidate_pairs for t in pair})
        for pos in open_positions:
            all_tickers.append(pos["ticker"])
        sector_map = await data_pipeline.get_sector_map(db, sorted(set(all_tickers)))

        self._reconcile_open_positions(open_positions, candidate_pairs, price_data, nav)

        if halt_status == "HALTED":
            self.broker.flatten_all_positions()
            self._open_positions.clear()
            self._consecutive_breaks.clear()
            self._single_stock._open_singles.clear()
            notifier.notify_halt(
                f"Drawdown breached flatten threshold (MTD: {drawdown_mtd:.1%}). "
                f"All positions flattened. Agent halted."
            )
            await decision_logger.log_decision(
                db,
                DecisionLogSchema(
                    cycle_id=cycle_id,
                    pair_a="ALL",
                    pair_b="ALL",
                    action="EXIT",
                    regime=regime,
                    reasoning=f"Drawdown breached flatten threshold (MTD: {drawdown_mtd:.1%})",
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
            return cycle_id

        allow_new_entries = halt_status == "NORMAL" and regime == "range_bound" and not var_breached
        if var_breached:
            logger.warning("Portfolio VaR %.2f exceeds cap — new entries disallowed", portfolio_var)

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

        if settings.enable_single_stock and single_stock_tickers:
            pair_legs = {t for pair in candidate_pairs for t in pair}
            await self._single_stock.evaluate_universe(
                db,
                cycle_id,
                single_stock_tickers,
                price_data,
                nav,
                open_positions,
                regime,
                allow_new_entries,
                sector_map,
                pair_legs,
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
        return cycle_id

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
        pair_key = (pair_a, pair_b)
        series_a = price_data.get(pair_a, {}).get("close")
        series_b = price_data.get(pair_b, {}).get("close")

        if series_a is None or series_b is None or not self._is_fresh(price_data, pair_a, pair_b):
            await self._reject(db, cycle_id, pair_a, pair_b, regime, "Stale or missing price data")
            return

        try:
            coint_result = stats_engine.check_cointegration(series_a, series_b, pair_a, pair_b)
        except Exception:
            logger.exception("Cointegration failed for %s/%s", pair_a, pair_b)
            return

        # Held position: check exits FIRST — dollar stop and z-score stop before coint break.
        if pair_key in self._open_positions:
            await self._evaluate_exit(
                db, cycle_id, pair_a, pair_b, series_a, series_b, coint_result, regime
            )
            return

        if not coint_result.is_cointegrated:
            hl = coint_result.half_life_days
            hl_text = f"{hl:.1f}d" if np.isfinite(hl) else "not mean-reverting"
            await self._reject(
                db,
                cycle_id,
                pair_a,
                pair_b,
                regime,
                f"Failed cointegration: p={coint_result.p_value:.4f}, half_life={hl_text}",
                p_value=coint_result.p_value,
                half_life_days=coint_result.half_life_days,
            )
            return

        prev_ratio = self._prev_hedge_ratios.get(pair_key)
        hedge_result = stats_engine.compute_hedge_ratio_kalman(series_a, series_b, pair_a, pair_b, prev_ratio)

        if hedge_result.drift_pct > settings.hedge_drift_max_pct:
            # Do NOT commit the drifted ratio as the new baseline — otherwise a single
            # large drift resets the reference and the filter never fires again.
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
        self._prev_hedge_ratios[pair_key] = hedge_result.hedge_ratio

        blackout_a = await earnings.check_earnings_blackout(pair_a)
        blackout_b = await earnings.check_earnings_blackout(pair_b)
        if blackout_a or blackout_b:
            blocked = pair_a if blackout_a else pair_b
            await self._reject(
                db,
                cycle_id,
                pair_a,
                pair_b,
                regime,
                f"Earnings blackout active for {blocked}",
                hedge_ratio=hedge_result.hedge_ratio,
                p_value=coint_result.p_value,
                half_life_days=coint_result.half_life_days,
            )
            return

        spread = series_a - hedge_result.hedge_ratio * series_b
        z = stats_engine.calc_zscore(spread)

        if not np.isfinite(z):
            await self._reject(
                db,
                cycle_id,
                pair_a,
                pair_b,
                regime,
                "Z-score unavailable (flat or degenerate spread)",
                hedge_ratio=hedge_result.hedge_ratio,
                p_value=coint_result.p_value,
                half_life_days=coint_result.half_life_days,
            )
            return

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

        # Entry band: entry <= |z| < stop. Entering at |z| beyond the stop would be
        # stopped out immediately next cycle and signals a broken spread, not an edge.
        if abs(z) < settings.zscore_entry or abs(z) >= settings.zscore_stop:
            reason = (
                f"Z-score {z:.2f} below entry threshold"
                if abs(z) < settings.zscore_entry
                else f"Z-score {z:.2f} at/beyond stop threshold ({settings.zscore_stop}) - spread dislocated"
            )
            await self._reject(
                db,
                cycle_id,
                pair_a,
                pair_b,
                regime,
                reason,
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
        open_pair_count = self._count_open_pairs()

        # Per-name cap must hold for BOTH legs — leg B notional can exceed leg A
        # when the hedge ratio is large.
        risk_check = risk_engine.check_risk_limits(
            max(notional_a, notional_b),
            nav,
            max_sector_notional,
            gross_exposure,
            net_exposure,
            open_pair_count,
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

        self._open_positions[pair_key] = {
            "direction": "LONG_SPREAD" if action == "ENTER_LONG" else "SHORT_SPREAD",
            "entry_price_a": price_a,
            "entry_price_b": price_b,
            "hedge_ratio": hedge_result.hedge_ratio,
            # Hedge-aware stop: cut when the PAIR loses the risk budget, instead of
            # stopping on leg A's price alone (which fires on market-wide moves even
            # when the spread is intact).
            "max_loss_dollars": nav * settings.max_risk_pct_per_trade,
            "entry_date": datetime.now(timezone.utc),
            "half_life_days": coint_result.half_life_days,
            "entry_z": z,
            "shares_a": size,
            "shares_b": qty_b,
        }
        self._consecutive_breaks[pair_key] = 0

        notifier.notify_trade_signal(pair_a, pair_b, action, side_a, side_b, size, qty_b, z, hedge_result.hedge_ratio)

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
                reasoning=(
                    f"Z-score {z:.2f} triggered entry, p={coint_result.p_value:.4f}, "
                    f"ATR stop={stop_distance:.2f}"
                ),
                model_version=MODEL_VERSION,
                nav_at_decision=nav,
                position_size=notional_a,
            ),
        )
        logger.info("Executed %s on %s/%s: %s, %s", action, pair_a, pair_b, order_a, order_b)

    @staticmethod
    def _pair_unrealized_pnl(pos: dict, price_a: float, price_b: float) -> float:
        if pos["direction"] == "LONG_SPREAD":
            return float(
                pos["shares_a"] * (price_a - pos["entry_price_a"])
                - pos["shares_b"] * (price_b - pos["entry_price_b"])
            )
        return float(
            -pos["shares_a"] * (price_a - pos["entry_price_a"])
            + pos["shares_b"] * (price_b - pos["entry_price_b"])
        )

    def _hold_limit_days(self, half_life_days: float | None) -> float:
        """Time stop: N half-lives, capped at max_hold_days."""
        if half_life_days is None or not np.isfinite(half_life_days) or half_life_days <= 0:
            return float(settings.max_hold_days)
        return min(settings.time_stop_half_lives * half_life_days, float(settings.max_hold_days))

    async def _evaluate_exit(
        self, db, cycle_id, pair_a, pair_b, series_a, series_b, coint_result, regime
    ):
        pair_key = (pair_a, pair_b)
        pos = self._open_positions[pair_key]
        price_a_today = float(series_a.iloc[-1])
        price_b_today = float(series_b.iloc[-1])

        unrealized = self._pair_unrealized_pnl(pos, price_a_today, price_b_today)
        hit_pnl_stop = unrealized <= -pos["max_loss_dollars"]

        z = None
        if coint_result.is_cointegrated:
            try:
                hedge_result = stats_engine.compute_hedge_ratio_kalman(
                    series_a, series_b, pair_a, pair_b, self._prev_hedge_ratios.get(pair_key)
                )
                self._prev_hedge_ratios[pair_key] = hedge_result.hedge_ratio
                spread = series_a - hedge_result.hedge_ratio * series_b
                z = stats_engine.calc_zscore(spread)
                if not np.isfinite(z):
                    z = None
            except Exception:
                z = None

        hit_zscore_stop = z is not None and abs(z) >= settings.zscore_stop
        hit_target = z is not None and abs(z) <= settings.zscore_exit

        entry_date = pos.get("entry_date")
        held_days = (datetime.now(timezone.utc) - entry_date).total_seconds() / 86400 if entry_date else 0.0
        hit_time_stop = held_days >= self._hold_limit_days(pos.get("half_life_days"))

        if coint_result.is_cointegrated:
            self._consecutive_breaks[pair_key] = 0
        else:
            self._consecutive_breaks[pair_key] = self._consecutive_breaks.get(pair_key, 0) + 1

        hit_coint_break = self._consecutive_breaks.get(pair_key, 0) >= settings.consecutive_breaks_to_exit

        exit_reason = None
        if hit_pnl_stop:
            exit_reason = "pnl_stop"
        elif hit_zscore_stop:
            exit_reason = "zscore_stop"
        elif hit_target:
            exit_reason = "target"
        elif hit_coint_break:
            exit_reason = "cointegration_broke"
        elif hit_time_stop:
            exit_reason = "time_stop"

        if exit_reason is None:
            return

        side_a = "SELL" if pos["direction"] == "LONG_SPREAD" else "BUY"
        side_b = "BUY" if pos["direction"] == "LONG_SPREAD" else "SELL"

        order_a = self.broker.place_market_order(pair_a, pos["shares_a"], side_a)
        order_b = self.broker.place_market_order(pair_b, pos["shares_b"], side_b)

        commission = (pos["shares_a"] + pos["shares_b"]) * settings.commission_per_share * 2
        realized_pnl = float(unrealized - commission)

        await live_performance.record_closed_trade(
            db,
            strategy_type="PAIR",
            symbol_a=pair_a,
            symbol_b=pair_b,
            direction=pos["direction"],
            entry_price=pos["entry_price_a"],
            exit_price=price_a_today,
            shares=pos["shares_a"],
            realized_pnl=realized_pnl,
            exit_reason=exit_reason,
            z_entry=pos.get("entry_z"),
            z_exit=z,
            model_version=MODEL_VERSION,
        )

        notifier.notify_exit(pair_a, pair_b, exit_reason, z_score=z, realized_pnl=realized_pnl)

        await decision_logger.log_decision(
            db,
            DecisionLogSchema(
                cycle_id=cycle_id,
                pair_a=pair_a,
                pair_b=pair_b,
                action="EXIT",
                z_score=z,
                hedge_ratio=pos["hedge_ratio"],
                p_value=coint_result.p_value,
                half_life_days=coint_result.half_life_days,
                regime=regime,
                reasoning=f"Exit on {exit_reason}, realized_pnl=${realized_pnl:.2f}",
                model_version=MODEL_VERSION,
                nav_at_decision=self.broker.get_account_nav(),
            ),
        )
        logger.info("Exited %s/%s on %s: %s, %s", pair_a, pair_b, exit_reason, order_a, order_b)

        del self._open_positions[pair_key]
        self._consecutive_breaks.pop(pair_key, None)

    def _reconcile_open_positions(
        self,
        open_positions: list[dict],
        candidate_pairs: list[tuple[str, str]],
        price_data: dict,
        nav: float,
    ):
        """Drop stale in-memory entries; rebuild from broker if both legs exist."""
        broker_tickers = {p["ticker"] for p in open_positions}
        pos_by_ticker = {p["ticker"]: p for p in open_positions}

        for pair_key in list(self._open_positions.keys()):
            pair_a, pair_b = pair_key
            if pair_a not in broker_tickers or pair_b not in broker_tickers:
                logger.warning("Dropping stale in-memory position for %s/%s", pair_a, pair_b)
                self._open_positions.pop(pair_key, None)
                self._consecutive_breaks.pop(pair_key, None)

        for pair_a, pair_b in candidate_pairs:
            pair_key = (pair_a, pair_b)
            if pair_key in self._open_positions:
                continue
            if pair_a not in broker_tickers or pair_b not in broker_tickers:
                continue

            series_a = price_data.get(pair_a, {}).get("close")
            series_b = price_data.get(pair_b, {}).get("close")
            if series_a is None or series_b is None:
                continue

            leg_a = pos_by_ticker[pair_a]
            leg_b = pos_by_ticker[pair_b]
            long_a = leg_a["qty"] > 0
            direction = "LONG_SPREAD" if long_a else "SHORT_SPREAD"

            entry_a = float(leg_a["avg_entry_price"])
            entry_b = float(leg_b["avg_entry_price"])

            self._open_positions[pair_key] = {
                "direction": direction,
                "entry_price_a": entry_a,
                "entry_price_b": entry_b,
                "hedge_ratio": abs(float(leg_b["qty"])) / abs(float(leg_a["qty"])) if leg_a["qty"] else 1.0,
                "max_loss_dollars": nav * settings.max_risk_pct_per_trade,
                # Entry date unknown after restart — conservatively start the clock now.
                "entry_date": datetime.now(timezone.utc),
                "half_life_days": None,
                "entry_z": None,
                "shares_a": abs(float(leg_a["qty"])),
                "shares_b": abs(float(leg_b["qty"])),
            }
            self._consecutive_breaks.setdefault(pair_key, 0)
            logger.info("Reconciled open position from broker for %s/%s", pair_a, pair_b)

    def _count_open_pairs(self) -> int:
        return len(self._open_positions)

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
            new_returns, existing_returns, settings.correlation_threshold
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
