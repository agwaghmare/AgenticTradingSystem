"""
Backtest script for the mean-reversion pairs strategy.

Replays the SAME logic the live agent uses (app/services/stats_engine.py),
day by day, over historical data — using only data available up to each
simulated "today" (no lookahead).

Portfolio mode enforces max concurrent positions (like live) and deducts
per-share commission. Per-pair mode is available for discovery runs.

Run from the project root:
    python scripts/backtest.py
    python scripts/backtest.py --universe full
    python scripts/backtest.py --universe filtered --per-pair
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import warnings

warnings.filterwarnings("ignore")

import time
from datetime import datetime

import httpx
import numpy as np
import pandas as pd
import yfinance as yf

from app.config import settings
from app.services import stats_engine, risk_engine
from app.pairs_universe import (
    FULL_CANDIDATE_PAIRS,
    HIGH_CONVICTION_PAIRS,
    SINGLE_STOCK_UNIVERSE,
    generate_candidate_pairs,
)

# ---------------------------------------------------------------------
# CONFIG — thresholds from app/config.py; universe via CLI
# ---------------------------------------------------------------------

START_DATE = "2021-01-01"
END_DATE = "2026-01-01"

LOOKBACK_WINDOW = 100
ZSCORE_LOOKBACK = 20
STEP_DAYS = 1

COINTEGRATION_PVALUE_MAX = settings.cointegration_pvalue_max
HALF_LIFE_MIN_DAYS = settings.half_life_min_days
HALF_LIFE_MAX_DAYS = settings.half_life_max_days
ZSCORE_ENTRY = settings.zscore_entry
ZSCORE_EXIT = settings.zscore_exit
ZSCORE_STOP = settings.zscore_stop
HEDGE_DRIFT_MAX_PCT = settings.hedge_drift_max_pct
CONSECUTIVE_BREAKS_TO_EXIT = settings.consecutive_breaks_to_exit
COMMISSION_PER_SHARE = settings.commission_per_share
TIME_STOP_HALF_LIVES = settings.time_stop_half_lives
MAX_HOLD_DAYS = settings.max_hold_days

INITIAL_CAPITAL = 100_000
RISK_PCT_PER_TRADE = settings.max_risk_pct_per_trade
STOP_LOSS_PCT = 0.03
MAX_CONCURRENT_POSITIONS = settings.max_concurrent_positions
MAX_SINGLE_POSITIONS = settings.max_single_stock_positions


def _hold_limit_days(half_life: float | None) -> float:
    """Time stop: N half-lives, capped at MAX_HOLD_DAYS."""
    if half_life is None or not np.isfinite(half_life) or half_life <= 0:
        return float(MAX_HOLD_DAYS)
    return min(TIME_STOP_HALF_LIVES * half_life, float(MAX_HOLD_DAYS))


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------
def _load_via_chart_api(ticker: str) -> pd.Series:
    start_ts = int(datetime.strptime(START_DATE, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(END_DATE, "%Y-%m-%d").timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    response = httpx.get(
        url,
        params={"period1": start_ts, "period2": end_ts, "interval": "1d", "events": "history"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()["chart"]["result"][0]
    timestamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    data = {
        pd.Timestamp(ts, unit="s", tz="UTC").tz_localize(None): close
        for ts, close in zip(timestamps, closes)
        if close is not None
    }
    return pd.Series(data).sort_index()


def load_price_data(tickers: list[str]) -> dict[str, pd.Series]:
    print(f"Downloading {len(tickers)} tickers from {START_DATE} to {END_DATE}...")
    series_map: dict[str, pd.Series] = {}
    for i, ticker in enumerate(tickers):
        if i > 0 and i % 10 == 0:
            print(f"  ... {i}/{len(tickers)} tickers loaded")
        if i > 0:
            time.sleep(0.35)
        loaded = False
        for attempt in range(3):
            try:
                s = _load_via_chart_api(ticker)
                if not s.empty:
                    series_map[ticker] = s
                    loaded = True
                    break
            except Exception as exc:
                if attempt == 2:
                    print(f"  chart API failed for {ticker}: {exc}")
                time.sleep(1 * (attempt + 1))
        if not loaded:
            try:
                data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
                if not data.empty:
                    close_col = "Close" if "Close" in data.columns else "close"
                    series_map[ticker] = data[close_col].dropna()
                    loaded = True
            except Exception:
                pass
        if not loaded:
            print(f"  WARNING: no data for {ticker}")
    return series_map


# ---------------------------------------------------------------------
# SIMULATED TRADE STATE
# ---------------------------------------------------------------------
class PairPosition:
    def __init__(self):
        self.is_open = False
        self.direction = None
        self.entry_z = None
        self.entry_date = None
        self.entry_spread = None
        self.hedge_ratio_at_entry = None
        self.shares_a = None
        self.shares_b = None
        self.entry_price_a = None
        self.entry_price_b = None
        self.max_loss_dollars = None
        self.half_life_at_entry = None
        self.consecutive_breaks = 0


def _trade_log_row(pair_a: str, pair_b: str, today, reason: str, pnl: float, position: PairPosition, exit_z) -> dict:
    return {
        "pair": f"{pair_a}/{pair_b}",
        "exit_date": today,
        "reason": reason,
        "pnl": pnl,
        "exit_z": exit_z,
        "shares_a": position.shares_a,
        "shares_b": position.shares_b,
    }


def _precompute_pair_signals(
    pair_a: str,
    pair_b: str,
    common_idx: pd.Index,
    series_a_full: pd.Series,
    series_b_full: pd.Series,
) -> dict[pd.Timestamp, dict]:
    """Walk-forward signals for one pair — computed once, reused in portfolio sim."""
    prev_hedge_ratio = None
    timeline: dict[pd.Timestamp, dict] = {}

    for i in range(LOOKBACK_WINDOW + ZSCORE_LOOKBACK, len(common_idx), STEP_DAYS):
        today = common_idx[i]
        window_a = series_a_full.iloc[max(0, i - LOOKBACK_WINDOW) : i + 1]
        window_b = series_b_full.iloc[max(0, i - LOOKBACK_WINDOW) : i + 1]

        row = {
            "price_a": float(window_a.iloc[-1]),
            "price_b": float(window_b.iloc[-1]),
            "z": None,
            "is_cointegrated": False,
            "hedge_ratio": None,
            "drift_pct": 0.0,
            "half_life": None,
            "entry_ok": False,
        }

        try:
            coint_result = stats_engine.check_cointegration(window_a, window_b, pair_a, pair_b)
            row["is_cointegrated"] = coint_result.is_cointegrated
            row["half_life"] = coint_result.half_life_days
        except Exception:
            timeline[today] = row
            continue

        if coint_result.is_cointegrated:
            try:
                hedge_result = stats_engine.compute_hedge_ratio_kalman(
                    window_a, window_b, pair_a, pair_b, prev_hedge_ratio
                )
                prev_hedge_ratio = hedge_result.hedge_ratio
                row["hedge_ratio"] = hedge_result.hedge_ratio
                row["drift_pct"] = hedge_result.drift_pct
                spread = window_a - hedge_result.hedge_ratio * window_b
                if len(spread) >= ZSCORE_LOOKBACK:
                    z_val = stats_engine.calc_zscore(spread, lookback=ZSCORE_LOOKBACK)
                    if not np.isnan(z_val):
                        row["z"] = float(z_val)
            except Exception:
                pass

        # Entry band: entry <= |z| < stop (entering beyond the stop is a dislocation)
        if (
            row["is_cointegrated"]
            and row["z"] is not None
            and ZSCORE_ENTRY <= abs(row["z"]) < ZSCORE_STOP
            and row["drift_pct"] <= HEDGE_DRIFT_MAX_PCT
            and row["hedge_ratio"] is not None
        ):
            row["entry_ok"] = True

        timeline[today] = row

    return timeline


def _open_position_from_signal(signal: dict, z: float, today: pd.Timestamp) -> PairPosition:
    position = PairPosition()
    position.is_open = True
    position.direction = "LONG_SPREAD" if z < 0 else "SHORT_SPREAD"
    position.entry_z = z
    position.entry_date = today
    position.hedge_ratio_at_entry = signal["hedge_ratio"]
    position.entry_price_a = signal["price_a"]
    position.entry_price_b = signal["price_b"]
    position.half_life_at_entry = signal.get("half_life")
    position.consecutive_breaks = 0

    stop_distance = signal["price_a"] * STOP_LOSS_PCT
    shares_a = risk_engine.calc_position_size(
        INITIAL_CAPITAL, RISK_PCT_PER_TRADE, stop_distance, signal["price_a"]
    )
    position.shares_a = shares_a
    position.shares_b = shares_a * signal["hedge_ratio"]
    # Hedge-aware stop: cut when the PAIR loses the risk budget, not when leg A
    # alone moves (market-wide moves shouldn't stop out an intact spread).
    position.max_loss_dollars = INITIAL_CAPITAL * RISK_PCT_PER_TRADE
    return position


def _try_exit_position(
    pair_a: str,
    pair_b: str,
    today: pd.Timestamp,
    position: PairPosition,
    signal: dict,
) -> tuple[PairPosition | None, dict | None, float]:
    price_a_today = signal["price_a"]
    price_b_today = signal["price_b"]
    z = signal["z"]

    hit_pnl_stop = _check_pnl_stop(position, price_a_today, price_b_today)
    hit_zscore_stop = z is not None and abs(z) >= ZSCORE_STOP
    hit_target = z is not None and abs(z) <= ZSCORE_EXIT
    hit_time_stop = _check_time_stop(position, today)

    if not signal["is_cointegrated"]:
        position.consecutive_breaks += 1
    else:
        position.consecutive_breaks = 0

    hit_coint_break = position.consecutive_breaks >= CONSECUTIVE_BREAKS_TO_EXIT

    if hit_pnl_stop or hit_zscore_stop:
        pnl = _close_position(position, price_a_today, price_b_today)
        reason = "pnl_stop" if hit_pnl_stop else "zscore_stop"
        return None, _trade_log_row(pair_a, pair_b, today, reason, pnl, position, z), pnl
    if hit_target:
        pnl = _close_position(position, price_a_today, price_b_today)
        return None, _trade_log_row(pair_a, pair_b, today, "target", pnl, position, z), pnl
    if hit_coint_break:
        pnl = _close_position(position, price_a_today, price_b_today)
        return None, _trade_log_row(pair_a, pair_b, today, "cointegration_broke", pnl, position, z), pnl
    if hit_time_stop:
        pnl = _close_position(position, price_a_today, price_b_today)
        return None, _trade_log_row(pair_a, pair_b, today, "time_stop", pnl, position, z), pnl

    return position, None, 0.0


def _pair_unrealized(position: PairPosition, price_a: float, price_b: float) -> float:
    if position.direction == "LONG_SPREAD":
        return float(
            position.shares_a * (price_a - position.entry_price_a)
            - position.shares_b * (price_b - position.entry_price_b)
        )
    return float(
        -position.shares_a * (price_a - position.entry_price_a)
        + position.shares_b * (price_b - position.entry_price_b)
    )


def _check_pnl_stop(position: PairPosition, price_a_today: float, price_b_today: float) -> bool:
    if position.max_loss_dollars is None:
        return False
    return _pair_unrealized(position, price_a_today, price_b_today) <= -position.max_loss_dollars


def _check_time_stop(position: PairPosition, today: pd.Timestamp) -> bool:
    if position.entry_date is None:
        return False
    held_days = (today - position.entry_date).days
    return held_days >= _hold_limit_days(position.half_life_at_entry)


# ---------------------------------------------------------------------
# SINGLE-STOCK MEAN REVERSION (mirrors app/agents/single_stock.py)
# ---------------------------------------------------------------------
class SinglePosition:
    def __init__(self):
        self.direction = None  # LONG | SHORT
        self.entry_price = None
        self.shares = None
        self.dollar_stop = None
        self.entry_z = None
        self.entry_date = None
        self.half_life_at_entry = None


def _precompute_single_signals(ticker: str, series: pd.Series) -> dict[pd.Timestamp, dict]:
    """Walk-forward single-name mean-reversion signals (log-price z + OU half-life)."""
    timeline: dict[pd.Timestamp, dict] = {}
    for i in range(LOOKBACK_WINDOW + ZSCORE_LOOKBACK, len(series), STEP_DAYS):
        today = series.index[i]
        window = series.iloc[max(0, i - LOOKBACK_WINDOW) : i + 1]
        row = {"price": float(window.iloc[-1]), "z": None, "half_life": None, "is_mr": False, "entry_ok": False}
        try:
            mr = stats_engine.check_price_mean_reversion(window, ticker)
        except Exception:
            timeline[today] = row
            continue
        row["z"] = float(mr.z_score) if np.isfinite(mr.z_score) else None
        row["half_life"] = mr.half_life_days
        row["is_mr"] = mr.is_mean_reverting
        if mr.is_mean_reverting and row["z"] is not None and ZSCORE_ENTRY <= abs(row["z"]) < ZSCORE_STOP:
            row["entry_ok"] = True
        timeline[today] = row
    return timeline


def _open_single_from_signal(signal: dict, today: pd.Timestamp) -> SinglePosition:
    z = signal["z"]
    price = signal["price"]
    pos = SinglePosition()
    pos.direction = "LONG" if z < 0 else "SHORT"
    stop_distance = price * STOP_LOSS_PCT
    pos.shares = risk_engine.calc_position_size(INITIAL_CAPITAL, RISK_PCT_PER_TRADE, stop_distance, price)
    pos.entry_price = price
    pos.dollar_stop = price - stop_distance if pos.direction == "LONG" else price + stop_distance
    pos.entry_z = z
    pos.entry_date = today
    pos.half_life_at_entry = signal["half_life"]
    return pos


def _close_single(pos: SinglePosition, price: float) -> float:
    if pos.direction == "LONG":
        gross = pos.shares * (price - pos.entry_price)
    else:
        gross = pos.shares * (pos.entry_price - price)
    commission = pos.shares * COMMISSION_PER_SHARE * 2
    return float(gross - commission)


def _try_exit_single(
    ticker: str, today: pd.Timestamp, pos: SinglePosition, signal: dict
) -> tuple[SinglePosition | None, dict | None, float]:
    price = signal["price"]
    z = signal["z"]

    hit_dollar_stop = price <= pos.dollar_stop if pos.direction == "LONG" else price >= pos.dollar_stop
    hit_zscore_stop = z is not None and abs(z) >= ZSCORE_STOP
    hit_target = z is not None and abs(z) <= ZSCORE_EXIT
    held_days = (today - pos.entry_date).days
    hit_time_stop = held_days >= _hold_limit_days(pos.half_life_at_entry)

    exit_reason = None
    if hit_dollar_stop:
        exit_reason = "dollar_stop"
    elif hit_zscore_stop:
        exit_reason = "zscore_stop"
    elif hit_target:
        exit_reason = "target"
    elif not signal["is_mr"]:
        exit_reason = "mean_reversion_broke"
    elif hit_time_stop:
        exit_reason = "time_stop"

    if exit_reason is None:
        return pos, None, 0.0

    pnl = _close_single(pos, price)
    entry = {
        "pair": f"{ticker}/SINGLE",
        "exit_date": today,
        "reason": exit_reason,
        "pnl": pnl,
        "exit_z": z,
        "shares_a": pos.shares,
        "shares_b": 0.0,
    }
    return None, entry, pnl


def _close_position(position: PairPosition, price_a: float, price_b: float) -> float:
    gross = _pair_unrealized(position, price_a, price_b)
    commission = (position.shares_a + position.shares_b) * COMMISSION_PER_SHARE * 2
    return float(gross - commission)


def _evaluate_pair_day(
    pair_a: str,
    pair_b: str,
    i: int,
    common_idx: pd.Index,
    series_a_full: pd.Series,
    series_b_full: pd.Series,
    position: PairPosition,
    prev_hedge_ratio: float | None,
) -> tuple[PairPosition, float | None, dict | None, float, int]:
    """Returns (position, prev_hedge_ratio, trade_log_entry, pnl_delta, trades_entered)."""
    today = common_idx[i]
    window_a = series_a_full.iloc[max(0, i - LOOKBACK_WINDOW) : i + 1]
    window_b = series_b_full.iloc[max(0, i - LOOKBACK_WINDOW) : i + 1]

    try:
        coint_result = stats_engine.check_cointegration(window_a, window_b, pair_a, pair_b)
    except Exception:
        return position, prev_hedge_ratio, None, 0.0, 0

    price_a_today = window_a.iloc[-1]
    price_b_today = window_b.iloc[-1]

    z = None
    hedge_result = None
    if coint_result.is_cointegrated:
        try:
            hedge_result = stats_engine.compute_hedge_ratio_kalman(
                window_a, window_b, pair_a, pair_b, prev_hedge_ratio
            )
            prev_hedge_ratio = hedge_result.hedge_ratio
            spread = window_a - hedge_result.hedge_ratio * window_b
            if len(spread) >= ZSCORE_LOOKBACK:
                z_val = stats_engine.calc_zscore(spread, lookback=ZSCORE_LOOKBACK)
                if not np.isnan(z_val):
                    z = z_val
        except Exception:
            pass

    if position.is_open:
        hit_pnl_stop = _check_pnl_stop(position, price_a_today, price_b_today)
        hit_zscore_stop = z is not None and abs(z) >= ZSCORE_STOP
        hit_target = z is not None and abs(z) <= ZSCORE_EXIT
        hit_time_stop = _check_time_stop(position, today)

        if not coint_result.is_cointegrated:
            position.consecutive_breaks += 1
        else:
            position.consecutive_breaks = 0

        hit_coint_break = position.consecutive_breaks >= CONSECUTIVE_BREAKS_TO_EXIT

        if hit_pnl_stop or hit_zscore_stop:
            pnl = _close_position(position, price_a_today, price_b_today)
            reason = "pnl_stop" if hit_pnl_stop else "zscore_stop"
            entry = _trade_log_row(pair_a, pair_b, today, reason, pnl, position, z)
            return PairPosition(), prev_hedge_ratio, entry, pnl, 0
        if hit_target:
            pnl = _close_position(position, price_a_today, price_b_today)
            entry = _trade_log_row(pair_a, pair_b, today, "target", pnl, position, z)
            return PairPosition(), prev_hedge_ratio, entry, pnl, 0
        if hit_coint_break:
            pnl = _close_position(position, price_a_today, price_b_today)
            entry = _trade_log_row(pair_a, pair_b, today, "cointegration_broke", pnl, position, z)
            return PairPosition(), prev_hedge_ratio, entry, pnl, 0
        if hit_time_stop:
            pnl = _close_position(position, price_a_today, price_b_today)
            entry = _trade_log_row(pair_a, pair_b, today, "time_stop", pnl, position, z)
            return PairPosition(), prev_hedge_ratio, entry, pnl, 0

        return position, prev_hedge_ratio, None, 0.0, 0

    if (
        coint_result.is_cointegrated
        and hedge_result is not None
        and z is not None
        and ZSCORE_ENTRY <= abs(z) < ZSCORE_STOP
        and hedge_result.drift_pct <= HEDGE_DRIFT_MAX_PCT
    ):
        position = PairPosition()
        position.is_open = True
        position.direction = "LONG_SPREAD" if z < 0 else "SHORT_SPREAD"
        position.entry_z = z
        position.entry_date = today
        position.hedge_ratio_at_entry = hedge_result.hedge_ratio
        position.entry_price_a = price_a_today
        position.entry_price_b = price_b_today
        position.half_life_at_entry = coint_result.half_life_days
        position.consecutive_breaks = 0

        stop_distance = price_a_today * STOP_LOSS_PCT
        shares_a = risk_engine.calc_position_size(
            INITIAL_CAPITAL, RISK_PCT_PER_TRADE, stop_distance, price_a_today
        )
        position.shares_a = shares_a
        position.shares_b = shares_a * hedge_result.hedge_ratio
        position.max_loss_dollars = INITIAL_CAPITAL * RISK_PCT_PER_TRADE
        return position, prev_hedge_ratio, None, 0.0, 1

    return position, prev_hedge_ratio, None, 0.0, 0


def run_backtest(
    candidate_pairs: list[tuple[str, str]] | None = None,
    portfolio_mode: bool = True,
    single_tickers: list[str] | None = None,
):
    pairs = candidate_pairs or FULL_CANDIDATE_PAIRS
    singles = single_tickers or []
    tickers = sorted({t for pair in pairs for t in pair} | set(singles))
    all_series = load_price_data(tickers)
    mode_label = "portfolio" if portfolio_mode else "per-pair"
    print(
        f"Running walk-forward backtest ({mode_label}) on {len(pairs)} pairs "
        f"+ {len(singles)} single-name MR tickers — "
        f"z_entry={ZSCORE_ENTRY}, stop={STOP_LOSS_PCT:.0%}, lookback={LOOKBACK_WINDOW}, "
        f"max_positions={MAX_CONCURRENT_POSITIONS}, max_singles={MAX_SINGLE_POSITIONS}"
    )

    if portfolio_mode:
        return _run_portfolio_backtest(pairs, all_series, singles)
    return _run_per_pair_backtest(pairs, all_series)


def _run_per_pair_backtest(pairs: list[tuple[str, str]], all_series: dict[str, pd.Series]):
    results = []
    trade_log = []

    for pair_idx, (pair_a, pair_b) in enumerate(pairs, start=1):
        if pair_a not in all_series or pair_b not in all_series:
            continue

        if pair_idx % 25 == 0:
            print(f"  pair {pair_idx}/{len(pairs)}: {pair_a}/{pair_b}")

        series_a_full = all_series[pair_a]
        series_b_full = all_series[pair_b]
        common_idx = series_a_full.index.intersection(series_b_full.index)
        series_a_full = series_a_full.loc[common_idx]
        series_b_full = series_b_full.loc[common_idx]

        if len(common_idx) < LOOKBACK_WINDOW + ZSCORE_LOOKBACK + 10:
            continue

        position = PairPosition()
        prev_hedge_ratio = None
        pair_trades = 0
        pair_signals_evaluated = 0
        equity_curve = []
        pnl_total = 0.0

        for i in range(LOOKBACK_WINDOW + ZSCORE_LOOKBACK, len(common_idx), STEP_DAYS):
            pair_signals_evaluated += 1
            position, prev_hedge_ratio, entry, pnl_delta, entered = _evaluate_pair_day(
                pair_a, pair_b, i, common_idx, series_a_full, series_b_full, position, prev_hedge_ratio
            )
            if entry:
                trade_log.append(entry)
            pnl_total += pnl_delta
            pair_trades += entered
            equity_curve.append({"date": common_idx[i], "cum_pnl": pnl_total})

        if pair_trades > 0 or pnl_total != 0:
            results.append(
                {
                    "pair": f"{pair_a}/{pair_b}",
                    "signals_evaluated": pair_signals_evaluated,
                    "trades_entered": pair_trades,
                    "total_pnl": pnl_total,
                    "equity_curve": equity_curve,
                }
            )

    _print_summary(results, trade_log, portfolio_mode=False)
    return results, trade_log


def _run_portfolio_backtest(
    pairs: list[tuple[str, str]],
    all_series: dict[str, pd.Series],
    single_tickers: list[str] | None = None,
):
    timelines: dict[tuple[str, str], dict[pd.Timestamp, dict]] = {}
    all_dates: set[pd.Timestamp] = set()

    for pair_idx, (pair_a, pair_b) in enumerate(pairs, start=1):
        if pair_a not in all_series or pair_b not in all_series:
            continue
        series_a = all_series[pair_a]
        series_b = all_series[pair_b]
        common_idx = series_a.index.intersection(series_b.index)
        if len(common_idx) < LOOKBACK_WINDOW + ZSCORE_LOOKBACK + 10:
            continue

        if pair_idx % 10 == 0:
            print(f"  precomputing signals {pair_idx}/{len(pairs)}: {pair_a}/{pair_b}")

        timeline = _precompute_pair_signals(
            pair_a, pair_b, common_idx, series_a.loc[common_idx], series_b.loc[common_idx]
        )
        timelines[(pair_a, pair_b)] = timeline
        all_dates.update(timeline.keys())

    single_timelines: dict[str, dict[pd.Timestamp, dict]] = {}
    for ticker in single_tickers or []:
        series = all_series.get(ticker)
        if series is None or len(series) < LOOKBACK_WINDOW + ZSCORE_LOOKBACK + 10:
            continue
        single_timelines[ticker] = _precompute_single_signals(ticker, series)
        all_dates.update(single_timelines[ticker].keys())
    if single_timelines:
        print(f"  precomputed single-name MR signals for {len(single_timelines)} tickers")

    sorted_dates = sorted(all_dates)
    print(f"  simulating portfolio over {len(sorted_dates)} trading days...")

    positions: dict[tuple[str, str], PairPosition] = {}
    single_positions: dict[str, SinglePosition] = {}
    trade_log: list[dict] = []
    pair_stats: dict[str, dict] = {}
    daily_pnl: list[tuple[pd.Timestamp, float]] = []
    cum_pnl = 0.0

    def _stats_for(key: str) -> dict:
        return pair_stats.setdefault(
            key,
            {"pair": key, "signals_evaluated": 0, "trades_entered": 0, "total_pnl": 0.0, "equity_curve": []},
        )

    for day_num, today in enumerate(sorted_dates):
        if day_num % 250 == 0 and day_num > 0:
            print(
                f"  day {day_num}/{len(sorted_dates)} — open pairs: {len(positions)}, "
                f"open singles: {len(single_positions)}"
            )

        day_pnl = 0.0

        for pair_key in list(positions.keys()):
            timeline = timelines.get(pair_key)
            if timeline is None or today not in timeline:
                continue
            pair_a, pair_b = pair_key
            position, entry, pnl_delta = _try_exit_position(
                pair_a, pair_b, today, positions[pair_key], timeline[today]
            )
            if entry:
                trade_log.append(entry)
                day_pnl += pnl_delta
                _stats_for(entry["pair"])["total_pnl"] += entry["pnl"]
                positions.pop(pair_key, None)
            elif position is not None:
                positions[pair_key] = position
            else:
                positions.pop(pair_key, None)

        for ticker in list(single_positions.keys()):
            timeline = single_timelines.get(ticker)
            if timeline is None or today not in timeline:
                continue
            pos, entry, pnl_delta = _try_exit_single(ticker, today, single_positions[ticker], timeline[today])
            if entry:
                trade_log.append(entry)
                day_pnl += pnl_delta
                _stats_for(entry["pair"])["total_pnl"] += entry["pnl"]
                single_positions.pop(ticker, None)
            elif pos is None:
                single_positions.pop(ticker, None)

        open_slots = MAX_CONCURRENT_POSITIONS - len(positions)
        if open_slots > 0:
            entry_candidates: list[tuple[float, tuple[str, str], PairPosition]] = []
            for pair_key, timeline in timelines.items():
                if pair_key in positions or today not in timeline:
                    continue
                signal = timeline[today]
                if signal["entry_ok"] and signal["z"] is not None:
                    entry_candidates.append(
                        (abs(signal["z"]), pair_key, _open_position_from_signal(signal, signal["z"], today))
                    )

            entry_candidates.sort(key=lambda x: x[0], reverse=True)
            for _, pair_key, position in entry_candidates[:open_slots]:
                positions[pair_key] = position
                _stats_for(f"{pair_key[0]}/{pair_key[1]}")["trades_entered"] += 1

        single_slots = MAX_SINGLE_POSITIONS - len(single_positions)
        if single_slots > 0 and single_timelines:
            # Skip tickers already exposed via an open pair leg (mirrors live agent).
            pair_leg_exposure = {t for pair_key in positions for t in pair_key}
            single_candidates: list[tuple[float, str]] = []
            for ticker, timeline in single_timelines.items():
                if ticker in single_positions or ticker in pair_leg_exposure or today not in timeline:
                    continue
                signal = timeline[today]
                if signal["entry_ok"] and signal["z"] is not None:
                    single_candidates.append((abs(signal["z"]), ticker))

            single_candidates.sort(key=lambda x: x[0], reverse=True)
            for _, ticker in single_candidates[:single_slots]:
                single_positions[ticker] = _open_single_from_signal(single_timelines[ticker][today], today)
                _stats_for(f"{ticker}/SINGLE")["trades_entered"] += 1

        cum_pnl += day_pnl
        daily_pnl.append((today, day_pnl))

    results = [s for s in pair_stats.values() if s["trades_entered"] > 0 or s["total_pnl"] != 0]
    _print_summary(results, trade_log, portfolio_mode=True, daily_pnl=daily_pnl, cum_pnl=cum_pnl)
    return results, trade_log


def calc_portfolio_sharpe(trade_log: list[dict], initial_capital: float = INITIAL_CAPITAL) -> float:
    if len(trade_log) < 5:
        return 0.0
    pnls = [t["pnl"] for t in trade_log]
    returns = [p / initial_capital for p in pnls]
    mean_r = np.mean(returns)
    std_r = np.std(returns)
    if std_r == 0:
        return 0.0
    return float((mean_r / std_r) * np.sqrt(252))


def _print_summary(
    results: list[dict],
    trade_log: list[dict],
    portfolio_mode: bool = False,
    daily_pnl: list | None = None,
    cum_pnl: float | None = None,
):
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY" + (" (PORTFOLIO MODE)" if portfolio_mode else ""))
    print("=" * 60)

    total_pnl = cum_pnl if cum_pnl is not None else sum(r["total_pnl"] for r in results)
    total_trades = len(trade_log)

    top = sorted(results, key=lambda r: r["total_pnl"], reverse=True)[:10]
    bottom = sorted(results, key=lambda r: r["total_pnl"])[:10]

    print(f"Pairs with activity: {len(results)}")
    print("\nTop 10 pairs by P&L:")
    for r in top:
        print(f"  {r['pair']}: ${r['total_pnl']:,.2f} ({r.get('trades_entered', '?')} trades)")

    if bottom:
        print("\nBottom 10 pairs by P&L:")
        for r in bottom:
            print(f"  {r['pair']}: ${r['total_pnl']:,.2f} ({r.get('trades_entered', '?')} trades)")

    print("\n" + "-" * 60)
    print(f"TOTAL trades:                  {total_trades}")
    print(f"TOTAL P&L (after commission):  ${total_pnl:,.2f}")
    print(f"Return on initial capital:     {(total_pnl / INITIAL_CAPITAL) * 100:.2f}%")

    if trade_log:
        wins = [t for t in trade_log if t["pnl"] > 0]
        losses = [t for t in trade_log if t["pnl"] <= 0]
        win_rate = len(wins) / len(trade_log) * 100
        print(f"Win rate:                      {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")

        trade_sharpe = calc_portfolio_sharpe(trade_log)
        print(f"Trade-level Sharpe (ann.):     {trade_sharpe:.3f}")

        if daily_pnl:
            # Include ALL trading days — dropping zero-PnL days overstates Sharpe.
            daily_returns = [p / INITIAL_CAPITAL for _, p in daily_pnl]
            if len(daily_returns) >= 5:
                d_mean = np.mean(daily_returns)
                d_std = np.std(daily_returns)
                daily_sharpe = (d_mean / d_std) * np.sqrt(252) if d_std > 0 else 0.0
                print(f"Daily P&L Sharpe (ann.):       {daily_sharpe:.3f}")

        print("\nExit reason breakdown:")
        reasons: dict[str, int] = {}
        for t in trade_log:
            reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    print("\n" + "=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Pairs strategy backtest")
    parser.add_argument(
        "--universe",
        choices=["full", "filtered"],
        default="full",
        help="Pair universe: all sector pairs or high-conviction only",
    )
    parser.add_argument(
        "--per-pair",
        action="store_true",
        help="Run independent per-pair backtest (no position cap)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shorter date range (2023-2025) for fast validation",
    )
    parser.add_argument(
        "--no-singles",
        action="store_true",
        help="Exclude single-stock mean-reversion signals (pairs only)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        globals()["START_DATE"] = "2023-01-01"
        globals()["END_DATE"] = "2025-01-01"
    if args.universe == "filtered":
        pairs = HIGH_CONVICTION_PAIRS
    else:
        pairs = FULL_CANDIDATE_PAIRS

    singles = [] if (args.no_singles or args.per_pair) else SINGLE_STOCK_UNIVERSE
    print(f"Candidate pairs: {len(pairs)}, single-name MR tickers: {len(singles)}")
    results, trade_log = run_backtest(pairs, portfolio_mode=not args.per_pair, single_tickers=singles)

    if trade_log:
        df = pd.DataFrame(trade_log)
        out_path = os.path.join(os.path.dirname(__file__), "backtest_trade_log.csv")
        df.to_csv(out_path, index=False)
        print(f"\nFull trade log written to: {out_path}")
