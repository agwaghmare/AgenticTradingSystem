"""
Backtest script for the mean-reversion pairs strategy.

Replays the SAME logic the live agent uses (app/services/stats_engine.py),
day by day, over historical data — using only data available up to each
simulated "today" (no lookahead). Outputs: how many signals fired, what
the hypothetical trades would have looked like, and simple P&L stats.

This does NOT touch Alpaca, Robinhood, or any broker. It only uses yfinance
for historical data and reuses your real stats_engine.py logic, so a good
backtest result here means the actual agent logic is producing sane
signals — not a separate, simplified approximation of it.

Run from the project root:
    python scripts/backtest.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import time
from datetime import datetime
from itertools import combinations

import httpx
import numpy as np
import pandas as pd
import yfinance as yf

from app.services import stats_engine, risk_engine

# ---------------------------------------------------------------------
# CONFIG — edit these to match your candidate universe / thresholds
# ---------------------------------------------------------------------
SECTOR_UNIVERSE = {
    "consumer_staples": ["KO", "PEP", "PG", "CL", "COST", "WMT", "KMB", "CLX", "GIS", "KHC"],
    "energy": ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "OXY", "HES"],
    "payments": ["V", "MA", "PYPL", "FIS", "FI", "GPN"],
    "retail": ["HD", "LOW", "TGT", "WMT", "TJX", "ROST", "DG", "DLTR", "BBY"],
    "banks": ["JPM", "BAC", "GS", "MS", "WFC", "USB", "PNC", "TFC", "C", "COF"],
    "healthcare": ["JNJ", "PFE", "UNH", "CVS", "MRK", "ABBV", "BMY", "LLY", "CI", "HUM"],
    "semis": ["QCOM", "AVGO", "TXN", "ADI", "MU", "NXPI", "MCHP", "ON"],
    "airlines": ["DAL", "UAL", "AAL", "LUV", "ALK", "JBLU"],
    "telecom": ["VZ", "T", "TMUS"],
    "industrials": ["HON", "GE", "MMM", "EMR", "ITW", "ETN", "PH", "ROK"],
    "reits": ["O", "SPG", "PLD", "AMT", "EQIX", "PSA", "DLR", "WELL"],
    "insurance": ["TRV", "ALL", "PGR", "CB", "AIG", "MET", "PRU"],
    "asset_managers": ["BLK", "BX", "KKR", "APO", "TROW", "BEN"],
    "autos": ["GM", "F", "STLA"],
    "beverages_alcohol": ["STZ", "BF-B", "TAP"],
}


def generate_candidate_pairs(sector_universe: dict[str, list[str]]) -> list[tuple[str, str]]:
    """All unique intra-sector combinations — cointegration is only economically
    plausible within a sector, so we don't generate cross-sector pairs."""
    pairs = []
    for sector, tickers in sector_universe.items():
        for a, b in combinations(sorted(set(tickers)), 2):
            pairs.append((a, b))
    return pairs


CANDIDATE_PAIRS = generate_candidate_pairs(SECTOR_UNIVERSE)

START_DATE = "2021-01-01"
END_DATE = "2026-01-01"

LOOKBACK_WINDOW = 100
ZSCORE_LOOKBACK = 20
STEP_DAYS = 1

COINTEGRATION_PVALUE_MAX = 0.05
HALF_LIFE_MAX_DAYS = 30
ZSCORE_ENTRY = 2.0
ZSCORE_EXIT = 0.5
ZSCORE_STOP = 3.5

CONSECUTIVE_BREAKS_TO_EXIT = 3

INITIAL_CAPITAL = 100_000
RISK_PCT_PER_TRADE = 0.01
STOP_LOSS_PCT = 0.02


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
        self.dollar_stop_price_a = None
        self.consecutive_breaks = 0


def run_backtest():
    all_series = load_price_data(sorted({t for pair in CANDIDATE_PAIRS for t in pair}))
    print(f"Running walk-forward backtest on {len(CANDIDATE_PAIRS)} intra-sector pairs...")

    results = []
    trade_log = []

    for pair_idx, (pair_a, pair_b) in enumerate(CANDIDATE_PAIRS, start=1):
        if pair_a not in all_series or pair_b not in all_series:
            continue

        if pair_idx % 25 == 0:
            print(f"  pair {pair_idx}/{len(CANDIDATE_PAIRS)}: {pair_a}/{pair_b}")

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
            today = common_idx[i]
            window_a = series_a_full.iloc[max(0, i - LOOKBACK_WINDOW):i + 1]
            window_b = series_b_full.iloc[max(0, i - LOOKBACK_WINDOW):i + 1]

            pair_signals_evaluated += 1

            try:
                coint_result = stats_engine.check_cointegration(window_a, window_b, pair_a, pair_b)
            except Exception:
                continue

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
                hit_dollar_stop = _check_dollar_stop(position, price_a_today)
                hit_zscore_stop = z is not None and abs(z) >= ZSCORE_STOP
                hit_target = z is not None and abs(z) <= ZSCORE_EXIT

                if not coint_result.is_cointegrated:
                    position.consecutive_breaks += 1
                else:
                    position.consecutive_breaks = 0

                hit_coint_break = position.consecutive_breaks >= CONSECUTIVE_BREAKS_TO_EXIT

                if hit_dollar_stop or hit_zscore_stop:
                    pnl = _close_position(position, price_a_today, price_b_today)
                    pnl_total += pnl
                    reason = "dollar_stop" if hit_dollar_stop else "zscore_stop"
                    trade_log.append({
                        "pair": f"{pair_a}/{pair_b}", "exit_date": today,
                        "reason": reason, "pnl": pnl, "exit_z": z,
                    })
                    position = PairPosition()
                elif hit_target:
                    pnl = _close_position(position, price_a_today, price_b_today)
                    pnl_total += pnl
                    trade_log.append({
                        "pair": f"{pair_a}/{pair_b}", "exit_date": today,
                        "reason": "target", "pnl": pnl, "exit_z": z,
                    })
                    position = PairPosition()
                elif hit_coint_break:
                    pnl = _close_position(position, price_a_today, price_b_today)
                    pnl_total += pnl
                    trade_log.append({
                        "pair": f"{pair_a}/{pair_b}", "exit_date": today,
                        "reason": "cointegration_broke", "pnl": pnl, "exit_z": z,
                    })
                    position = PairPosition()

                equity_curve.append({"date": today, "cum_pnl": pnl_total})
                continue

            if coint_result.is_cointegrated and hedge_result is not None and z is not None:
                if abs(z) >= ZSCORE_ENTRY:
                    position.is_open = True
                    position.direction = "LONG_SPREAD" if z < 0 else "SHORT_SPREAD"
                    position.entry_z = z
                    position.entry_date = today
                    position.hedge_ratio_at_entry = hedge_result.hedge_ratio
                    position.entry_price_a = price_a_today
                    position.entry_price_b = price_b_today
                    position.consecutive_breaks = 0

                    stop_distance = price_a_today * STOP_LOSS_PCT
                    shares_a = risk_engine.calc_position_size(
                        INITIAL_CAPITAL, RISK_PCT_PER_TRADE, stop_distance, price_a_today
                    )
                    position.shares_a = shares_a
                    position.shares_b = shares_a * hedge_result.hedge_ratio
                    if position.direction == "LONG_SPREAD":
                        position.dollar_stop_price_a = price_a_today - stop_distance
                    else:
                        position.dollar_stop_price_a = price_a_today + stop_distance

                    pair_trades += 1

            equity_curve.append({"date": today, "cum_pnl": pnl_total})

        if pair_trades > 0 or pnl_total != 0:
            results.append({
                "pair": f"{pair_a}/{pair_b}",
                "signals_evaluated": pair_signals_evaluated,
                "trades_entered": pair_trades,
                "total_pnl": pnl_total,
                "equity_curve": equity_curve,
            })

    _print_summary(results, trade_log)
    return results, trade_log


def _check_dollar_stop(position: PairPosition, price_a_today: float) -> bool:
    if position.dollar_stop_price_a is None:
        return False
    if position.direction == "LONG_SPREAD":
        return price_a_today <= position.dollar_stop_price_a
    else:
        return price_a_today >= position.dollar_stop_price_a


def _close_position(position: PairPosition, price_a: float, price_b: float) -> float:
    shares_a = position.shares_a
    shares_b = position.shares_b

    if position.direction == "LONG_SPREAD":
        pnl = shares_a * (price_a - position.entry_price_a) - shares_b * (price_b - position.entry_price_b)
    else:
        pnl = -shares_a * (price_a - position.entry_price_a) + shares_b * (price_b - position.entry_price_b)
    return float(pnl)


def _print_summary(results: list[dict], trade_log: list[dict]):
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Pairs with activity: {len(results)} / {len(CANDIDATE_PAIRS)} candidate pairs")

    total_pnl = sum(r["total_pnl"] for r in results)
    total_trades = sum(r["trades_entered"] for r in results)

    top = sorted(results, key=lambda r: r["total_pnl"], reverse=True)[:10]
    bottom = sorted(results, key=lambda r: r["total_pnl"])[:10]

    print("\nTop 10 pairs by P&L:")
    for r in top:
        print(f"  {r['pair']}: ${r['total_pnl']:,.2f} ({r['trades_entered']} trades)")

    print("\nBottom 10 pairs by P&L:")
    for r in bottom:
        print(f"  {r['pair']}: ${r['total_pnl']:,.2f} ({r['trades_entered']} trades)")

    print("\n" + "-" * 60)
    print(f"TOTAL trades across all pairs: {total_trades}")
    print(f"TOTAL P&L:                     ${total_pnl:,.2f}")
    print(f"Return on initial capital:     {(total_pnl / INITIAL_CAPITAL) * 100:.2f}%")

    if trade_log:
        wins = [t for t in trade_log if t["pnl"] > 0]
        losses = [t for t in trade_log if t["pnl"] <= 0]
        win_rate = len(wins) / len(trade_log) * 100 if trade_log else 0
        print(f"Win rate:                      {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")

        print("\nExit reason breakdown:")
        reasons = {}
        for t in trade_log:
            reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    print("\nSANITY CHECKS:")
    if total_trades == 0:
        print("  WARNING: ZERO trades fired.")
    elif total_trades > 500:
        print(f"  WARNING: Very high trade count ({total_trades}).")
    else:
        print(f"  Trade count ({total_trades}) — review top/bottom pairs before trusting P&L.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print(f"Candidate pairs generated: {len(CANDIDATE_PAIRS)}")
    results, trade_log = run_backtest()

    if trade_log:
        df = pd.DataFrame(trade_log)
        out_path = os.path.join(os.path.dirname(__file__), "backtest_trade_log.csv")
        df.to_csv(out_path, index=False)
        print(f"\nFull trade log written to: {out_path}")
