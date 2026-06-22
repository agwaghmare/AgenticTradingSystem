"""
Parameter sweep for the pairs trading backtest.

Tests combinations of zscore_entry, stop_loss_pct, and lookback_window with
commission friction ($0.005/share round-trip per leg), ranked by Sharpe.

Imports and monkey-patches scripts/backtest.run_backtest() — core fixes apply
automatically. Runs sequentially to avoid Yahoo 429s; checkpoints each combo.

Run from project root:
    python scripts/param_sweep.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

warnings.filterwarnings("ignore")

import json
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

PARAM_GRID = {
    "zscore_entry": [1.5, 2.0, 2.5],
    "stop_loss_pct": [0.02, 0.03, 0.05],
    "lookback_window": [60, 100, 120],
}

COMMISSION_PER_SHARE = 0.005

RESULTS_CSV = Path(__file__).parent / "param_sweep_results.csv"
BEST_TXT = Path(__file__).parent / "param_sweep_best.txt"


def _close_position_with_commission(position, price_a: float, price_b: float) -> float:
    shares_a = position.shares_a
    shares_b = position.shares_b

    if position.direction == "LONG_SPREAD":
        gross = shares_a * (price_a - position.entry_price_a) - shares_b * (price_b - position.entry_price_b)
    else:
        gross = -shares_a * (price_a - position.entry_price_a) + shares_b * (price_b - position.entry_price_b)

    commission = (shares_a + shares_b) * COMMISSION_PER_SHARE * 2
    return float(gross - commission)


def _calc_sharpe(trade_log: list[dict], initial_capital: float = 100_000) -> float:
    if len(trade_log) < 5:
        return -999.0
    pnls = [t["pnl"] for t in trade_log]
    returns = [p / initial_capital for p in pnls]
    mean_r = np.mean(returns)
    std_r = np.std(returns)
    if std_r == 0:
        return 0.0
    return float((mean_r / std_r) * np.sqrt(252))


def run_single_combo(combo: dict) -> dict:
    import importlib
    import scripts.backtest as bt

    importlib.reload(bt)

    bt.ZSCORE_ENTRY = combo["zscore_entry"]
    bt.STOP_LOSS_PCT = combo["stop_loss_pct"]
    bt.LOOKBACK_WINDOW = combo["lookback_window"]
    bt._close_position = _close_position_with_commission

    t0 = time.time()
    try:
        results, trade_log = bt.run_backtest()
    except Exception as e:
        return {**combo, "error": str(e), "sharpe": -999, "total_pnl": None, "trades": 0, "win_rate": None}

    elapsed = time.time() - t0

    total_pnl = sum(r["total_pnl"] for r in results)
    trades = sum(r["trades_entered"] for r in results)
    wins = sum(1 for t in trade_log if t["pnl"] > 0)
    win_rate = wins / len(trade_log) * 100 if trade_log else 0.0
    sharpe = _calc_sharpe(trade_log)

    reasons = {}
    for t in trade_log:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1

    return {
        **combo,
        "sharpe": round(sharpe, 4),
        "total_pnl": round(total_pnl, 2),
        "trades": trades,
        "win_rate": round(win_rate, 1),
        "dollar_stops": reasons.get("dollar_stop", 0),
        "coint_broke": reasons.get("cointegration_broke", 0),
        "targets": reasons.get("target", 0),
        "elapsed_min": round(elapsed / 60, 1),
    }


def main():
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Parameter sweep: {len(combos)} combinations")
    print(f"Parameters: {json.dumps(PARAM_GRID, indent=2)}")
    print(f"Commission: ${COMMISSION_PER_SHARE}/share/side (round-trip both legs)")
    print(f"Results will be saved to: {RESULTS_CSV}")
    print()

    completed = set()
    if RESULTS_CSV.exists():
        existing = pd.read_csv(RESULTS_CSV)
        for _, row in existing.iterrows():
            key = (row["zscore_entry"], row["stop_loss_pct"], row["lookback_window"])
            completed.add(key)
        print(f"Resuming: {len(completed)} combos already done, {len(combos) - len(completed)} remaining")

    remaining = [
        c for c in combos if (c["zscore_entry"], c["stop_loss_pct"], c["lookback_window"]) not in completed
    ]

    if not remaining:
        print("All combos already complete — reading results from CSV")
    else:
        print(f"Running {len(remaining)} combos sequentially (avoids Yahoo 429s)...")
        print("Kill anytime — progress is saved per combo.\n")

        for i, combo in enumerate(remaining):
            print(f"[{i+1}/{len(remaining)}] Testing: {combo}")
            result = run_single_combo(combo)
            print(
                f"  Sharpe: {result['sharpe']:.3f} | P&L: ${result.get('total_pnl', 'ERR'):,} | "
                f"Trades: {result['trades']} | WR: {result.get('win_rate', '?')}%"
            )

            row_df = pd.DataFrame([result])
            if RESULTS_CSV.exists():
                row_df.to_csv(RESULTS_CSV, mode="a", header=False, index=False)
            else:
                row_df.to_csv(RESULTS_CSV, index=False)
            print(f"  Saved to {RESULTS_CSV}")

    all_results = pd.read_csv(RESULTS_CSV).sort_values("sharpe", ascending=False)

    print("\n" + "=" * 70)
    print("PARAMETER SWEEP RESULTS — RANKED BY SHARPE")
    print("=" * 70)
    print(
        all_results[
            [
                "zscore_entry",
                "stop_loss_pct",
                "lookback_window",
                "sharpe",
                "total_pnl",
                "trades",
                "win_rate",
                "dollar_stops",
                "targets",
                "coint_broke",
            ]
        ].to_string(index=False)
    )

    top3 = all_results.head(3)
    summary_lines = [
        "TOP 3 CONFIGURATIONS BY SHARPE (after $0.005/share commission)\n",
        "=" * 60,
    ]
    for rank, (_, row) in enumerate(top3.iterrows(), 1):
        summary_lines.append(
            f"\n#{rank}: zscore_entry={row['zscore_entry']}, "
            f"stop_loss_pct={row['stop_loss_pct']}, "
            f"lookback_window={int(row['lookback_window'])}"
        )
        summary_lines.append(f"  Sharpe:   {row['sharpe']:.4f}")
        summary_lines.append(f"  Net P&L:  ${row['total_pnl']:,.2f}")
        summary_lines.append(f"  Trades:   {int(row['trades'])}")
        summary_lines.append(f"  Win rate: {row['win_rate']:.1f}%")
        summary_lines.append(
            f"  Stops:    {int(row['dollar_stops'])} dollar / "
            f"{int(row['targets'])} target / {int(row['coint_broke'])} coint broke"
        )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    BEST_TXT.write_text(summary_text)
    print(f"\nSummary saved to: {BEST_TXT}")


if __name__ == "__main__":
    main()
