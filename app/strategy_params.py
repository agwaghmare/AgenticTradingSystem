"""
FROZEN live strategy parameters — do not tune from historical backtests.

The 19-pair universe was selected using 2021-2026 in-sample results (selection bias).
In-sample Sharpe (~8+) is an upper bound, not a forecast. Out-of-sample Sharpe is
unknown until 30-50 paper trades accumulate.

Do NOT run param_sweep on the live universe. Paper trade with these values locked.
"""

from datetime import date

FROZEN_AS_OF = date(2026, 7, 2)
MODEL_VERSION = "agentic-trading-v0.5"

# --- Pairs (locked) ---
ZSCORE_ENTRY = 2.5
ZSCORE_EXIT = 0.5
ZSCORE_STOP = 3.5
# Entries require ZSCORE_ENTRY <= |z| < ZSCORE_STOP (no entering beyond the stop)
TIME_STOP_HALF_LIVES = 3.0
MAX_HOLD_DAYS = 20
LOOKBACK_WINDOW = 100
ZSCORE_LOOKBACK = 20
STOP_LOSS_PCT = 0.03
HALF_LIFE_MIN_DAYS = 3.0
HALF_LIFE_MAX_DAYS = 20.0
HEDGE_DRIFT_MAX_PCT = 0.15
COINTEGRATION_PVALUE_MAX = 0.05
CONSECUTIVE_BREAKS_TO_EXIT = 3
MAX_CONCURRENT_PAIR_POSITIONS = 8

# --- Single-stock mean reversion (locked, separate signal) ---
SINGLE_ZSCORE_ENTRY = 2.5
SINGLE_ZSCORE_EXIT = 0.5
SINGLE_ZSCORE_STOP = 3.5
SINGLE_LOOKBACK = 20
MAX_CONCURRENT_SINGLE_POSITIONS = 4

# Minimum |z| for "high conviction" — same bar for pairs and singles
MIN_CONVICTION_Z = 2.5

# Paper-trade targets for OOS validation
OOS_MIN_TRADES_FOR_SHARPE = 30
OOS_TARGET_SHARPE = 1.5
