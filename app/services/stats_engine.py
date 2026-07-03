import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from pykalman import KalmanFilter

from app.models.schemas import CointegrationResult, HedgeRatioResult, PriceMeanReversionResult
from app.config import settings


def check_cointegration(series_a: pd.Series, series_b: pd.Series, pair_a: str, pair_b: str) -> CointegrationResult:
    """Engle-Granger cointegration test + Ornstein-Uhlenbeck half-life."""
    _, p_value, _ = coint(series_a, series_b)

    # static OLS hedge ratio for spread construction (half-life calc only)
    x = add_constant(series_b)
    model = OLS(series_a, x).fit()
    hedge_ratio_static = model.params.iloc[1]
    spread = series_a - hedge_ratio_static * series_b

    half_life = _ornstein_uhlenbeck_half_life(spread)

    return CointegrationResult(
        pair_a=pair_a,
        pair_b=pair_b,
        p_value=float(p_value),
        half_life_days=float(half_life),
        is_cointegrated=p_value < settings.cointegration_pvalue_max
        and settings.half_life_min_days <= half_life < settings.half_life_max_days,
    )


def _ornstein_uhlenbeck_half_life(spread: pd.Series) -> float:
    spread_lag = spread.shift(1).dropna()
    spread_ret = spread.diff().dropna()
    spread_lag = spread_lag.loc[spread_ret.index]

    x = add_constant(spread_lag)
    model = OLS(spread_ret, x).fit()
    theta = model.params.iloc[1]

    if theta >= 0:
        return float("inf")  # not mean-reverting
    half_life = -np.log(2) / theta
    return max(half_life, 0.0)


def compute_hedge_ratio_kalman(
    series_a: pd.Series, series_b: pd.Series, pair_a: str, pair_b: str, prev_ratio: float | None = None
) -> HedgeRatioResult:
    """Dynamic hedge ratio via Kalman filter. Returns latest ratio + drift vs prev session."""
    obs_mat = np.vstack([series_b.values, np.ones(len(series_b))]).T[:, np.newaxis]

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=[0, 0],
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=1.0,
        transition_covariance=1e-4 * np.eye(2),
    )

    state_means, _ = kf.filter(series_a.values)
    latest_ratio = float(state_means[-1, 0])

    drift_pct = 0.0
    if prev_ratio is not None and prev_ratio != 0:
        drift_pct = abs(latest_ratio - prev_ratio) / abs(prev_ratio)

    return HedgeRatioResult(pair_a=pair_a, pair_b=pair_b, hedge_ratio=latest_ratio, drift_pct=drift_pct)


def calc_zscore(spread: pd.Series, lookback: int = 20) -> float:
    """Latest z-score of spread vs rolling mean. Returns NaN when the spread is
    flat/degenerate (zero std) instead of blowing up to +/-inf."""
    rolling_mean = spread.rolling(lookback).mean()
    rolling_std = spread.rolling(lookback).std()
    mean = float(rolling_mean.iloc[-1])
    std = float(rolling_std.iloc[-1])
    if not np.isfinite(std) or std <= 1e-12 or not np.isfinite(mean):
        return float("nan")
    return float((float(spread.iloc[-1]) - mean) / std)


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Average True Range over `period` bars."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean()
    value = float(atr.iloc[-1])
    if pd.isna(value) or value <= 0:
        return float(close.iloc[-1] * 0.02)
    return value


def check_price_mean_reversion(
    close: pd.Series,
    ticker: str,
    lookback: int = 20,
) -> PriceMeanReversionResult:
    """Log-price z-score vs rolling mean; OU half-life on demeaned log price."""
    log_price = np.log(close.astype(float))
    residual = log_price - log_price.rolling(lookback).mean()
    z = calc_zscore(residual, lookback=lookback)
    half_life = _ornstein_uhlenbeck_half_life(residual.dropna())

    is_mr = (
        np.isfinite(z)
        and settings.half_life_min_days <= half_life < settings.half_life_max_days
    )

    # Keep NaN as NaN — substituting 0.0 would look like a "reverted" spread and
    # trip target exits downstream. Consumers must check finiteness.
    return PriceMeanReversionResult(
        ticker=ticker,
        z_score=float(z),
        half_life_days=float(half_life),
        is_mean_reverting=is_mr,
    )
