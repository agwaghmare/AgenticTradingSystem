import numpy as np
from app.config import settings
from app.models.schemas import RiskCheckResult


def calc_position_size(account_nav: float, risk_pct: float, stop_distance: float, price: float) -> float:
    """Position size in shares, based on % NAV at risk and stop distance."""
    if stop_distance <= 0 or price <= 0:
        return 0.0
    risk_dollars = account_nav * risk_pct
    shares = risk_dollars / stop_distance
    max_shares_by_nav = (account_nav * settings.max_nav_pct_per_name) / price
    return float(min(shares, max_shares_by_nav))


def check_risk_limits(
    proposed_notional: float,
    nav: float,
    sector_notional: float,
    gross_exposure: float,
    net_exposure: float,
    open_positions: int,
) -> RiskCheckResult:
    reasons = []

    if proposed_notional / nav > settings.max_nav_pct_per_name:
        reasons.append(f"Exceeds max NAV per name ({settings.max_nav_pct_per_name:.0%})")

    if sector_notional / nav > settings.max_nav_pct_per_sector:
        reasons.append(f"Exceeds max NAV per sector ({settings.max_nav_pct_per_sector:.0%})")

    if gross_exposure > settings.max_gross_leverage:
        reasons.append(f"Exceeds max gross leverage ({settings.max_gross_leverage}x)")

    if abs(net_exposure) > settings.max_net_leverage:
        reasons.append(f"Exceeds max net leverage ({settings.max_net_leverage}x)")

    if open_positions >= settings.max_concurrent_positions:
        reasons.append(f"At max concurrent positions ({settings.max_concurrent_positions})")

    return RiskCheckResult(passed=len(reasons) == 0, reasons=reasons)


def check_correlation_exposure(
    new_position_returns, existing_positions_returns: dict[str, "np.ndarray"], threshold: float = 0.7
) -> RiskCheckResult:
    high_corr_count = 0
    reasons = []
    for ticker, returns in existing_positions_returns.items():
        if len(returns) < 2 or len(new_position_returns) < 2:
            continue
        corr = np.corrcoef(new_position_returns, returns)[0, 1]
        if abs(corr) > threshold:
            high_corr_count += 1
            reasons.append(f"Correlation with {ticker} = {corr:.2f}")

    passed = high_corr_count < 3
    if not passed:
        reasons.insert(0, "3+ positions with pairwise correlation > 0.7")
    return RiskCheckResult(passed=passed, reasons=reasons)


def get_drawdown_status(drawdown_mtd_pct: float) -> str:
    if drawdown_mtd_pct <= settings.drawdown_flatten_all_pct:
        return "HALTED"
    if drawdown_mtd_pct <= settings.drawdown_halt_entries_pct:
        return "EXITS_ONLY"
    return "NORMAL"


def calc_portfolio_var(position_returns: list["np.ndarray"], weights: list[float], confidence: float = 0.95) -> float:
    """Simple historical/parametric VaR (95%, 1-day) on weighted portfolio returns."""
    if not position_returns:
        return 0.0
    returns_matrix = np.array(position_returns)
    weights_arr = np.array(weights)
    portfolio_returns = returns_matrix.T @ weights_arr
    var = -np.percentile(portfolio_returns, (1 - confidence) * 100)
    return float(var)
