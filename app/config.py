from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://postgres:localpassword@localhost:5432/agentictradingsystem"

    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    anthropic_api_key: str = ""
    mistral_api_key: str = ""
    finnhub_api_key: str = ""
    discord_webhook_url: str = ""

    # Data pipeline
    price_history_days: int = 180
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0
    market_benchmark_ticker: str = "SPY"

    # Position limits
    max_nav_pct_per_name: float = 0.05
    max_nav_pct_per_sector: float = 0.20
    max_gross_leverage: float = 2.0
    max_net_leverage: float = 1.0
    max_risk_pct_per_trade: float = 0.01
    max_concurrent_positions: int = 8

    # Drawdown / regime halts
    drawdown_halt_entries_pct: float = -0.08
    drawdown_flatten_all_pct: float = -0.15

    # Portfolio risk
    daily_var_95_max_pct: float = 0.03
    correlation_threshold: float = 0.7
    max_correlated_positions: int = 3

    # Pairs validation
    cointegration_pvalue_max: float = 0.05
    half_life_max_days: float = 30.0
    hedge_drift_max_pct: float = 0.20
    earnings_blackout_days: int = 1

    # Z-score thresholds
    zscore_entry: float = 2.0
    zscore_exit: float = 0.5
    zscore_stop: float = 3.5

    # Execution
    max_order_pct_adv: float = 0.10
    slippage_bps_max: float = 5.0
    stale_price_seconds: int = 60

    # Governance
    sharpe_drift_short_window_days: int = 10
    sharpe_drift_long_window_days: int = 60

    class Config:
        env_file = ".env"


settings = Settings()
