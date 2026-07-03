import uuid
from datetime import datetime, date
from sqlalchemy import String, Numeric, Text, ForeignKey, CheckConstraint, Boolean, Date, Integer
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column
from app.db.session import Base


class DecisionLog(Base):
    __tablename__ = "decision_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    cycle_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow, index=True)
    pair_a: Mapped[str] = mapped_column(String, nullable=False)
    pair_b: Mapped[str] = mapped_column(String, nullable=False)
    action: Mapped[str] = mapped_column(String, nullable=False)
    z_score: Mapped[float | None] = mapped_column(Numeric(10, 4))
    hedge_ratio: Mapped[float | None] = mapped_column(Numeric(12, 6))
    hedge_drift_pct: Mapped[float | None] = mapped_column(Numeric(6, 3))
    p_value: Mapped[float | None] = mapped_column(Numeric(10, 8))
    half_life_days: Mapped[float | None] = mapped_column(Numeric(8, 2))
    regime: Mapped[str | None] = mapped_column(String)
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    rejection_reason: Mapped[str | None] = mapped_column(Text)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    nav_at_decision: Mapped[float | None] = mapped_column(Numeric(16, 2))
    position_size: Mapped[float | None] = mapped_column(Numeric(16, 2))
    order_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))

    __table_args__ = (
        CheckConstraint(
            "action IN ('ENTER_LONG','ENTER_SHORT','EXIT','HOLD','REJECT')", name="ck_decision_action"
        ),
        CheckConstraint(
            "regime IN ('range_bound','trending','high_vol','unknown')", name="ck_decision_regime"
        ),
    )


class OrderLog(Base):
    __tablename__ = "order_log"

    order_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    decision_id: Mapped[int | None] = mapped_column(ForeignKey("decision_log.id"))
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    ticker: Mapped[str] = mapped_column(String, nullable=False)
    side: Mapped[str] = mapped_column(String, nullable=False)
    qty: Mapped[float] = mapped_column(Numeric(16, 4), nullable=False)
    order_type: Mapped[str] = mapped_column(String, nullable=False)
    requested_price: Mapped[float | None] = mapped_column(Numeric(16, 4))
    fill_price: Mapped[float | None] = mapped_column(Numeric(16, 4))
    slippage_bps: Mapped[float | None] = mapped_column(Numeric(8, 2))
    status: Mapped[str] = mapped_column(String, nullable=False)
    broker_order_id: Mapped[str | None] = mapped_column(String)

    __table_args__ = (
        CheckConstraint("side IN ('BUY','SELL')", name="ck_order_side"),
        CheckConstraint("status IN ('PENDING','FILLED','CANCELLED','REJECTED')", name="ck_order_status"),
    )


class RiskSnapshot(Base):
    __tablename__ = "risk_snapshot"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    cycle_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    nav: Mapped[float] = mapped_column(Numeric(16, 2), nullable=False)
    gross_leverage: Mapped[float | None] = mapped_column(Numeric(6, 3))
    net_leverage: Mapped[float | None] = mapped_column(Numeric(6, 3))
    daily_var_95: Mapped[float | None] = mapped_column(Numeric(16, 2))
    drawdown_mtd_pct: Mapped[float | None] = mapped_column(Numeric(6, 3))
    open_positions: Mapped[int | None] = mapped_column(Integer)
    regime: Mapped[str | None] = mapped_column(String)
    halt_status: Mapped[str | None] = mapped_column(String)

    __table_args__ = (
        CheckConstraint("halt_status IN ('NORMAL','EXITS_ONLY','HALTED')", name="ck_risk_halt_status"),
    )


class ModelPerformance(Base):
    __tablename__ = "model_performance"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    sharpe_10d: Mapped[float | None] = mapped_column(Numeric(8, 4))
    sharpe_60d: Mapped[float | None] = mapped_column(Numeric(8, 4))
    flagged_decay: Mapped[bool] = mapped_column(Boolean, default=False)


class PriceHistory(Base):
    __tablename__ = "price_history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String, nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, index=True)
    interval: Mapped[str] = mapped_column(String, nullable=False, default="1d")
    open: Mapped[float | None] = mapped_column(Numeric(16, 4))
    high: Mapped[float | None] = mapped_column(Numeric(16, 4))
    low: Mapped[float | None] = mapped_column(Numeric(16, 4))
    close: Mapped[float] = mapped_column(Numeric(16, 4), nullable=False)
    volume: Mapped[float | None] = mapped_column(Numeric(20, 2))


class TickerMetadata(Base):
    __tablename__ = "ticker_metadata"

    ticker: Mapped[str] = mapped_column(String, primary_key=True)
    sector: Mapped[str | None] = mapped_column(String)
    industry: Mapped[str | None] = mapped_column(String)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)


class LiveTrade(Base):
    """Closed paper/live trades for out-of-sample Sharpe tracking."""

    __tablename__ = "live_trade"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    closed_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, index=True)
    strategy_type: Mapped[str] = mapped_column(String, nullable=False)  # PAIR | SINGLE
    symbol_a: Mapped[str] = mapped_column(String, nullable=False)
    symbol_b: Mapped[str | None] = mapped_column(String)
    direction: Mapped[str] = mapped_column(String, nullable=False)
    entry_price: Mapped[float] = mapped_column(Numeric(16, 4), nullable=False)
    exit_price: Mapped[float] = mapped_column(Numeric(16, 4), nullable=False)
    shares: Mapped[float] = mapped_column(Numeric(16, 4), nullable=False)
    realized_pnl: Mapped[float] = mapped_column(Numeric(16, 2), nullable=False)
    exit_reason: Mapped[str] = mapped_column(String, nullable=False)
    z_entry: Mapped[float | None] = mapped_column(Numeric(10, 4))
    z_exit: Mapped[float | None] = mapped_column(Numeric(10, 4))
    model_version: Mapped[str] = mapped_column(String, nullable=False)
