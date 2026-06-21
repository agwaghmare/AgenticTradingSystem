"""initial schema

Revision ID: 001_initial
Revises:
Create Date: 2026-06-20

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "decision_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("cycle_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("timestamp", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("pair_a", sa.String(), nullable=False),
        sa.Column("pair_b", sa.String(), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("z_score", sa.Numeric(10, 4), nullable=True),
        sa.Column("hedge_ratio", sa.Numeric(12, 6), nullable=True),
        sa.Column("hedge_drift_pct", sa.Numeric(6, 3), nullable=True),
        sa.Column("p_value", sa.Numeric(10, 8), nullable=True),
        sa.Column("half_life_days", sa.Numeric(8, 2), nullable=True),
        sa.Column("regime", sa.String(), nullable=True),
        sa.Column("reasoning", sa.Text(), nullable=False),
        sa.Column("rejection_reason", sa.Text(), nullable=True),
        sa.Column("model_version", sa.String(), nullable=False),
        sa.Column("nav_at_decision", sa.Numeric(16, 2), nullable=True),
        sa.Column("position_size", sa.Numeric(16, 2), nullable=True),
        sa.Column("order_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.CheckConstraint(
            "action IN ('ENTER_LONG','ENTER_SHORT','EXIT','HOLD','REJECT')", name="ck_decision_action"
        ),
        sa.CheckConstraint(
            "regime IN ('range_bound','trending','high_vol','unknown')", name="ck_decision_regime"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_decision_log_cycle_id", "decision_log", ["cycle_id"])
    op.create_index("ix_decision_log_timestamp", "decision_log", ["timestamp"])

    op.create_table(
        "order_log",
        sa.Column("order_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("decision_id", sa.Integer(), nullable=True),
        sa.Column("timestamp", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("side", sa.String(), nullable=False),
        sa.Column("qty", sa.Numeric(16, 4), nullable=False),
        sa.Column("order_type", sa.String(), nullable=False),
        sa.Column("requested_price", sa.Numeric(16, 4), nullable=True),
        sa.Column("fill_price", sa.Numeric(16, 4), nullable=True),
        sa.Column("slippage_bps", sa.Numeric(8, 2), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("broker_order_id", sa.String(), nullable=True),
        sa.CheckConstraint("side IN ('BUY','SELL')", name="ck_order_side"),
        sa.CheckConstraint("status IN ('PENDING','FILLED','CANCELLED','REJECTED')", name="ck_order_status"),
        sa.ForeignKeyConstraint(["decision_id"], ["decision_log.id"]),
        sa.PrimaryKeyConstraint("order_id"),
    )

    op.create_table(
        "risk_snapshot",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("cycle_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("timestamp", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("nav", sa.Numeric(16, 2), nullable=False),
        sa.Column("gross_leverage", sa.Numeric(6, 3), nullable=True),
        sa.Column("net_leverage", sa.Numeric(6, 3), nullable=True),
        sa.Column("daily_var_95", sa.Numeric(16, 2), nullable=True),
        sa.Column("drawdown_mtd_pct", sa.Numeric(6, 3), nullable=True),
        sa.Column("open_positions", sa.Integer(), nullable=True),
        sa.Column("regime", sa.String(), nullable=True),
        sa.Column("halt_status", sa.String(), nullable=True),
        sa.CheckConstraint("halt_status IN ('NORMAL','EXITS_ONLY','HALTED')", name="ck_risk_halt_status"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_risk_snapshot_cycle_id", "risk_snapshot", ["cycle_id"])

    op.create_table(
        "model_performance",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("model_version", sa.String(), nullable=False),
        sa.Column("sharpe_10d", sa.Numeric(8, 4), nullable=True),
        sa.Column("sharpe_60d", sa.Numeric(8, 4), nullable=True),
        sa.Column("flagged_decay", sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "price_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("timestamp", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("interval", sa.String(), nullable=False),
        sa.Column("open", sa.Numeric(16, 4), nullable=True),
        sa.Column("high", sa.Numeric(16, 4), nullable=True),
        sa.Column("low", sa.Numeric(16, 4), nullable=True),
        sa.Column("close", sa.Numeric(16, 4), nullable=False),
        sa.Column("volume", sa.Numeric(20, 2), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_price_history_ticker", "price_history", ["ticker"])
    op.create_index("ix_price_history_timestamp", "price_history", ["timestamp"])

    op.create_table(
        "ticker_metadata",
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("sector", sa.String(), nullable=True),
        sa.Column("industry", sa.String(), nullable=True),
        sa.Column("updated_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("ticker"),
    )


def downgrade() -> None:
    op.drop_table("ticker_metadata")
    op.drop_index("ix_price_history_timestamp", table_name="price_history")
    op.drop_index("ix_price_history_ticker", table_name="price_history")
    op.drop_table("price_history")
    op.drop_table("model_performance")
    op.drop_index("ix_risk_snapshot_cycle_id", table_name="risk_snapshot")
    op.drop_table("risk_snapshot")
    op.drop_table("order_log")
    op.drop_index("ix_decision_log_timestamp", table_name="decision_log")
    op.drop_index("ix_decision_log_cycle_id", table_name="decision_log")
    op.drop_table("decision_log")
