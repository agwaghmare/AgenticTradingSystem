"""live_trade table for OOS performance tracking

Revision ID: 002_live_trade
Revises: 001_initial
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002_live_trade"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "live_trade",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("closed_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("strategy_type", sa.String(), nullable=False),
        sa.Column("symbol_a", sa.String(), nullable=False),
        sa.Column("symbol_b", sa.String(), nullable=True),
        sa.Column("direction", sa.String(), nullable=False),
        sa.Column("entry_price", sa.Numeric(16, 4), nullable=False),
        sa.Column("exit_price", sa.Numeric(16, 4), nullable=False),
        sa.Column("shares", sa.Numeric(16, 4), nullable=False),
        sa.Column("realized_pnl", sa.Numeric(16, 2), nullable=False),
        sa.Column("exit_reason", sa.String(), nullable=False),
        sa.Column("z_entry", sa.Numeric(10, 4), nullable=True),
        sa.Column("z_exit", sa.Numeric(10, 4), nullable=True),
        sa.Column("model_version", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_live_trade_closed_at", "live_trade", ["closed_at"])


def downgrade() -> None:
    op.drop_index("ix_live_trade_closed_at", table_name="live_trade")
    op.drop_table("live_trade")
