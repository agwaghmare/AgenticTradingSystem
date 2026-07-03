import math
import uuid
from typing import Literal, Optional
from pydantic import BaseModel, field_validator

# Max absolute values the decision_log NUMERIC columns can hold (precision - scale digits).
_DECISION_LOG_CAPS = {
    "z_score": 999_999.9999,          # NUMERIC(10, 4)
    "hedge_ratio": 999_999.999999,    # NUMERIC(12, 6)
    "hedge_drift_pct": 999.999,       # NUMERIC(6, 3)
    "p_value": 99.99999999,           # NUMERIC(10, 8)
    "half_life_days": 999_999.99,     # NUMERIC(8, 2)
    "nav_at_decision": 99_999_999_999_999.99,  # NUMERIC(16, 2)
    "position_size": 99_999_999_999_999.99,    # NUMERIC(16, 2)
}


class DecisionLogSchema(BaseModel):
    cycle_id: uuid.UUID
    pair_a: str
    pair_b: str
    action: Literal["ENTER_LONG", "ENTER_SHORT", "EXIT", "HOLD", "REJECT"]
    z_score: Optional[float] = None
    hedge_ratio: Optional[float] = None
    hedge_drift_pct: Optional[float] = None
    p_value: Optional[float] = None
    half_life_days: Optional[float] = None
    regime: Optional[Literal["range_bound", "trending", "high_vol", "unknown"]] = None
    reasoning: str
    rejection_reason: Optional[str] = None
    model_version: str
    nav_at_decision: Optional[float] = None
    position_size: Optional[float] = None
    order_id: Optional[uuid.UUID] = None

    @field_validator(*_DECISION_LOG_CAPS.keys(), mode="before")
    @classmethod
    def _sanitize_numeric(cls, value, info):
        """inf/NaN -> NULL; clamp finite values to what the DB column can store."""
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        cap = _DECISION_LOG_CAPS[info.field_name]
        return max(-cap, min(cap, value))


class CointegrationResult(BaseModel):
    pair_a: str
    pair_b: str
    p_value: float
    half_life_days: float
    is_cointegrated: bool


class HedgeRatioResult(BaseModel):
    pair_a: str
    pair_b: str
    hedge_ratio: float
    drift_pct: float


class PriceMeanReversionResult(BaseModel):
    ticker: str
    z_score: float
    half_life_days: float
    is_mean_reverting: bool


class RiskCheckResult(BaseModel):
    passed: bool
    reasons: list[str] = []


class TradeProposal(BaseModel):
    pair_a: str
    pair_b: str
    side_a: Literal["BUY", "SELL"]
    side_b: Literal["BUY", "SELL"]
    z_score: float
    hedge_ratio: float
    proposed_size_a: float
    proposed_size_b: float
