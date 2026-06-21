import uuid
from typing import Literal, Optional
from pydantic import BaseModel


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
