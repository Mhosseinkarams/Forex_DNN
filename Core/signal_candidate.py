from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime

@dataclass
class SignalCandidate:
    """
    Universal SignalCandidate representing a trade setup or signal.
    This is the ONLY interface between trading strategies and StrategyManager / execution services.
    """
    signal_id: int
    strategy_name: str
    strategy_version: str
    symbol: str
    timeframe: str
    timestamp: str            # ISO or formatted bar timestamp
    direction: int            # 1 for BUY, -1 for SELL
    signal_type: str          # e.g., standard, high_risk, reversal, pullback, breakout
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    market_state: str         # RANGE, TREND, TRANSITION, etc.
    trend: str                # Bull, Bear, Neutral
    signal_quality: int       # Common scoring convention (0 to 100)
    confidence: float        # Common scoring convention (0.0 to 1.0)
    risk_multiplier: float = 1.0

    # Engine specific snapshots and info
    strong_candle_info: Dict[str, Any] = field(default_factory=dict)
    refusal_info: Dict[str, Any] = field(default_factory=dict)
    market_structure_snapshot: Dict[str, Any] = field(default_factory=dict)
    supply_demand_snapshot: Dict[str, Any] = field(default_factory=dict)
    ml_predictions: Dict[str, Any] = field(default_factory=dict)

    reasoning: str = ""
    priority: str = "MEDIUM"   # LOW, MEDIUM, HIGH, IMMEDIATE
    status: str = "CREATED"    # CREATED, VALIDATED, ML_APPROVED, EXECUTED, REJECTED, EXPIRED, CANCELLED, CLOSED

    # Future label placeholders
    future_labels: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the candidate to a flat dictionary or JSON serializable format."""
        return asdict(self)
