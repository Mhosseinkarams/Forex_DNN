from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime

from Market_Data_Pipeline.structure_graph import MarketStructureGraph

@dataclass
class StateContext:
    regime: str  # 'Trending', 'Ranging', 'Transition', 'Expansion', 'Compression'
    trend_direction: str  # 'Bull', 'Bear', 'Neutral'
    volatility_regime: str  # 'High', 'Low', 'Normal'
    confidence_score: float  # 0.0 to 1.0
    ema_slope: float
    ema_distance_atr: float
    atr: float
    timestamp: Optional[datetime] = None

class MarketStateEngine:
    """
    Purpose:
        Analyze MarketStructureGraph to classify current market state regimes
        (Trending, Ranging, Transition, Expansion, Compression) along with confidence values.
        Replaces strategy-specific trend detection.
    """
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold

    def evaluate(self, msg: MarketStructureGraph) -> StateContext:
        """
        Evaluate the shared MarketStructureGraph to identify the market regime and state metrics.
        """
        # Determine trend direction
        trend = msg.trend_direction if msg.trend_direction in ["Bull", "Bear"] else "Neutral"

        # Volatility classification
        vol_regime = "Normal"
        if msg.volatility > 1.5:
            vol_regime = "High"
        elif msg.volatility < 0.5:
            vol_regime = "Low"

        # Determine market regime
        regime = "Ranging"
        confidence = 0.5

        # We can analyze the structure and EMA separation to identify the state
        if msg.ema_relationship in ["BullishSeparated", "BearishSeparated"] and msg.ema_distance_atr >= 1.5:
            regime = "Trending"
            confidence = min(0.95, 0.5 + (msg.ema_distance_atr / 10.0))
        elif msg.ema_relationship == "Converged" or msg.ema_distance_atr < 1.0:
            regime = "Ranging"
            confidence = 0.8

        # Check for Transition regime (recent CHOCH)
        if msg.choch and (msg.choch[-1].index >= (msg.timestamp or datetime.now()).timestamp() - 10 if isinstance(msg.timestamp, datetime) else True):
            # If recent change of character, label transition
            regime = "Transition"
            confidence = 0.7

        # Check for Expansion / Compression
        # If ATR is high relative to past, expansion
        if msg.volatility > 1.3:
            regime = "Expansion"
            confidence = 0.75
        elif msg.volatility < 0.7:
            regime = "Compression"
            confidence = 0.7

        return StateContext(
            regime=regime,
            trend_direction=trend,
            volatility_regime=vol_regime,
            confidence_score=confidence,
            ema_slope=0.0, # Will be set during evaluation if available
            ema_distance_atr=msg.ema_distance_atr,
            atr=msg.atr,
            timestamp=msg.timestamp
        )
