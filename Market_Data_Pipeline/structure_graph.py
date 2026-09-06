from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

@dataclass
class StructureLevel:
    price: float
    index: int
    timestamp: Optional[datetime] = None
    strength: int = 1
    level_type: str = "SwingPoint"  # 'SwingHigh', 'SwingLow', 'ProtectedHigh', 'ProtectedLow'

    # Version 1.0 additions
    strength_score: float = 1.0
    confirmation_candle: int = 0
    confirmation_delay: int = 0
    bars_since_confirmation: int = -1
    is_valid: bool = True
    broken: bool = False
    reason: str = "active"
    structure_type: str = "Major"  # 'Major', 'Minor', 'Internal'
    parent_index: Optional[int] = None

    # Debugging and Support fields
    why_detected: str = ""
    rule_fired: str = ""
    thresholds_satisfied: List[str] = field(default_factory=list)
    thresholds_failed: List[str] = field(default_factory=list)
    confidence_score: float = 1.0

    @property
    def datetime(self) -> Optional[datetime]:
        return self.timestamp

@dataclass
class LiquidityPool:
    upper: float
    lower: float
    index: int
    timestamp: Optional[datetime] = None
    pool_type: str = "SwingSweep"  # 'EqualHighs', 'EqualLows', 'SwingSweep'
    is_swept: bool = False
    swept_idx: Optional[int] = None

@dataclass
class BOS:
    index: int
    direction: int  # 1 for Bullish, -1 for Bearish
    broken_level: float
    timestamp: Optional[datetime] = None
    strength: int = 1
    distance: float = 0.0
    atr_normalized_distance: float = 0.0

    # Version 1.0 additions
    break_candle: int = 0
    impulse_size: float = 0.0
    atr_normalized_impulse: float = 0.0
    volume: float = 0.0
    break_strength: float = 1.0

    # Debugging and Support fields
    why_detected: str = ""
    rule_fired: str = ""
    thresholds_satisfied: List[str] = field(default_factory=list)
    thresholds_failed: List[str] = field(default_factory=list)
    confidence_score: float = 1.0

@dataclass
class CHOCH:
    index: int
    previous_trend: int
    new_trend: int
    timestamp: Optional[datetime] = None
    price: float = 0.0
    strength: int = 1

    # Version 1.0 additions
    confirmation_score: float = 1.0

    # Debugging and Support fields
    why_detected: str = ""
    rule_fired: str = ""
    thresholds_satisfied: List[str] = field(default_factory=list)
    thresholds_failed: List[str] = field(default_factory=list)
    confidence_score: float = 1.0

@dataclass
class Zone:
    upper: float
    lower: float
    type: str  # 'Supply' or 'Demand'
    created_time: Optional[datetime] = None
    created_idx: int = 0
    freshness: bool = True
    touch_count: int = 0
    broken: bool = False
    broken_idx: Optional[int] = None
    mitigated: bool = False
    mitigated_idx: Optional[int] = None
    strength_score: float = 0.0

    # Version 1.0 additions
    creation_candle: int = 0
    origin_candle: int = 0
    freshness_score: float = 1.0
    number_of_reactions: int = 0
    average_rejection: float = 0.0
    average_penetration: float = 0.0
    invalidated: bool = False
    active: bool = True
    nested_inside_idx: Optional[int] = None

    # Debugging and Support fields
    why_detected: str = ""
    rule_fired: str = ""
    thresholds_satisfied: List[str] = field(default_factory=list)
    thresholds_failed: List[str] = field(default_factory=list)
    confidence_score: float = 1.0

    @property
    def mid(self) -> float:
        return (self.upper + self.lower) / 2

    @property
    def width(self) -> float:
        return self.upper - self.lower

@dataclass
class MarketStructureGraph:
    symbol: str
    timeframe: str
    timestamp: Optional[datetime] = None

    # Swings and Structural Points
    swing_highs: List[StructureLevel] = field(default_factory=list)
    swing_lows: List[StructureLevel] = field(default_factory=list)
    protected_high: Optional[StructureLevel] = None
    protected_low: Optional[StructureLevel] = None

    # Breaks
    bos: List[BOS] = field(default_factory=list)
    choch: List[CHOCH] = field(default_factory=list)

    # Zones
    supply_zones: List[Zone] = field(default_factory=list)
    demand_zones: List[Zone] = field(default_factory=list)

    # Liquidity
    liquidity_pools: List[LiquidityPool] = field(default_factory=list)

    # Indicators / Context
    trend_direction: str = "Neutral"  # "Bull", "Bear", "Neutral"
    ema_relationship: str = "Flat"    # "BullishSeparated", "BearishSeparated", "Converged"
    ema_distance_atr: float = 0.0
    atr: float = 0.0
    volatility: float = 0.0
    range_width_pips: float = 0.0
    session: str = "Unknown"          # "London", "NewYork", "Asian"

    # Version 1.0 additions
    current_bias: str = "Neutral"      # e.g. "Bullish", "Bearish", "Neutral"
    structure_confidence: float = 1.0

    def get_nearest_demand(self, price: float) -> Optional[Zone]:
        active_demands = [z for z in self.demand_zones if not z.broken and z.upper < price]
        return max(active_demands, key=lambda z: z.upper) if active_demands else None

    def get_nearest_supply(self, price: float) -> Optional[Zone]:
        active_supplies = [z for z in self.supply_zones if not z.broken and z.lower > price]
        return min(active_supplies, key=lambda z: z.lower) if active_supplies else None

    # Point-in-time Causal Query Methods
    def get_active_demands(self, idx: int) -> List[Zone]:
        """Returns demand zones that were created at or before idx and not yet broken by idx."""
        return [
            z for z in self.demand_zones
            if z.created_idx <= idx and (z.broken_idx is None or z.broken_idx > idx)
        ]

    def get_active_supplies(self, idx: int) -> List[Zone]:
        """Returns supply zones that were created at or before idx and not yet broken by idx."""
        return [
            z for z in self.supply_zones
            if z.created_idx <= idx and (z.broken_idx is None or z.broken_idx > idx)
        ]

    def get_nearest_demand_at(self, price: float, idx: int) -> Optional[Zone]:
        """Point-in-time query for nearest active demand zone below price at index idx."""
        active = [z for z in self.get_active_demands(idx) if z.upper < price]
        return max(active, key=lambda z: z.upper) if active else None

    def get_nearest_supply_at(self, price: float, idx: int) -> Optional[Zone]:
        """Point-in-time query for nearest active supply zone above price at index idx."""
        active = [z for z in self.get_active_supplies(idx) if z.lower > price]
        return min(active, key=lambda z: z.lower) if active else None

    def get_confirmed_swings_high(self, idx: int) -> List[StructureLevel]:
        """Point-in-time query for swing highs confirmed at or before index idx."""
        return [s for s in self.swing_highs if s.confirmation_candle <= idx]

    def get_confirmed_swings_low(self, idx: int) -> List[StructureLevel]:
        """Point-in-time query for swing lows confirmed at or before index idx."""
        return [s for s in self.swing_lows if s.confirmation_candle <= idx]
