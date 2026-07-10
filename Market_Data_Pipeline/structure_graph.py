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

@dataclass
class CHOCH:
    index: int
    previous_trend: int
    new_trend: int
    timestamp: Optional[datetime] = None
    price: float = 0.0
    strength: int = 1

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

    def get_nearest_demand(self, price: float) -> Optional[Zone]:
        active_demands = [z for z in self.demand_zones if not z.broken and z.upper < price]
        return max(active_demands, key=lambda z: z.upper) if active_demands else None

    def get_nearest_supply(self, price: float) -> Optional[Zone]:
        active_supplies = [z for z in self.supply_zones if not z.broken and z.lower > price]
        return min(active_supplies, key=lambda z: z.lower) if active_supplies else None
