from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Type, Union

@dataclass
class FeatureDefinition:
    name: str
    display_name: str
    category: str  # e.g., Trend, Momentum, Structure, SupplyDemand, Volatility, Time, Session, etc.
    dtype: Union[Type, str]  # e.g., float, int, str, or "float", "int", "str"
    units: str  # e.g., "price/bar", "pips", "count", "seconds", "ratio", "none"
    description: str
    source_module: str  # e.g., "IndicatorEngine", "MarketStructureEngine", "SupplyDemandEngine", "FeaturePipeline"
    version: str = "1.0"
    enabled: bool = True
    required: bool = True
    normalize: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    default_value: Any = 0.0
    missing_value_policy: str = "fill_zero"  # "fill_zero", "drop", "mean", "median", "ffill"
    creation_timestamp: datetime = field(default_factory=datetime.now)
    last_modified_timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "category": self.category,
            "dtype": self.dtype.__name__ if isinstance(self.dtype, type) else str(self.dtype),
            "units": self.units,
            "description": self.description,
            "source_module": self.source_module,
            "version": self.version,
            "enabled": self.enabled,
            "required": self.required,
            "normalize": self.normalize,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "default_value": self.default_value,
            "missing_value_policy": self.missing_value_policy,
            "creation_timestamp": self.creation_timestamp.isoformat(),
            "last_modified_timestamp": self.last_modified_timestamp.isoformat(),
        }
