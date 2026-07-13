from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import pandas as pd

@dataclass
class MarketStructureResult:
    """Strongly-typed result container for Market Structure Engine."""
    swings: List[Any] = field(default_factory=list)
    bos: List[Any] = field(default_factory=list)
    choch: List[Any] = field(default_factory=list)
    protected_high: Optional[Any] = None
    protected_low: Optional[Any] = None
    df_indicators: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SupplyDemandResult:
    """Strongly-typed result container for Supply and Demand Engine."""
    zones: List[Any] = field(default_factory=list)
    df_zones: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FeatureVector:
    """Strongly-typed container for feature vectors."""
    features: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LabelResult:
    """Strongly-typed result container for Label Engine."""
    label: Optional[str] = None
    confidence: float = 0.0
    info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DatasetSample:
    """Strongly-typed row in the generated dataset."""
    sample_id: str
    symbol: str
    timeframe: str
    window_start_datetime: str
    window_end_datetime: str
    feature_vector: FeatureVector
    label_result: LabelResult
    raw_prices: Dict[str, Any] = field(default_factory=dict)
    raw_emas: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flattens the sample into a flat dictionary suitable for pandas DataFrame."""
        flat = {
            "sample_id": self.sample_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "window_start": self.metadata.get("window_start"),
            "window_end": self.metadata.get("window_end"),
            "datetime": self.window_end_datetime,
            "target": self.label_result.label,
            "confidence": self.label_result.confidence
        }
        # Add features
        flat.update(self.feature_vector.features)
        # Add raw prices
        flat.update(self.raw_prices)
        # Add raw EMAs
        flat.update(self.raw_emas)
        # Add labeler rule info
        for k, v in self.label_result.info.items():
            flat[f"meta_labeler_{k}"] = v
        # Add metadata keys to prevent duplicates or collisions
        for k, v in self.metadata.items():
            if k not in ["window_start", "window_end"]:
                flat[f"meta_{k}"] = v
        return flat
