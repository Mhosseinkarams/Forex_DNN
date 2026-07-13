from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging

from ML.feature_registry import FeatureRegistry
from Market_Data_Pipeline.structure_graph import MarketStructureGraph

logger = logging.getLogger("Pipeline")

@dataclass
class MarketStructureResult:
    swings: List[Any]
    protected_high: Optional[Any]
    protected_low: Optional[Any]
    bos: List[Any]
    choch: List[Any]
    dataframe: pd.DataFrame

@dataclass
class SupplyDemandResult:
    zones: List[Any]
    dataframe: pd.DataFrame

@dataclass
class FeatureVector:
    features: Dict[str, Any]

@dataclass
class LabelResult:
    label: Optional[str]
    confidence: float
    label_info: Dict[str, Any]

@dataclass
class DatasetSample:
    sample_id: str
    symbol: str
    timeframe: str
    window_start: int
    window_end: int
    datetime: str
    features: FeatureVector
    label: LabelResult
    raw_prices: Dict[str, Any]

class Pipeline:
    def __init__(self):
        self.steps = []

    def register(self, step: Any):
        self.steps.append(step)
        return self

    def execute(self, df_ohlcv: pd.DataFrame, symbol: str, timeframe: str, window_size: int, window_stride: int) -> Tuple[List[DatasetSample], pd.DataFrame]:
        df = df_ohlcv.copy()
        if "ema_50" not in df.columns:
            from Collecting_Data.indicators import IndicatorEngine
            ind_engine = IndicatorEngine(ema_periods=[50, 600, 800], slope_period=32)
            df = ind_engine.calculate(df)

        # Execute registered engines sequentially
        swing_highs = []
        swing_lows = []
        protected_high = None
        protected_low = None
        bos = []
        choch = []
        supply_zones = []
        demand_zones = []
        liquidity_pools = []

        registry = None
        label_engine = None

        for step in self.steps:
            if isinstance(step, FeatureRegistry):
                registry = step
                continue

            # Check LabelEngine without circular import
            if step.__class__.__name__ == "LabelEngine":
                label_engine = step
                continue

            # Process dataframe if step has a process method
            if hasattr(step, "process"):
                df = step.process(df)

            # Gather structures dynamically
            if hasattr(step, "swings"):
                swings = getattr(step, "swings") or []
                swing_highs.extend([s for s in swings if s.level_type == "SwingHigh"])
                swing_lows.extend([s for s in swings if s.level_type == "SwingLow"])
            if hasattr(step, "protected_high") and getattr(step, "protected_high"):
                protected_high = getattr(step, "protected_high")
            if hasattr(step, "protected_low") and getattr(step, "protected_low"):
                protected_low = getattr(step, "protected_low")
            if hasattr(step, "bos_list"):
                bos.extend(list(getattr(step, "bos_list") or []))
            if hasattr(step, "choch_list"):
                choch.extend(list(getattr(step, "choch_list") or []))
            if hasattr(step, "zones"):
                zones = getattr(step, "zones") or []
                supply_zones.extend([z for z in zones if z.type == "Supply"])
                demand_zones.extend([z for z in zones if z.type == "Demand"])
            if hasattr(step, "liquidity_pools"):
                liquidity_pools.extend(list(getattr(step, "liquidity_pools") or []))

        # Instantiate MarketStructureResult and SupplyDemandResult to verify typed outputs
        ms_result = MarketStructureResult(
            swings=swing_highs + swing_lows,
            protected_high=protected_high,
            protected_low=protected_low,
            bos=bos,
            choch=choch,
            dataframe=df
        )

        sd_result = SupplyDemandResult(
            zones=supply_zones + demand_zones,
            dataframe=df
        )

        # Determine trend & ATR
        current_trend = 0
        if "trend" in df.columns:
            current_trend = int(df.iloc[-1]["trend"])
        atr_val = 0.0001
        if "atr_14" in df.columns:
            atr_val = float(df.iloc[-1]["atr_14"])

        # Construct MSG
        msg = MarketStructureGraph(
            symbol=symbol,
            timeframe=timeframe,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            protected_high=protected_high,
            protected_low=protected_low,
            bos=bos,
            choch=choch,
            supply_zones=supply_zones,
            demand_zones=demand_zones,
            liquidity_pools=liquidity_pools,
            trend_direction="Bull" if current_trend == 1 else ("Bear" if current_trend == -1 else "Neutral"),
            atr=atr_val
        )

        if not registry:
            registry = FeatureRegistry(load_defaults=True)
        if not label_engine:
            from ML.label_engine import LabelEngine
            label_engine = LabelEngine(window_size=window_size, window_stride=window_stride, registry=registry)

        from ML.feature_pipeline import FeaturePipeline
        pipeline_extractor = FeaturePipeline(registry)

        samples = []
        n_bars = len(df)
        if n_bars < window_size:
            return samples, df

        for start_idx in range(0, n_bars - window_size + 1, window_stride):
            end_idx = start_idx + window_size - 1

            # Get LabelResult
            label, confidence, label_info = label_engine.labeler.label_window(df, msg, start_idx, end_idx)
            label_result = LabelResult(label=label, confidence=confidence, label_info=label_info)

            if label is None:
                continue

            # Feature extraction
            feats = pipeline_extractor.extract_all(df, msg, idx=end_idx)
            feature_vector = FeatureVector(features=feats)

            # Sample ID
            start_dt = df.iloc[start_idx].get("Datetime")
            end_dt = df.iloc[end_idx].get("Datetime")

            def format_dt(dt):
                if isinstance(dt, pd.Timestamp):
                    return dt.strftime("%Y-%m-%dT%H:%M")
                return str(dt).replace(" ", "T")[:16]

            start_str = format_dt(start_dt)
            end_str = format_dt(end_dt)
            sample_id = f"{symbol}_{timeframe}_{start_str}_{end_str}".replace(":", "-")

            raw_prices = {}
            for raw_col in ["Open", "High", "Low", "Close", "TickVolume", "ema_50", "ema_600", "ema_800"]:
                if raw_col in df.columns:
                    raw_prices[raw_col] = df.iloc[end_idx][raw_col]

            sample = DatasetSample(
                sample_id=sample_id,
                symbol=symbol,
                timeframe=timeframe,
                window_start=start_idx,
                window_end=end_idx,
                datetime=format_dt(end_dt),
                features=feature_vector,
                label=label_result,
                raw_prices=raw_prices
            )
            samples.append(sample)

        return samples, df
