import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from ML.feature_registry import FeatureRegistry
from ML.feature_pipeline import FeaturePipeline
from Market_Data_Pipeline.structure_graph import MarketStructureGraph, Zone

logger = logging.getLogger("DatasetBuilder")

class DatasetBuilder:
    """
    Purpose:
        Automatically construct labeled datasets (Market State and Level Break)
        from historical OHLCV data using MarketStructureGraph and analytical engines.
    """
    def __init__(self, registry: Optional[FeatureRegistry] = None):
        self.registry = registry or FeatureRegistry()
        self.pipeline = FeaturePipeline(self.registry)

    def build_market_state_dataset(self, df: pd.DataFrame, msg: MarketStructureGraph) -> pd.DataFrame:
        """
        Build Dataset A: Market State Dataset (One row = one candle).
        Labels: TREND, RANGE, TRANSITION.
        Stores confidence of each label.
        """
        rows = []
        enabled_features = [f.name for f in self.registry.list_enabled()]

        # Calculate features and labels for each index (skip warmup period)
        warmup = 100
        for idx in range(warmup, len(df)):
            feats = self.pipeline.extract_all(df, msg, idx)

            # Derive label based on objective rules
            label, confidence = self._determine_market_state(df, msg, idx)

            # Construct row
            row_data = {**feats, "label": label, "confidence": confidence, "timestamp": df.iloc[idx].get("Datetime", idx)}
            rows.append(row_data)

        dataset = pd.DataFrame(rows)
        return dataset

    def build_level_break_dataset(
        self,
        df: pd.DataFrame,
        msg: MarketStructureGraph,
        lookahead_bars: int = 20,
        break_threshold_atr: float = 0.5,
        rejection_threshold_atr: float = 1.0
    ) -> pd.DataFrame:
        """
        Build Dataset B: Level Break Dataset.
        One row = Price approaching significant supply/demand level.
        Target: 1 = level breaks, 0 = level rejects.
        """
        rows = []
        warmup = 100

        total_len = len(df) - lookahead_bars - warmup
        if total_len <= 0:
            return pd.DataFrame()

        logger.info(f"[{msg.symbol}] LEVEL BREAK PROCESSING: Starting level break labeling for {total_len} candles...")
        log_interval = max(1, total_len // 10)
        count = 0

        for idx in range(warmup, len(df) - lookahead_bars):
            count += 1
            row_close = df.iloc[idx]["Close"]

            if count % log_interval == 0 or count == total_len:
                pct = int(count / total_len * 100)
                logger.info(f"[{msg.symbol}] LEVEL BREAK PROCESSING: {pct}% complete ({count}/{total_len} bars)")
            row_high = df.iloc[idx]["High"]
            row_low = df.iloc[idx]["Low"]
            atr = df.iloc[idx].get("atr_14", 0.0001)

            # Identify nearest active zone
            # Active zones up to index 'idx'
            active_supplies = [z for z in msg.supply_zones if (z.created_idx <= idx and (not z.broken or z.broken_idx > idx))]
            active_demands = [z for z in msg.demand_zones if (z.created_idx <= idx and (not z.broken or z.broken_idx > idx))]

            # Check proximity to supply zone (close is within 0.5 ATR below lower bound)
            near_supply = None
            for s in active_supplies:
                if 0 < (s.lower - row_close) <= 0.5 * atr:
                    near_supply = s
                    break

            # Check proximity to demand zone (close is within 0.5 ATR above upper bound)
            near_demand = None
            for d in active_demands:
                if 0 < (row_close - d.upper) <= 0.5 * atr:
                    near_demand = d
                    break

            if not near_supply and not near_demand:
                continue

            # Perform lookahead to label 1 (break) or 0 (rejection)
            target = None
            if near_supply:
                # Supply Break: price exceeds upper boundary
                # Supply Reject: price drops below lower boundary by rejection_threshold_atr * ATR
                for l in range(1, lookahead_bars + 1):
                    future_bar = df.iloc[idx + l]
                    if future_bar["High"] > near_supply.upper:
                        target = 1  # Break
                        break
                    elif future_bar["Low"] < near_supply.lower - rejection_threshold_atr * atr:
                        target = 0  # Rejection
                        break
            elif near_demand:
                # Demand Break: price drops below lower boundary
                # Demand Reject: price rises above upper boundary by rejection_threshold_atr * ATR
                for l in range(1, lookahead_bars + 1):
                    future_bar = df.iloc[idx + l]
                    if future_bar["Low"] < near_demand.lower:
                        target = 1  # Break
                        break
                    elif future_bar["High"] > near_demand.upper + rejection_threshold_atr * atr:
                        target = 0  # Rejection
                        break

            if target is None:
                continue  # Indeterminate within lookahead window

            feats = self.pipeline.extract_all(df, msg, idx)
            row_data = {
                **feats,
                "target": target,
                "zone_type": "Supply" if near_supply else "Demand",
                "timestamp": df.iloc[idx].get("Datetime", idx)
            }
            rows.append(row_data)

        dataset = pd.DataFrame(rows)
        return dataset

    def _determine_market_state(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> Tuple[str, float]:
        """
        Core objective labeling logic for Market States.
        TREND: Repeated BOS and strong EMA alignment
        RANGE: Prolonged absence of BOS and multiple rejection at zones (or converged EMAs)
        TRANSITION: Everything else
        """
        row = df.iloc[idx]
        atr = row.get("atr_14", 0.0001)

        # Check EMA separation
        fast_ema = row.get("ema_50")
        slow_ema = row.get("ema_600", row.get("ema_800"))

        ema_aligned = False
        ema_separation_atr = 0.0
        if fast_ema is not None and slow_ema is not None:
            ema_separation_atr = abs(fast_ema - slow_ema) / (atr + 1e-9)
            if ema_separation_atr > 1.5:
                ema_aligned = True

        # Check BOS count in the last 100 bars
        bos_count = 0
        for b in msg.bos:
            if idx - 100 <= b.index <= idx:
                bos_count += 1

        # Alternating CHOCH check
        choch_count = 0
        for c in msg.choch:
            if idx - 100 <= c.index <= idx:
                choch_count += 1

        # repeated rejection at zones
        rejections = 0
        active_zones = [z for z in msg.supply_zones + msg.demand_zones if z.created_idx <= idx]
        for z in active_zones:
            if z.touch_count > 1:
                rejections += 1

        # Decision rules
        if ema_aligned and bos_count >= 1:
            confidence = min(1.0, 0.5 + (ema_separation_atr / 10.0) + (bos_count * 0.1))
            return "TREND", float(confidence)

        if (bos_count == 0 and choch_count >= 1) or rejections >= 2 or ema_separation_atr < 0.8:
            confidence = min(1.0, 0.6 + (choch_count * 0.05) + (rejections * 0.05))
            return "RANGE", float(confidence)

        # Default to Transition
        return "TRANSITION", 0.5
