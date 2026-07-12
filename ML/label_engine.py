import os
import json
import logging
import copy
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

from ML.feature_registry import FeatureRegistry
from ML.feature_pipeline import FeaturePipeline
from Market_Data_Pipeline.structure_engine import MarketStructureEngine
from Market_Data_Pipeline.supply_demand_engine import SupplyDemandEngine
from Market_Data_Pipeline.structure_graph import MarketStructureGraph
from Collecting_Data.indicators import IndicatorEngine

logger = logging.getLogger("LabelEngine")


class BaseLabeler:
    """
    Base class/interface for strategy-based, deterministic labelers.
    Extend this to implement custom labeling rules.
    """
    label_version: str = "1.0"
    engine_version: str = "1.0"

    def label_window(
        self,
        df: pd.DataFrame,
        msg: MarketStructureGraph,
        window_start: int,
        window_end: int
    ) -> Tuple[Optional[str], float, Optional[str]]:
        """
        Labels a specific window.

        Returns:
            - label (str or None): The generated target label. If None, the sample is removed.
            - confidence (float): The confidence score of the label (0.0 to 1.0).
            - reason (str or None): Explanation/reasoning for the assigned label or rejection.
        """
        raise NotImplementedError


class LabelEngine:
    """
    Responsible for generating deterministic, rule-based labels for machine learning datasets.
    Supports a configurable sliding window, feature extraction via FeaturePipeline, and saving
    detailed metadata and reproducible manifests.
    """
    def __init__(
        self,
        window_size: int = 35,
        window_stride: int = 1,
        registry: Optional[FeatureRegistry] = None,
        struct_engine: Optional[MarketStructureEngine] = None,
        sd_engine: Optional[SupplyDemandEngine] = None,
        indicator_engine: Optional[IndicatorEngine] = None,
    ):
        self.window_size = window_size
        self.window_stride = window_stride
        self.registry = registry or FeatureRegistry()
        self.pipeline = FeaturePipeline(self.registry)
        self.struct_engine = struct_engine or MarketStructureEngine(lookback=3)
        self.sd_engine = sd_engine or SupplyDemandEngine(atr_period=14, impulse_threshold=2.0)
        self.indicator_engine = indicator_engine or IndicatorEngine(ema_periods=[50, 600, 800])

    def generate_point_in_time_graph(
        self,
        symbol: str,
        timeframe: str,
        df_enriched: pd.DataFrame,
        end_idx: int,
        full_swings: list,
        full_bos: list,
        full_choch: list,
        full_zones: list
    ) -> MarketStructureGraph:
        """
        Constructs a point-in-time correct MarketStructureGraph at end_idx by filtering
        the pre-computed lists and resetting mitigations/breakages occurring after end_idx.
        This completely eliminates lookahead bias (data leakage) during training.
        """
        row = df_enriched.iloc[end_idx]
        dt = row["Datetime"] if "Datetime" in df_enriched.columns else datetime.now()

        # 1. Swings: s is confirmed and active if index + confirmation_delay <= end_idx
        swings = [s for s in full_swings if s.index + s.confirmation_delay <= end_idx]
        swing_highs = [s for s in swings if s.level_type == "SwingHigh"]
        swing_lows = [s for s in swings if s.level_type == "SwingLow"]

        # 2. Breaks: bos or choch occurred at or before end_idx
        bos_list = [b for b in full_bos if b.index <= end_idx]
        choch_list = [c for c in full_choch if c.index <= end_idx]

        # 3. Supply/Demand Zones
        supply_zones = []
        demand_zones = []

        for z in full_zones:
            if z.created_idx <= end_idx:
                # Deep copy to prevent side effects or modifying the full list in memory
                z_pit = copy.deepcopy(z)

                # Reset mitigation if it occurred after end_idx
                if z_pit.mitigated and z_pit.mitigated_idx is not None and z_pit.mitigated_idx > end_idx:
                    z_pit.mitigated = False
                    z_pit.mitigated_idx = None
                    z_pit.freshness = True
                    z_pit.touch_count = 0
                    z_pit.number_of_reactions = 0
                    z_pit.freshness_score = 1.0

                # Reset breakage if it occurred after end_idx
                if z_pit.broken and z_pit.broken_idx is not None and z_pit.broken_idx > end_idx:
                    z_pit.broken = False
                    z_pit.broken_idx = None
                    z_pit.active = True
                    z_pit.invalidated = False

                if z_pit.type == "Supply":
                    supply_zones.append(z_pit)
                else:
                    demand_zones.append(z_pit)

        # 4. Resolve protected high/low based on point-in-time active swings
        protected_high = None
        protected_low = None

        valid_highs = [s for s in swing_highs if not s.broken or s.index > end_idx]
        valid_lows = [s for s in swing_lows if not s.broken or s.index > end_idx]

        if valid_highs:
            protected_high = max(valid_highs, key=lambda s: s.index)
        if valid_lows:
            protected_low = max(valid_lows, key=lambda s: s.index)

        # 5. Extract scalar attributes
        atr_val = float(row.get("atr_14", 0.0001))
        fast_ema = row.get("ema_50", 0.0)
        slow_ema = row.get("ema_600", row.get("ema_800", 0.0))
        ema_separation = abs(fast_ema - slow_ema) / (atr_val + 1e-9)

        return MarketStructureGraph(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=dt,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            protected_high=protected_high,
            protected_low=protected_low,
            bos=bos_list,
            choch=choch_list,
            supply_zones=supply_zones,
            demand_zones=demand_zones,
            trend_direction="Bull" if row.get("trend", 0) == 1 else ("Bear" if row.get("trend", 0) == -1 else "Neutral"),
            ema_distance_atr=ema_separation,
            atr=atr_val,
            volatility=float(atr_val * 10000.0)
        )

    def generate(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        labeler: BaseLabeler,
        output_csv_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Processes the input DataFrame, enriches it with indicators and structural states,
        performs sliding window iteration, extracts features, generates rule-based labels,
        filters out unlabeled samples, and optionally saves the dataset and its manifest.
        """
        logger.info(f"Generating labeled dataset for {symbol} {timeframe} (window_size={self.window_size}, stride={self.window_stride})")

        if len(df) < self.window_size:
            raise ValueError(f"Input DataFrame is too short ({len(df)}) for window_size={self.window_size}")

        df = df.copy()

        # Ensure datetime is parsed and sorted
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df = df.sort_values("Datetime").reset_index(drop=True)

        # 1. Run indicators calculation if not already present
        required_indicators = ["ema_50", "atr_14"]
        if not all(col in df.columns for col in required_indicators):
            logger.info("Computing technical indicators via IndicatorEngine...")
            df = self.indicator_engine.calculate(df)

        # 2. Run deterministic structural engines on full DataFrame
        logger.info("Running MarketStructureEngine and SupplyDemandEngine...")
        df_struct = self.struct_engine.process(df)
        df_enriched = self.sd_engine.process(df_struct)

        # Cache lists for dynamic point-in-time correct graph construction inside loop
        full_swings = list(self.struct_engine.swings)
        full_bos = list(self.struct_engine.bos_list)
        full_choch = list(self.struct_engine.choch_list)
        full_zones = list(self.sd_engine.zones)

        # 3. Sliding Window processing
        samples = []
        removed_reasons: Dict[str, int] = {}
        total_windows = 0

        # Loop from index 0 to len(df_enriched) - window_size
        n_rows = len(df_enriched)
        for start_idx in range(0, n_rows - self.window_size + 1, self.window_stride):
            end_idx = start_idx + self.window_size - 1
            total_windows += 1

            # Build a point-in-time MarketStructureGraph specifically up to end_idx to prevent lookahead bias!
            msg_pit = self.generate_point_in_time_graph(
                symbol=symbol,
                timeframe=timeframe,
                df_enriched=df_enriched,
                end_idx=end_idx,
                full_swings=full_swings,
                full_bos=full_bos,
                full_choch=full_choch,
                full_zones=full_zones
            )

            # Determine the label for this window using point-in-time graph
            label, confidence, reason = labeler.label_window(df_enriched, msg_pit, start_idx, end_idx)

            if label is None:
                reason_str = reason or "unknown_reason"
                removed_reasons[reason_str] = removed_reasons.get(reason_str, 0) + 1
                continue

            # Extract features relative to the end of the window (using point-in-time graph)
            features = self.pipeline.extract_all(df_enriched, msg_pit, idx=end_idx)

            # Build row
            timestamp_val = df_enriched.iloc[end_idx]["Datetime"] if "Datetime" in df_enriched.columns else str(end_idx)
            row_data = {
                **features,
                "label": label,
                "confidence": float(confidence),
                # Metadata
                "symbol": symbol,
                "timeframe": timeframe,
                "window_start": int(start_idx),
                "window_end": int(end_idx),
                "datetime": str(timestamp_val),
                # Labeling metadata
                "label_version": labeler.label_version,
                "engine_version": labeler.engine_version,
                "label_reason": reason or ""
            }
            samples.append(row_data)

        # Create output DataFrame
        dataset_df = pd.DataFrame(samples)

        # If we have no samples, handle empty dataframe gracefully
        if dataset_df.empty:
            logger.warning("No samples were generated after rule-based labeling and filtering.")
            # Create an empty dataframe with correct columns
            feature_cols = [f.name for f in self.registry.list_enabled()]
            meta_cols = ["label", "confidence", "symbol", "timeframe", "window_start", "window_end", "datetime", "label_version", "engine_version", "label_reason"]
            dataset_df = pd.DataFrame(columns=feature_cols + meta_cols)

        # Log removal summary
        total_removed = sum(removed_reasons.values())
        logger.info(f"Sliding window completed. Total windows: {total_windows}, Retained: {len(dataset_df)}, Removed: {total_removed}")
        for r_reason, r_count in removed_reasons.items():
            logger.info(f" - Removed due to '{r_reason}': {r_count}")

        # Calculate class distribution
        class_distribution = {}
        if not dataset_df.empty and "label" in dataset_df.columns:
            counts = dataset_df["label"].value_counts()
            total_retained = len(dataset_df)
            for c_lbl, cnt in counts.items():
                class_distribution[str(c_lbl)] = {
                    "count": int(cnt),
                    "percentage": float(cnt / total_retained * 100.0)
                }

        # 5. Save dataset and manifest if requested
        if output_csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
            dataset_df.to_csv(output_csv_path, index=False)
            logger.info(f"Dataset CSV saved to {output_csv_path}")

            # Manifest fields
            date_range = {"start": None, "end": None}
            if "Datetime" in df_enriched.columns:
                date_range["start"] = str(df_enriched.iloc[0]["Datetime"])
                date_range["end"] = str(df_enriched.iloc[-1]["Datetime"])

            manifest = {
                "window_size": int(self.window_size),
                "window_stride": int(self.window_stride),
                "label_version": labeler.label_version,
                "feature_registry_version": self.registry.compute_hash(),
                "market_structure_engine_version": "1.0",
                "supply_demand_engine_version": "1.0",
                "symbols": [symbol],
                "timeframes": [timeframe],
                "date_range": date_range,
                "total_windows_generated": int(total_windows),
                "samples_removed": {
                    "total": int(total_removed),
                    "by_reason": removed_reasons
                },
                "final_class_distribution": class_distribution,
                "generated_at": datetime.now().isoformat()
            }

            manifest_path = output_csv_path + ".manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=4)
            logger.info(f"Dataset manifest saved to {manifest_path}")

        return dataset_df
