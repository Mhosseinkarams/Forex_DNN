import json
import os
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
import numpy as np

from ML.feature_registry import FeatureRegistry
from ML.feature_pipeline import FeaturePipeline
from ML.market_state_labeler import BaseLabeler, MarketStateLabeler
from Market_Data_Pipeline.structure_engine import MarketStructureEngine
from Market_Data_Pipeline.supply_demand_engine import SupplyDemandEngine
from Market_Data_Pipeline.structure_graph import MarketStructureGraph

logger = logging.getLogger("LabelEngine")


class LabelEngine:
    """
    The central coordinator for rule-based, deterministic sliding-window labeling
    in the Forex_DNN framework. Integrates with FeatureRegistry, FeaturePipeline,
    MarketStructureEngine, and SupplyDemandEngine.
    """

    def __init__(
        self,
        window_size: int = 35,
        window_stride: int = 1,
        labeler: Optional[BaseLabeler] = None,
        registry: Optional[FeatureRegistry] = None
    ):
        """
        Args:
            window_size: The number of candles inside each sliding window (default 35).
            window_stride: The stride between consecutive windows (default 1).
            labeler: The labeling class implementing BaseLabeler (default is MarketStateLabeler).
            registry: The FeatureRegistry instance (defaults to loaded defaults).
        """
        self.window_size = window_size
        self.window_stride = window_stride
        self.labeler = labeler or MarketStateLabeler()
        self.registry = registry or FeatureRegistry(load_defaults=True)
        self.pipeline = FeaturePipeline(self.registry)

        # Statistics/Tracking
        self.total_windows_processed = 0
        self.removed_samples_count = 0
        self.removal_reasons: Dict[str, int] = {}

    def generate_dataset(
        self,
        data_inputs: Union[pd.DataFrame, Dict[Tuple[str, str], pd.DataFrame]],
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        ms_engine: Optional[MarketStructureEngine] = None,
        sd_engine: Optional[SupplyDemandEngine] = None
    ) -> pd.DataFrame:
        """
        Generates the sliding-window dataset across symbols and timeframes.

        Args:
            data_inputs: A single DataFrame (requires symbol and timeframe params)
                         or a dict keyed by (symbol, timeframe) -> DataFrame.
            symbol: Active symbol if data_inputs is a single DataFrame.
            timeframe: Active timeframe if data_inputs is a single DataFrame.
            ms_engine: Custom MarketStructureEngine instance (or default).
            sd_engine: Custom SupplyDemandEngine instance (or default).

        Returns:
            pd.DataFrame: Consolidated labeled dataset with features, target, and metadata.
        """
        # Clean/reset stats
        self.total_windows_processed = 0
        self.removed_samples_count = 0
        self.removal_reasons = {}

        # Standardize inputs to a dict
        df_dict: Dict[Tuple[str, str], pd.DataFrame] = {}
        if isinstance(data_inputs, pd.DataFrame):
            if not symbol or not timeframe:
                raise ValueError("Must specify symbol and timeframe when passing a single DataFrame.")
            df_dict[(symbol, timeframe)] = data_inputs
        else:
            df_dict = data_inputs

        all_samples = []

        # Engines
        m_engine = ms_engine or MarketStructureEngine(lookback=3)
        s_engine = sd_engine or SupplyDemandEngine(atr_period=14, impulse_threshold=2.0)

        for (sym, tf), df_ohlcv in df_dict.items():
            logger.info(f"LabelEngine processing {sym} {tf} with {len(df_ohlcv)} bars...")

            # Ensure indicators are calculated
            # Let's check if indicators are present; if not, we can run a simple indicator engine
            df_processed = df_ohlcv.copy()
            if "ema_50" not in df_processed.columns:
                from Collecting_Data.indicators import IndicatorEngine
                ind_engine = IndicatorEngine(ema_periods=[50, 600], slope_period=32)
                df_processed = ind_engine.calculate(df_processed)

            # Compute Market Structure and Supply Demand
            df_struct = m_engine.process(df_processed)
            df_final = s_engine.process(df_struct)

            # Build consolidated MarketStructureGraph for lookup
            # Swings must be separated by high/low in graph
            msg = MarketStructureGraph(
                symbol=sym,
                timeframe=tf,
                swing_highs=[s for s in m_engine.swings if s.level_type == "SwingHigh"],
                swing_lows=[s for s in m_engine.swings if s.level_type == "SwingLow"],
                protected_high=m_engine.protected_high,
                protected_low=m_engine.protected_low,
                bos=list(m_engine.bos_list),
                choch=list(m_engine.choch_list),
                supply_zones=[z for z in s_engine.zones if z.type == "Supply"],
                demand_zones=[z for z in s_engine.zones if z.type == "Demand"],
                trend_direction="Bull" if df_final.iloc[-1].get("trend", 0) == 1 else ("Bear" if df_final.iloc[-1].get("trend", 0) == -1 else "Neutral"),
                atr=float(df_final.iloc[-1].get("atr_14", 0.0001))
            )

            n_bars = len(df_final)
            if n_bars < self.window_size:
                logger.warning(f"DataFrame for {sym} {tf} has fewer bars ({n_bars}) than window size ({self.window_size}). Skipping.")
                continue

            # Sliding Window generation
            # Slide start index across df
            for start_idx in range(0, n_bars - self.window_size + 1, self.window_stride):
                end_idx = start_idx + self.window_size - 1
                self.total_windows_processed += 1

                # Step 1: Labeling of current window
                label, confidence, label_info = self.labeler.label_window(
                    df_final, msg, start_idx, end_idx
                )

                if label is None:
                    # Indeterminate sample - remove/skip
                    self.removed_samples_count += 1
                    reason = label_info.get("rule_fired", "unknown_ambiguous")
                    self.removal_reasons[reason] = self.removal_reasons.get(reason, 0) + 1
                    continue

                # Step 2: Feature Extraction at window_end
                # All historical values up to end_idx are safe to use
                feats = self.pipeline.extract_all(df_final, msg, idx=end_idx)

                # Step 3: Package Metadata & target
                row_datetime = df_final.iloc[end_idx].get("Datetime")
                if isinstance(row_datetime, pd.Timestamp):
                    datetime_str = row_datetime.isoformat()
                else:
                    datetime_str = str(row_datetime)

                row_data = {
                    **feats,
                    "target": label,
                    "confidence": confidence,
                    # Metadata
                    "symbol": sym,
                    "timeframe": tf,
                    "window_start": start_idx,
                    "window_end": end_idx,
                    "datetime": datetime_str,
                    "label_version": self.labeler.label_version,
                    "engine_version": "1.0.0",
                }

                # Add individual rule info for explanation
                for k, v in label_info.items():
                    row_data[f"meta_labeler_{k}"] = v

                all_samples.append(row_data)

        logger.info(f"Dataset generation complete. Total windows: {self.total_windows_processed}. Labeled: {len(all_samples)}. Removed: {self.removed_samples_count}.")
        for r, cnt in self.removal_reasons.items():
            logger.info(f"  Removed due to '{r}': {cnt} samples.")

        return pd.DataFrame(all_samples)

    def save_dataset_and_manifest(
        self,
        df: pd.DataFrame,
        output_path: str,
        manifest_path: str,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Saves the generated DataFrame to a CSV file, and writes a detailed
        dataset reproducibility manifest containing all critical hyper-parameters.
        """
        # Ensure directories exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)

        # 1. Save CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset CSV successfully saved to: {output_path}")

        # 2. Compute date range
        date_range = {"start": None, "end": None}
        if not df.empty and "datetime" in df.columns:
            date_range["start"] = str(df["datetime"].min())
            date_range["end"] = str(df["datetime"].max())

        # 3. Class distribution
        class_dist = {}
        if not df.empty and "target" in df.columns:
            class_dist = df["target"].value_counts().to_dict()
            # Convert numpy values to native Python types
            class_dist = {str(k): int(v) for k, v in class_dist.items()}

        # 4. Extract symbols and timeframes
        symbols_list = []
        timeframes_list = []
        if not df.empty:
            if "symbol" in df.columns:
                symbols_list = sorted(df["symbol"].unique().tolist())
            if "timeframe" in df.columns:
                timeframes_list = sorted(df["timeframe"].unique().tolist())

        # 5. Build Manifest JSON
        manifest = {
            "window_size": int(self.window_size),
            "window_stride": int(self.window_stride),
            "label_version": self.labeler.label_version,
            "feature_registry_version": self.registry.compute_hash(),
            "market_structure_engine_version": "1.0.0",
            "supply_demand_engine_version": "1.0.0",
            "symbols": symbols_list,
            "timeframes": timeframes_list,
            "date_range": date_range,
            "total_windows_generated": int(self.total_windows_processed),
            "samples_removed_due_to_missing_labels": int(self.removed_samples_count),
            "removal_reasons_distribution": self.removal_reasons,
            "final_class_distribution": class_dist,
            "dataset_rows": len(df)
        }

        if extra_metadata:
            manifest["extra_metadata"] = extra_metadata

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)

        logger.info(f"Dataset manifest successfully saved to: {manifest_path}")
