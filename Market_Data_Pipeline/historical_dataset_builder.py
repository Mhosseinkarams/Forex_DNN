import os
import re
import sys
import time
import logging
import json
import copy
import psutil
import hashlib
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm

from Configs.path_manager import PathManager
from ML.feature_registry import FeatureRegistry
from ML.label_engine import LabelEngine
from ML.dataset_validator import DatasetValidator
from Market_Data_Pipeline.structure_engine import MarketStructureEngine
from Market_Data_Pipeline.supply_demand_engine import SupplyDemandEngine
from Market_Data_Pipeline.structure_graph import MarketStructureGraph
from Market_Data_Pipeline.pipeline import Pipeline
from Market_Data_Pipeline.cache_manager import DatasetCacheManager
from Market_Data_Pipeline.version_manager import DatasetVersionManager
from Market_Data_Pipeline.dataset_types import (
    MarketStructureResult,
    SupplyDemandResult,
    FeatureVector,
    LabelResult,
    DatasetSample
)

logger = logging.getLogger("HistoricalDatasetBuilder")

class HistoricalDatasetBuilder:
    """
    HistoricalDatasetBuilder converts raw historical OHLCV data into one unified ML dataset.
    Refactored to become the central data pipeline with high modularity, scalability, and performance.
    """
    def __init__(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        window_size: int = 35,
        window_stride: int = 1,
        timeframe: str = "M5",
        version: Optional[str] = None,
        registry: Optional[FeatureRegistry] = None,
        label_engine: Optional[LabelEngine] = None,
        ms_engine: Optional[MarketStructureEngine] = None,
        sd_engine: Optional[SupplyDemandEngine] = None,
        cache_dir: Optional[str] = None,
        datasets_dir: Optional[str] = None
    ):
        """
        Args:
            input_dir: Path to directory containing raw historical symbol data.
            output_dir: Path to directory where output datasets and metadata will be saved.
            window_size: Size of sliding window (default 35).
            window_stride: Stride of sliding window (default 1).
            timeframe: Target timeframe (default "M5").
            version: Unique version string for dataset naming (e.g. "v001"). Automatically derived if None.
            registry: Optional custom FeatureRegistry instance.
            label_engine: Optional custom LabelEngine instance.
            ms_engine: Optional custom MarketStructureEngine instance.
            sd_engine: Optional custom SupplyDemandEngine instance.
            cache_dir: Path to the cache directory.
            datasets_dir: Path to the datasets version manager directory.
        """
        self.input_dir = input_dir or PathManager.get_relative_path("historical_data")
        self.output_dir = output_dir or PathManager.get_relative_path("temporary")
        self.window_size = window_size
        self.window_stride = window_stride
        self.timeframe = timeframe
        self.version = version

        if cache_dir is None:
            cache_dir = PathManager.get_relative_path("cache")
        if datasets_dir is None:
            datasets_dir = PathManager.get_relative_path("datasets")

        # Ensure Directory Layout is cleanly initialized
        PathManager.ensure_all_dirs()

        self.registry = registry or FeatureRegistry(load_defaults=True)
        self.ms_engine = ms_engine or MarketStructureEngine(lookback=3)
        self.sd_engine = sd_engine or SupplyDemandEngine(atr_period=14, impulse_threshold=2.0)

        self.label_engine = label_engine or LabelEngine(
            window_size=self.window_size,
            window_stride=self.window_stride,
            registry=self.registry
        )

        # Initialize modular pipeline
        self.pipeline = Pipeline()

        # Add default Indicators
        from Collecting_Data.indicators import IndicatorEngine
        self.indicator_engine = IndicatorEngine(ema_periods=[50, 600, 800], slope_period=32)
        self.pipeline.register(self.indicator_engine)

        # Add MarketStructure and SupplyDemand engines to the pipeline
        self.pipeline.register(self.ms_engine)
        self.pipeline.register(self.sd_engine)

        # Cache & Version managers
        self.cache_manager = DatasetCacheManager(cache_dir=cache_dir)
        self.version_manager = DatasetVersionManager(output_dir=datasets_dir)

    def register_engine(self, engine: Any) -> "HistoricalDatasetBuilder":
        """
        Registers a plug-in engine to the builder's pipeline.
        This allows future analytical engines to be executed and their features extracted automatically.
        """
        self.pipeline.register(engine)
        return self

    def find_files(self) -> Dict[str, str]:
        """
        Discovers symbol files matching the specified timeframe.
        Supports both flat structure and nested structure.
        """
        discovered = {}
        if not os.path.exists(self.input_dir):
            logger.warning(f"Input directory does not exist: {self.input_dir}")
            return discovered

        # Try nested structure first: self.input_dir/SYMBOL/TIMEFRAME.parquet or .csv
        for item in os.listdir(self.input_dir):
            subdir = os.path.join(self.input_dir, item)
            if os.path.isdir(subdir):
                # Check for parquet
                parquet_path = os.path.join(subdir, f"{self.timeframe}.parquet")
                if os.path.exists(parquet_path):
                    discovered[item] = parquet_path
                    continue
                # Check for csv
                csv_path = os.path.join(subdir, f"{self.timeframe}.csv")
                if os.path.exists(csv_path):
                    discovered[item] = csv_path
                    continue

        # Try flat structure: self.input_dir/SYMBOL_TIMEFRAME.parquet or .csv
        for file in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, file)
            if os.path.isfile(file_path):
                match = re.match(r"^([A-Z0-9]+)_" + re.escape(self.timeframe) + r"\.(parquet|csv)$", file, re.IGNORECASE)
                if match:
                    symbol = match.group(1)
                    discovered[symbol] = file_path

        return discovered

    def _load_file(self, file_path: str) -> pd.DataFrame:
        """Loads Parquet or CSV file into a pandas DataFrame."""
        if file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
        return df

    def _get_engine_versions(self) -> Dict[str, str]:
        """Collects version strings for all pipeline engines."""
        versions = {
            "builder": "1.9",
            "feature_registry": f"{self.registry.compute_hash()[:8]}",
            "label_engine": getattr(self.label_engine, "label_version", "1.0.0"),
            "market_structure": "1.0.0",
            "supply_demand": "1.0.0"
        }
        for stage in self.pipeline.stages:
            name = stage.__class__.__name__
            if hasattr(stage, "version"):
                versions[name] = str(stage.version)
            elif hasattr(stage, "label_version"):
                versions[name] = str(stage.label_version)
        return versions

    def process_symbol(self, symbol: str, file_path: str) -> pd.DataFrame:
        """
        Processes a single symbol file thread-safely:
        loads data, executes modular pipeline, computes structure & zones,
        generates sliding window samples, and extracts features/labels.
        """
        # Read from cache if it exists for automatic resume support
        version_str = self.version or "v001"
        cached_df = self.cache_manager.get_cached_symbol(symbol, self.timeframe, version_str)
        if cached_df is not None:
            logger.info(f"[{symbol}] [CACHE] Found cached datasets. Loading directly...")
            return cached_df

        # Avoid reentrancy / thread safety issues with fresh local instances of engines in a clone pipeline
        local_pipeline = Pipeline()
        for stage in self.pipeline.stages:
            if hasattr(stage, "clone") and callable(stage.clone):
                local_pipeline.register(stage.clone())
            else:
                try:
                    local_pipeline.register(copy.deepcopy(stage))
                except Exception:
                    # Instantiate standard engines safely
                    name = stage.__class__.__name__
                    if name == "MarketStructureEngine":
                        local_pipeline.register(MarketStructureEngine(lookback=getattr(stage, "lookback", 3)))
                    elif name == "SupplyDemandEngine":
                        local_pipeline.register(SupplyDemandEngine(
                            atr_period=getattr(stage, "atr_period", 14),
                            impulse_threshold=getattr(stage, "impulse_threshold", 2.0)
                        ))
                    elif name == "IndicatorEngine":
                        from Collecting_Data.indicators import IndicatorEngine
                        local_pipeline.register(IndicatorEngine(
                            ema_periods=getattr(stage, "ema_periods", [50, 600, 800]),
                            atr_period=getattr(stage, "atr_period", 14),
                            slope_period=getattr(stage, "slope_period", 32)
                        ))
                    else:
                        local_pipeline.register(copy.copy(stage))

        logger.info(f"[{symbol}] [1/4] READING: Loading raw candles from {os.path.basename(file_path)}...")
        df_ohlcv = self._load_file(file_path)
        logger.info(f"[{symbol}] [1/4] READING: Completed. Loaded {len(df_ohlcv)} bars.")

        # Run sequential transformations
        logger.info(f"[{symbol}] [2/4] PROCESSING (INDICATORS/SMC/SD): Executing transformation pipeline stages...")
        df_transformed = local_pipeline.execute(df_ohlcv, symbol, self.timeframe)
        logger.info(f"[{symbol}] [2/4] PROCESSING: Completed indicator and structure calculations.")

        # Retrieve engines for building structure graph
        ms_stage = local_pipeline.get_stage("MarketStructureEngine") or local_pipeline.get_stage(MarketStructureEngine)
        sd_stage = local_pipeline.get_stage("SupplyDemandEngine") or local_pipeline.get_stage(SupplyDemandEngine)

        swing_highs = []
        swing_lows = []
        bos_list = []
        choch_list = []
        supply_zones = []
        demand_zones = []
        atr_val = 0.0001
        trend_dir = "Neutral"

        if ms_stage:
            swing_highs = [s for s in ms_stage.swings if s.level_type == "SwingHigh"]
            swing_lows = [s for s in ms_stage.swings if s.level_type == "SwingLow"]
            bos_list = list(ms_stage.bos_list)
            choch_list = list(ms_stage.choch_list)
            protected_high = ms_stage.protected_high
            protected_low = ms_stage.protected_low
        else:
            protected_high = None
            protected_low = None

        if sd_stage:
            supply_zones = [z for z in sd_stage.zones if z.type == "Supply"]
            demand_zones = [z for z in sd_stage.zones if z.type == "Demand"]

        if not df_transformed.empty:
            atr_val = float(df_transformed.iloc[-1].get("atr_14", 0.0001))
            trend_val = df_transformed.iloc[-1].get("trend", 0)
            trend_dir = "Bull" if trend_val == 1 else ("Bear" if trend_val == -1 else "Neutral")

        msg = MarketStructureGraph(
            symbol=symbol,
            timeframe=self.timeframe,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            protected_high=protected_high,
            protected_low=protected_low,
            bos=bos_list,
            choch=choch_list,
            supply_zones=supply_zones,
            demand_zones=demand_zones,
            trend_direction=trend_dir,
            atr=atr_val
        )
        msg.pipeline = local_pipeline

        # Sliding Windows
        n_bars = len(df_transformed)
        if n_bars < self.window_size:
            logger.warning(f"DataFrame for {symbol} has fewer bars ({n_bars}) than window size ({self.window_size}).")
            return pd.DataFrame()

        from ML.feature_pipeline import FeaturePipeline
        local_feature_pipeline = FeaturePipeline(self.registry)

        local_label_engine = LabelEngine(
            window_size=self.window_size,
            window_stride=self.window_stride,
            registry=self.registry,
            labeler=self.label_engine.labeler
        )

        samples: List[DatasetSample] = []

        logger.info(f"[{symbol}] [3/4] LABELING & FEATURE EXTRACTION: Generating labeled samples across {n_bars} bars...")
        total_windows = len(range(0, n_bars - self.window_size + 1, self.window_stride))
        log_interval = max(1, total_windows // 10)
        window_count = 0

        for start_idx in range(0, n_bars - self.window_size + 1, self.window_stride):
            window_count += 1
            end_idx = start_idx + self.window_size - 1

            # Step 1: Label window
            label, confidence, label_info = local_label_engine.labeler.label_window(
                df_transformed, msg, start_idx, end_idx
            )

            if window_count % log_interval == 0 or window_count == total_windows:
                pct = int(window_count / total_windows * 100)
                logger.info(f"[{symbol}] [3/4] LABELING & FEATURE EXTRACTION: {pct}% complete ({window_count}/{total_windows} windows)")

            if label is None:
                # Discard sample
                local_label_engine.removed_samples_count += 1
                reason = label_info.get("rule_fired", "unknown_ambiguous")
                local_label_engine.removal_reasons[reason] = local_label_engine.removal_reasons.get(reason, 0) + 1
                continue

            # Step 2: Compute feature vector (completely decoupled from features names in builder)
            feats_dict = local_feature_pipeline.extract_all(df_transformed, msg, idx=end_idx)

            feat_vector = FeatureVector(
                features=feats_dict,
                vector=None # Can lazy-compute or set numpy array if needed
            )

            label_res = LabelResult(
                label=label,
                confidence=confidence,
                info=label_info
            )

            # Step 3: Get start/end timestamps and generate deterministic sample_id
            start_row = df_transformed.iloc[start_idx]
            end_row = df_transformed.iloc[end_idx]

            start_dt = start_row["Datetime"]
            end_dt = end_row["Datetime"]

            start_dt_str = start_dt.isoformat() if isinstance(start_dt, pd.Timestamp) else str(start_dt)
            end_dt_str = end_dt.isoformat() if isinstance(end_dt, pd.Timestamp) else str(end_dt)

            # Deterministic ID: symbol_timeframe_windowEndDatetime
            formatted_end_dt = end_dt.strftime("%Y-%m-%dT%H:%M") if isinstance(end_dt, pd.Timestamp) else str(end_dt).replace(" ", "T")
            sample_id = f"{symbol}_{self.timeframe}_{formatted_end_dt}"

            # Extract raw prices and EMAs for metadata preservation
            raw_prices = {}
            for col in ["Open", "High", "Low", "Close", "TickVolume"]:
                if col in df_transformed.columns:
                    raw_prices[col] = df_transformed.iloc[end_idx][col]

            raw_emas = {}
            for col in ["ema_50", "ema_600", "ema_800"]:
                if col in df_transformed.columns:
                    raw_emas[col] = df_transformed.iloc[end_idx][col]

            sample = DatasetSample(
                sample_id=sample_id,
                symbol=symbol,
                timeframe=self.timeframe,
                window_start_datetime=start_dt_str,
                window_end_datetime=end_dt_str,
                feature_vector=feat_vector,
                label_result=label_res,
                raw_prices=raw_prices,
                raw_emas=raw_emas,
                metadata={
                    "window_start": start_idx,
                    "window_end": end_idx,
                    "label_version": local_label_engine.labeler.label_version,
                    "engine_version": "1.0.0"
                }
            )

            samples.append(sample)

        # Flatten into DataFrame
        flat_samples = [s.to_flat_dict() for s in samples]
        df_labeled = pd.DataFrame(flat_samples)

        # Cache results for next run
        logger.info(f"[{symbol}] [4/4] SAVING TO CACHE: Serializing dataset copy to Cache...")
        self.cache_manager.cache_symbol(symbol, self.timeframe, version_str, df_labeled)
        logger.info(f"[{symbol}] [4/4] SAVING TO CACHE: Saved cached datasets for {symbol} successfully.")

        return df_labeled

    def resolve_next_version(self) -> str:
        """Finds the next available version number."""
        if self.version:
            return self.version
        return self.version_manager.resolve_next_version()

    def build_dataset(self, max_workers: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Discovers symbol files, processes them in parallel (at symbol-level), performs data validation,
        generates the quality report, snapshots configs, and saves versioned datasets.
        """
        start_time = time.time()
        symbol_files = self.find_files()
        if not symbol_files:
            raise FileNotFoundError(f"No matching historical data files found in {self.input_dir} for timeframe {self.timeframe}")

        all_dfs = []
        symbols_list = sorted(list(symbol_files.keys()))
        version_str = self.resolve_next_version()
        self.version = version_str

        # Display TQDM Progress Monitor
        pbar = tqdm(symbols_list, desc="Processing Symbols", unit="symbol")

        workers = max_workers or min(32, os.cpu_count() or 1)

        if workers > 1 and len(symbols_list) > 1:
            logger.info(f"Processing symbols in parallel using {workers} workers...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_symbol = {
                    executor.submit(self.process_symbol, sym, path): sym
                    for sym, path in symbol_files.items()
                }
                for future in concurrent.futures.as_completed(future_to_symbol):
                    sym = future_to_symbol[future]
                    total_samples = sum(len(df) for df in all_dfs)
                    pbar.set_postfix({
                        "Current Symbol": sym,
                        "Memory Usage": f"{psutil.Process().memory_info().rss / (1024 * 1024):.1f} MB",
                        "Dataset Size": f"{total_samples} samples"
                    })
                    try:
                        df_sym = future.result()
                        if not df_sym.empty:
                            all_dfs.append(df_sym)
                        pbar.update(1)
                    except Exception as exc:
                        logger.error(f"Symbol {sym} generated an exception: {exc}", exc_info=True)
                        pbar.update(1)
        else:
            logger.info("Processing symbols sequentially...")
            for sym in symbols_list:
                total_samples = sum(len(df) for df in all_dfs)
                pbar.set_postfix({
                    "Current Symbol": sym,
                    "Memory Usage": f"{psutil.Process().memory_info().rss / (1024 * 1024):.1f} MB",
                    "Dataset Size": f"{total_samples} samples"
                })
                try:
                    df_sym = self.process_symbol(sym, symbol_files[sym])
                    if not df_sym.empty:
                        all_dfs.append(df_sym)
                except Exception as exc:
                    logger.error(f"Symbol {sym} generated an exception: {exc}", exc_info=True)
                pbar.update(1)

        pbar.close()

        if not all_dfs:
            raise ValueError("No training samples were generated across any symbols.")

        df_final = pd.concat(all_dfs, ignore_index=True)

        # Sort combined dataset by datetime
        if "datetime" in df_final.columns:
            df_final["datetime_parsed"] = pd.to_datetime(df_final["datetime"])
            df_final.sort_values(by=["datetime_parsed", "symbol"], inplace=True)
            df_final.drop(columns=["datetime_parsed"], inplace=True)
            df_final.reset_index(drop=True, inplace=True)

        # Data Validation Checks
        logger.info("Running dataset validation checks...")
        validator = DatasetValidator()
        validation_report = validator.validate(df_final, expected_window_size=self.window_size)

        # Calculate metrics for Quality Report / statistics.json
        elapsed_time = time.time() - start_time
        samples_per_sec = len(df_final) / elapsed_time if elapsed_time > 0 else 0.0

        total_cells = df_final.size
        nan_count = int(df_final.isnull().sum().sum())
        nan_pct = (nan_count / total_cells * 100.0) if total_cells > 0 else 0.0

        duplicate_rows = int(df_final.duplicated().sum())
        dup_pct = (duplicate_rows / len(df_final) * 100.0) if len(df_final) > 0 else 0.0

        memory_usage_bytes = df_final.memory_usage(deep=True).sum()
        memory_usage_mb = memory_usage_bytes / (1024 * 1024)

        # Check feature variance and constant columns
        numeric_cols = df_final.select_dtypes(include=[np.number]).columns
        exclude_cols = ["target", "confidence", "window_start", "window_end", "Open", "High", "Low", "Close", "TickVolume", "ema_50", "ema_600", "ema_800"]
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]

        feature_variances = {}
        constant_columns = []
        for col in feature_cols:
            var = float(df_final[col].var())
            feature_variances[col] = var
            if var == 0 or np.isnan(var) or df_final[col].nunique() <= 1:
                constant_columns.append(col)

        # Label & Symbol distribution
        label_dist = {}
        if "target" in df_final.columns:
            label_dist = df_final["target"].value_counts().to_dict()
            label_dist = {str(k): int(v) for k, v in label_dist.items()}

        symbol_dist = {}
        if "symbol" in df_final.columns:
            symbol_dist = df_final["symbol"].value_counts().to_dict()
            symbol_dist = {str(k): int(v) for k, v in symbol_dist.items()}

        # ----------------- DATASET FINGERPRINTING -----------------
        # Compute SHA256 of df content deterministically
        def compute_df_hash(df: pd.DataFrame) -> str:
            try:
                sorted_df = df.reindex(sorted(df.columns), axis=1)
                m = hashlib.sha256()
                for chunk in np.array_split(sorted_df.to_numpy().astype(str), max(1, len(sorted_df)//1000)):
                    m.update(chunk.tobytes())
                return m.hexdigest()
            except Exception:
                import pickle
                return hashlib.sha256(pickle.dumps(df.to_dict("list"))).hexdigest()

        dataset_hash = compute_df_hash(df_final)
        feature_hash = self.registry.compute_hash()

        # Compute Engine Hash
        engine_versions = self._get_engine_versions()
        engine_serialized = ",".join(f"{k}:{v}" for k, v in sorted(engine_versions.items()))
        engine_hash = hashlib.sha256(engine_serialized.encode("utf-8")).hexdigest()

        # Retrieve Git Commit Hash safely
        git_commit = "unknown"
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        except Exception:
            pass

        creation_time = datetime.now(timezone.utc).isoformat() if hasattr(datetime, "now") else datetime.utcnow().isoformat()
        # ----------------------------------------------------------

        # Construct snapshots & metadata
        feature_registry_snapshot = [f.to_dict() for f in self.registry.list_all()]

        label_config = {
            "window_size": self.window_size,
            "window_stride": self.window_stride,
            "label_version": getattr(self.label_engine.labeler, "label_version", "1.0.0"),
            "rules": {
                "ema_separation_trend": getattr(self.label_engine.labeler, "ema_sep_trend", 1.5),
                "ema_separation_range": getattr(self.label_engine.labeler, "ema_sep_range", 0.8),
                "min_bos_trend": getattr(self.label_engine.labeler, "min_bos_trend", 1),
                "min_rejections_range": getattr(self.label_engine.labeler, "min_rejections_range", 2)
            }
        }

        metadata = {
            "version": version_str,
            "dataset_version": version_str,
            "window_size": self.window_size,
            "window": self.window_size,
            "symbols": symbols_list,
            "timeframe": self.timeframe,
            "timeframes": [self.timeframe],
            "label_engine": f"LabelEngine v{engine_versions['label_engine']}",
            "feature_registry": f"FeatureRegistry v{engine_versions['feature_registry']}",
            "structure_engine": "MarketStructureEngine v1.0.0",
            "supply_demand_engine": "SupplyDemandEngine v1.0.0",
            "samples": len(df_final),
            "sample_count": len(df_final),
            "feature_count": len(self.registry.list_enabled()),
            "label_distribution": label_dist,
            "symbol_distribution": symbol_dist,
            "created": datetime.now().strftime("%Y-%m-%d"),
            "generation_date": datetime.now().isoformat(),
            "fingerprint": {
                "dataset_hash": dataset_hash,
                "feature_hash": feature_hash,
                "engine_hash": engine_hash,
                "git_commit": git_commit,
                "creation_time": creation_time
            },
            "validation": {
                "is_valid": validation_report["is_valid"],
                "errors": validation_report["errors"],
                "warnings": validation_report["warnings"],
                "nan_count": nan_count,
                "nan_percentage": float(nan_pct),
                "duplicate_rows": duplicate_rows,
                "duplicate_percentage": float(dup_pct),
                "constant_columns": constant_columns,
                "feature_variances": feature_variances,
                "memory_usage_mb": float(memory_usage_mb)
            }
        }

        statistics_json = {
            "Rows": len(df_final),
            "Columns": len(df_final.columns),
            "Memory_MB": float(memory_usage_mb),
            "Features": len(self.registry.list_enabled()),
            "Classes": len(label_dist.keys()),
            "Label_Distribution": label_dist,
            "Missing_Percentage": float(nan_pct),
            "Generation_Time_Sec": float(elapsed_time),
            "Average_Windows_Per_Sec": float(samples_per_sec),
            "Dataset_Size_Bytes": int(memory_usage_bytes),
            "Largest_Symbols": sorted(symbol_dist.items(), key=lambda x: x[1], reverse=True)[:5],
            "Smallest_Symbols": sorted(symbol_dist.items(), key=lambda x: x[1])[:5],
            "Fingerprint": {
                "dataset_hash": dataset_hash,
                "feature_hash": feature_hash,
                "engine_hash": engine_hash,
                "git_commit": git_commit,
                "creation_time": creation_time
            }
        }

        manifest = {
            "dataset_name": f"Forex_DNN_Dataset_{version_str}",
            "version": version_str,
            "creation_date": datetime.now().isoformat(),
            "symbols": symbols_list,
            "timeframes": [self.timeframe],
            "window_size": self.window_size,
            "number_of_samples": len(df_final),
            "feature_count": len(self.registry.list_enabled()),
            "builder_version": engine_versions["builder"],
            "engine_versions": engine_versions,
            "fingerprint": {
                "dataset_hash": dataset_hash,
                "feature_hash": feature_hash,
                "engine_hash": engine_hash,
                "git_commit": git_commit,
                "creation_time": creation_time
            }
        }

        # Generate HTML Quality Report (Step 3 implementation)
        html_report = validator.generate_quality_report_html(df_final, metadata, statistics_json)

        # Save to Version Manager
        self.version_manager.save_version(
            version=version_str,
            df=df_final,
            metadata=metadata,
            feature_registry_json=feature_registry_snapshot,
            engine_versions_json=engine_versions,
            label_config_json=label_config,
            statistics_json=statistics_json,
            manifest_json=manifest,
            quality_report_html=html_report
        )

        # Print beautiful quality report
        self._print_quality_report(metadata, len(df_final.columns), elapsed_time, samples_per_sec, constant_columns, symbol_dist)

        # Write duplicate dataset outputs inside output_dir for legacy backward compatibility atomically
        print("Saving Dataset copy to output folder...")
        os.makedirs(self.output_dir, exist_ok=True)

        def save_df_atomically(df_to_save, path, as_parquet=True):
            temp_path = path + ".tmp"
            if as_parquet:
                df_to_save.to_parquet(temp_path, index=False)
            else:
                df_to_save.to_csv(temp_path, index=False)
            os.replace(temp_path, path)

        def save_json_atomically(path, data):
            temp_path = path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            os.replace(temp_path, path)

        save_df_atomically(df_final, os.path.join(self.output_dir, f"dataset_{version_str}.parquet"), as_parquet=True)
        save_df_atomically(df_final, os.path.join(self.output_dir, f"dataset_{version_str}.csv"), as_parquet=False)
        save_json_atomically(os.path.join(self.output_dir, f"dataset_{version_str}_metadata.json"), metadata)

        return df_final, metadata

    def _print_quality_report(
        self,
        metadata: Dict[str, Any],
        col_count: int,
        elapsed_time: float,
        samples_per_sec: float,
        constant_columns: List[str],
        symbol_dist: Dict[str, int]
    ):
        """Prints a detailed, formatted quality report."""
        print("\n" + "=" * 50)
        print("                 Dataset Summary")
        print("=" * 50)
        print(f"Symbols             : {', '.join(metadata['symbols'])}")
        print(f"Timeframes          : {metadata['timeframe']}")
        print(f"Rows                : {metadata['samples']}")
        print(f"Columns             : {col_count}")
        print(f"Features            : {metadata['feature_count']}")
        print(f"Label Distribution  : {metadata['label_distribution']}")
        print(f"NaN %               : {metadata['validation']['nan_percentage']:.4f}%")
        print(f"Duplicate %         : {metadata['validation']['duplicate_percentage']:.4f}%")
        print(f"Memory Usage        : {metadata['validation']['memory_usage_mb']:.2f} MB")
        print(f"Processing Time     : {elapsed_time:.2f} seconds")
        print(f"Average Samples/sec : {samples_per_sec:.2f}")
        print("-" * 50)
        print("              DATA QUALITY VALIDATION")
        print("-" * 50)
        print(f"Is Valid            : {metadata['validation']['is_valid']}")
        print(f"Constant Columns    : {constant_columns if constant_columns else 'None'}")
        print(f"Validation Errors   : {len(metadata['validation']['errors'])}")
        for err in metadata['validation']['errors']:
            print(f"  - ERROR: {err}")
        print(f"Validation Warnings : {len(metadata['validation']['warnings'])}")
        for warn in metadata['validation']['warnings']:
            print(f"  - WARNING: {warn}")
        print("=" * 50 + "\n")
