import os
import re
import sys
import time
import logging
import json
import hashlib
import copy
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm
import psutil

from ML.feature_registry import FeatureRegistry
from ML.label_engine import LabelEngine
from ML.dataset_validator import DatasetValidator
from Market_Data_Pipeline.structure_engine import MarketStructureEngine
from Market_Data_Pipeline.supply_demand_engine import SupplyDemandEngine
from Market_Data_Pipeline.pipeline_types import (
    MarketStructureResult,
    SupplyDemandResult,
    FeatureVector,
    LabelResult,
    DatasetSample,
    Pipeline
)

logger = logging.getLogger("HistoricalDatasetBuilder")


class DatasetVersionManager:
    """
    Manages dataset versions dynamically under datasets/ directory.
    Ensures previous datasets are never overwritten.
    """
    def __init__(self, base_dir: str = "datasets"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def resolve_next_version(self) -> str:
        max_v = 0
        if os.path.exists(self.base_dir):
            for item in os.listdir(self.base_dir):
                if os.path.isdir(os.path.join(self.base_dir, item)):
                    match = re.match(r"^v(\d+)$", item)
                    if match:
                        v_num = int(match.group(1))
                        if v_num > max_v:
                            max_v = v_num
        next_v = max_v + 1
        return f"v{next_v:03d}"

    def get_version_dir(self, version: str) -> str:
        return os.path.join(self.base_dir, version)


class HistoricalDatasetBuilder:
    """
    HistoricalDatasetBuilder converts raw historical OHLCV data into one unified ML dataset.
    This module sits between raw historical data and the LabelEngine, and is the single source of truth
    for generating ML training data across symbols and timeframes.
    """
    def __init__(
        self,
        input_dir: str = "HistoricalData",
        output_dir: str = "output",
        window_size: int = 35,
        window_stride: int = 1,
        timeframe: str = "M5",
        version: Optional[str] = None,
        registry: Optional[FeatureRegistry] = None,
        label_engine: Optional[LabelEngine] = None,
        ms_engine: Optional[MarketStructureEngine] = None,
        sd_engine: Optional[SupplyDemandEngine] = None
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
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.window_size = window_size
        self.window_stride = window_stride
        self.timeframe = timeframe
        self.version = version

        # Pipeline steps configuration
        self.pipeline = Pipeline()

        self.ms_engine = ms_engine or MarketStructureEngine(lookback=3)
        self.sd_engine = sd_engine or SupplyDemandEngine(atr_period=14, impulse_threshold=2.0)
        self.registry = registry or FeatureRegistry(load_defaults=True)

        self.pipeline.register(self.ms_engine)
        self.pipeline.register(self.sd_engine)
        self.pipeline.register(self.registry)

        self.label_engine = label_engine or LabelEngine(
            window_size=self.window_size,
            window_stride=self.window_stride,
            registry=self.registry
        )
        self.pipeline.register(self.label_engine)

        self.version_manager = DatasetVersionManager()

    def register_engine(self, engine: Any):
        """
        Registers custom Smart Money or indicator engines to the pipeline.
        Allowing future engines without modification.
        """
        insert_idx = len(self.pipeline.steps)
        for idx, step in enumerate(self.pipeline.steps):
            if isinstance(step, (FeatureRegistry, LabelEngine)) or step.__class__.__name__ == "LabelEngine":
                insert_idx = idx
                break
        self.pipeline.steps.insert(insert_idx, engine)
        logger.info(f"Successfully registered custom engine: {engine.__class__.__name__}")
        return self

    def _get_cache_dir(self) -> str:
        """
        Generate cache hash based on configuration to prevent cache mismatch.
        """
        config_str = f"{self.window_size}_{self.window_stride}_{self.timeframe}_{self.registry.compute_hash()}"
        cache_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]
        cache_dir = os.path.join("cache", cache_hash)
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def find_files(self) -> Dict[str, str]:
        """
        Discovers symbol files matching the specified timeframe.
        Supports both flat structure: <input_dir>/<SYMBOL>_<TIMEFRAME>.parquet
        and nested structure: <input_dir>/<SYMBOL>/<TIMEFRAME>.parquet (and .csv)
        """
        discovered = {}
        if not os.path.exists(self.input_dir):
            logger.warning(f"Input directory does not exist: {self.input_dir}")
            return discovered

        # Try to resolve symbol folder nested structure first: self.input_dir/SYMBOL/TIMEFRAME.parquet or .csv
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

        # Try to resolve flat structure: self.input_dir/SYMBOL_TIMEFRAME.parquet or .csv
        for file in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, file)
            if os.path.isfile(file_path):
                # We expect something like EURUSD_M5.parquet
                match = re.match(r"^([A-Z0-9]+)_" + re.escape(self.timeframe) + r"\.(parquet|csv)$", file, re.IGNORECASE)
                if match:
                    symbol = match.group(1)
                    discovered[symbol] = file_path

        return discovered

    def _load_file(self, file_path: str) -> pd.DataFrame:
        """
        Loads Parquet or CSV file into a pandas DataFrame.
        """
        if file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        # Ensure "Datetime" column is datetime type
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
        return df

    def process_symbol(self, symbol: str, file_path: str) -> pd.DataFrame:
        """
        Processes a single symbol file: loads it, runs engines, extracts features and labels.
        Implements automatic caching and resume.
        """
        cache_dir = self._get_cache_dir()
        cache_file = os.path.join(cache_dir, f"{symbol}_processed.parquet")

        if os.path.exists(cache_file):
            logger.info(f"Loading {symbol} from cache...")
            try:
                df_cached = pd.read_parquet(cache_file)
                if not df_cached.empty:
                    return df_cached
            except Exception as e:
                logger.warning(f"Error reading cache for {symbol}: {e}. Recomputing...")

        df_ohlcv = self._load_file(file_path)

        # Thread Safety: deepcopy the pipeline so that concurrent worker threads operate on independent step instances
        local_pipeline = copy.deepcopy(self.pipeline)

        # Execute registered pipeline
        samples, df_processed = local_pipeline.execute(
            df_ohlcv=df_ohlcv,
            symbol=symbol,
            timeframe=self.timeframe,
            window_size=self.window_size,
            window_stride=self.window_stride
        )

        if not samples:
            logger.warning(f"No labeled samples generated for {symbol}.")
            df_empty = pd.DataFrame()
            df_empty.to_parquet(cache_file)
            return df_empty

        # Convert DatasetSamples to DataFrame
        rows = []
        for sample in samples:
            row_data = {
                "sample_id": sample.sample_id,
                "symbol": sample.symbol,
                "timeframe": sample.timeframe,
                "window_start": sample.window_start,
                "window_end": sample.window_end,
                "datetime": sample.datetime,
                "target": sample.label.label,
                "confidence": sample.label.confidence,
            }
            # Add features dynamically
            row_data.update(sample.features.features)
            # Add raw prices
            row_data.update(sample.raw_prices)
            # Add diagnostic labeler info
            for k, v in sample.label.label_info.items():
                row_data[f"meta_labeler_{k}"] = v

            # Automatically forward any extra columns added to df_processed by custom plugins/engines!
            for col in df_processed.columns:
                if col not in row_data and col not in ["Datetime", "Spread"]:
                    row_data[col] = df_processed.iloc[sample.window_end][col]

            rows.append(row_data)

        df_labeled = pd.DataFrame(rows)

        # Cache results
        df_labeled.to_parquet(cache_file)
        return df_labeled

    def resolve_next_version(self) -> str:
        """
        Finds the next available version number automatically from output_dir or DatasetVersionManager.
        """
        if self.version:
            return self.version
        return self.version_manager.resolve_next_version()

    def build_dataset(self, max_workers: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Discovers symbol files, processes them in parallel or sequentially, performs data validation,
        generates the quality report, and saves Parquet, CSV, and Metadata files.
        """
        start_time = time.time()
        symbol_files = self.find_files()
        if not symbol_files:
            raise FileNotFoundError(f"No matching historical data files found in {self.input_dir} for timeframe {self.timeframe}")

        all_dfs = []
        pbar = tqdm(total=len(symbol_files), desc="Generating Dataset")

        def update_pbar(sym, df_len):
            mem = psutil.Process().get_memory_info().rss / (1024 * 1024) if hasattr(psutil.Process(), "get_memory_info") else psutil.Process().memory_info().rss / (1024 * 1024)
            current_dataset_size = sum(len(d) for d in all_dfs) + df_len
            pbar.set_postfix({
                "Symbol": sym,
                "Mem": f"{mem:.1f}MB",
                "Size": current_dataset_size
            })
            pbar.update(1)

        if max_workers is None or max_workers > 1:
            workers = max_workers or min(32, os.cpu_count() or 1)
            logger.info(f"Processing symbols in parallel using {workers} threads...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_symbol = {
                    executor.submit(self.process_symbol, sym, path): sym
                    for sym, path in symbol_files.items()
                }
                for future in concurrent.futures.as_completed(future_to_symbol):
                    sym = future_to_symbol[future]
                    try:
                        df_sym = future.result()
                        if not df_sym.empty:
                            all_dfs.append(df_sym)
                        update_pbar(sym, len(df_sym))
                    except Exception as exc:
                        logger.error(f"Symbol {sym} generated an exception: {exc}", exc_info=True)
                        pbar.update(1)
        else:
            logger.info("Processing symbols sequentially...")
            for sym, path in symbol_files.items():
                try:
                    df_sym = self.process_symbol(sym, path)
                    if not df_sym.empty:
                        all_dfs.append(df_sym)
                    update_pbar(sym, len(df_sym))
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

        # Resolve dataset version
        version_str = self.resolve_next_version()

        # Data Validation Checks
        logger.info("Running dataset validation checks...")
        validator = DatasetValidator()
        validation_report = validator.validate(df_final, expected_window_size=self.window_size)

        # Calculate metrics for Quality Report
        elapsed_time = time.time() - start_time
        samples_per_sec = len(df_final) / elapsed_time if elapsed_time > 0 else 0.0

        total_cells = df_final.size
        nan_count = int(df_final.isnull().sum().sum())
        nan_pct = (nan_count / total_cells * 100.0) if total_cells > 0 else 0.0

        duplicate_rows = int(df_final.duplicated().sum())
        dup_pct = (duplicate_rows / len(df_final) * 100.0) if len(df_final) > 0 else 0.0

        # Estimate memory usage
        memory_usage_bytes = df_final.memory_usage(deep=True).sum()
        memory_usage_mb = memory_usage_bytes / (1024 * 1024)

        # Dynamic Feature Variance check (no feature knowledge!)
        enabled_features = [f.name for f in self.registry.list_enabled()]
        feature_cols = [c for c in df_final.columns if c in enabled_features]

        feature_variances = {}
        constant_columns = []
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df_final[col]):
                var = float(df_final[col].var())
                feature_variances[col] = var
                if var == 0 or np.isnan(var) or df_final[col].nunique() <= 1:
                    constant_columns.append(col)
            else:
                if df_final[col].nunique() <= 1:
                    constant_columns.append(col)

        # Class distribution
        label_dist = {}
        if "target" in df_final.columns:
            label_dist = df_final["target"].value_counts().to_dict()
            label_dist = {str(k): int(v) for k, v in label_dist.items()}

        # Symbol distribution (imbalance)
        symbol_dist = {}
        if "symbol" in df_final.columns:
            symbol_dist = df_final["symbol"].value_counts().to_dict()
            symbol_dist = {str(k): int(v) for k, v in symbol_dist.items()}

        # Save files under DatasetVersionManager output directory
        version_dir = self.version_manager.get_version_dir(version_str)
        os.makedirs(version_dir, exist_ok=True)

        parquet_path = os.path.join(version_dir, "dataset.parquet")
        csv_path = os.path.join(version_dir, "dataset.csv")
        df_final.to_parquet(parquet_path, index=False)
        df_final.to_csv(csv_path, index=False)

        # For backward compatibility, also write to the older output path
        os.makedirs(self.output_dir, exist_ok=True)
        compat_parquet_path = os.path.join(self.output_dir, f"dataset_{version_str}.parquet")
        compat_csv_path = os.path.join(self.output_dir, f"dataset_{version_str}.csv")
        compat_metadata_path = os.path.join(self.output_dir, f"dataset_{version_str}_metadata.json")
        df_final.to_parquet(compat_parquet_path, index=False)
        df_final.to_csv(compat_csv_path, index=False)

        # 1. metadata.json
        metadata = {
            "version": version_str,
            "dataset_version": version_str,
            "window_size": self.window_size,
            "window": self.window_size,
            "symbols": sorted(list(symbol_files.keys())),
            "timeframe": self.timeframe,
            "timeframes": [self.timeframe],
            "label_engine": "LabelEngine v1.0.0",
            "feature_registry": f"FeatureRegistry v{self.registry.compute_hash()[:8]}",
            "structure_engine": "MarketStructureEngine v1.0.0",
            "supply_demand_engine": "SupplyDemandEngine v1.0.0",
            "samples": len(df_final),
            "sample_count": len(df_final),
            "feature_count": len(feature_cols),
            "label_distribution": label_dist,
            "symbol_distribution": symbol_dist,
            "created": datetime.now().strftime("%Y-%m-%d"),
            "generation_date": datetime.now().isoformat(),
            "validation": {
                "is_valid": validation_report.is_valid,
                "errors": validation_report.errors,
                "warnings": validation_report.warnings,
                "nan_count": nan_count,
                "nan_percentage": float(nan_pct),
                "duplicate_rows": duplicate_rows,
                "duplicate_percentage": float(dup_pct),
                "constant_columns": constant_columns,
                "feature_variances": feature_variances,
                "memory_usage_mb": float(memory_usage_mb)
            }
        }

        # Save metadata.json
        with open(os.path.join(version_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        with open(compat_metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        # 2. feature_registry.json
        registry_features = []
        for f in self.registry.list_all():
            registry_features.append({
                "name": f.name,
                "description": f.description,
                "normalization": getattr(f, "normalize", True),
                "category": f.category,
                "dependencies": getattr(f, "dependencies", [])
            })
        with open(os.path.join(version_dir, "feature_registry.json"), "w") as f:
            json.dump(registry_features, f, indent=4)

        # 3. engine_versions.json
        engine_versions = {
            "dataset_version": version_str,
            "feature_registry": getattr(self.registry, "version", "4.2"),
            "label_engine": "2.1",
            "market_structure": "3.0",
            "supply_demand": "2.4",
            "builder": "1.8"
        }
        with open(os.path.join(version_dir, "engine_versions.json"), "w") as f:
            json.dump(engine_versions, f, indent=4)

        # 4. label_config.json
        label_config = {
            "window_size": self.window_size,
            "window_stride": self.window_stride,
            "lookahead_bars": getattr(self.label_engine.labeler, "lookahead_bars", 20),
            "rules": [
                "trend_ema_sep_and_bos",
                "range_converged_or_retests",
                "transition_cross_or_choch_or_shrink"
            ],
            "thresholds": {
                "ema_separation_trend": getattr(self.label_engine.labeler, "ema_sep_trend", 1.5),
                "ema_separation_range": getattr(self.label_engine.labeler, "ema_sep_range", 0.8),
                "min_bos_trend": getattr(self.label_engine.labeler, "min_bos_trend", 1),
                "min_rejections_range": getattr(self.label_engine.labeler, "min_rejections_range", 2)
            },
            "invalidation_rules": [
                "unlabeled_ambiguous"
            ]
        }
        with open(os.path.join(version_dir, "label_config.json"), "w") as f:
            json.dump(label_config, f, indent=4)

        # 5. statistics.json
        dataset_size_bytes = os.path.getsize(parquet_path) + os.path.getsize(csv_path)
        sorted_symbols = sorted(symbol_dist.items(), key=lambda x: x[1], reverse=True)
        largest_symbols = [sym for sym, count in sorted_symbols[:3]]
        smallest_symbols = [sym for sym, count in sorted_symbols[-3:]]

        statistics = {
            "rows": len(df_final),
            "columns": len(df_final.columns),
            "memory_mb": float(memory_usage_mb),
            "features": len(feature_cols),
            "classes": len(validation_report.metrics.get("class_distribution", {})),
            "label_distribution": validation_report.metrics.get("class_distribution", {}),
            "missing_pct": float(nan_pct),
            "variance": feature_variances,
            "generation_time_sec": float(elapsed_time),
            "average_windows_per_sec": float(samples_per_sec),
            "dataset_size_bytes": int(dataset_size_bytes),
            "largest_symbols": largest_symbols,
            "smallest_symbols": smallest_symbols
        }
        with open(os.path.join(version_dir, "statistics.json"), "w") as f:
            json.dump(statistics, f, indent=4)

        # 6. manifest.json
        manifest = {
            "dataset_name": f"Forex_DNN_ML_Dataset_{version_str}",
            "version": version_str,
            "creation_date": datetime.now().strftime("%Y-%m-%d"),
            "git_commit": "unknown",
            "symbols": sorted(list(symbol_files.keys())),
            "timeframes": [self.timeframe],
            "window_size": int(self.window_size),
            "number_of_samples": len(df_final),
            "feature_count": len(feature_cols),
            "builder_version": "1.8",
            "engine_versions": engine_versions
        }
        with open(os.path.join(version_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=4)

        # Print the Beautiful Quality Report
        self._print_quality_report(metadata, len(df_final.columns), elapsed_time, samples_per_sec, constant_columns, symbol_dist)

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
        """
        Prints a detailed, formatted quality report as specified in the requirements.
        """
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
