import os
import re
import sys
import time
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
import numpy as np
import concurrent.futures

from ML.feature_registry import FeatureRegistry
from ML.label_engine import LabelEngine
from ML.dataset_validator import DatasetValidator
from Market_Data_Pipeline.structure_engine import MarketStructureEngine
from Market_Data_Pipeline.supply_demand_engine import SupplyDemandEngine

logger = logging.getLogger("HistoricalDatasetBuilder")

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

        self.registry = registry or FeatureRegistry(load_defaults=True)
        self.ms_engine = ms_engine or MarketStructureEngine(lookback=3)
        self.sd_engine = sd_engine or SupplyDemandEngine(atr_period=14, impulse_threshold=2.0)

        self.label_engine = label_engine or LabelEngine(
            window_size=self.window_size,
            window_stride=self.window_stride,
            registry=self.registry
        )

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
        """
        print(f"Loading {symbol}...")
        df_ohlcv = self._load_file(file_path)

        print(f"Generating windows for {symbol}...")
        # Since we might be parallel processing, construct a fresh thread-safe LabelEngine instance
        local_label_engine = LabelEngine(
            window_size=self.window_size,
            window_stride=self.window_stride,
            registry=self.registry
        )

        print(f"Running Structure Engine for {symbol}...")
        print(f"Running S&D for {symbol}...")
        print(f"Running Labels for {symbol}...")
        df_labeled = local_label_engine.generate_dataset(
            data_inputs=df_ohlcv,
            symbol=symbol,
            timeframe=self.timeframe,
            ms_engine=self.ms_engine,
            sd_engine=self.sd_engine
        )

        return df_labeled

    def resolve_next_version(self) -> str:
        """
        Finds the next available version number automatically from output_dir.
        If output_dir contains dataset_v001.parquet, returns v002.
        If no datasets found, returns v001.
        """
        if self.version:
            return self.version

        if not os.path.exists(self.output_dir):
            return "v001"

        max_v = 0
        for file in os.listdir(self.output_dir):
            match = re.match(r"^dataset_v(\d+)\.(parquet|csv)$", file)
            if match:
                v_num = int(match.group(1))
                if v_num > max_v:
                    max_v = v_num

        next_v = max_v + 1
        return f"v{next_v:03d}"

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

        if max_workers is None or max_workers > 1:
            # Parallel processing
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
                            logger.info(f"Successfully processed {sym} ({len(df_sym)} labeled rows).")
                        else:
                            logger.warning(f"No labeled samples generated for {sym}.")
                    except Exception as exc:
                        logger.error(f"Symbol {sym} generated an exception: {exc}", exc_info=True)
        else:
            # Sequential processing
            logger.info("Processing symbols sequentially...")
            for sym, path in symbol_files.items():
                try:
                    df_sym = self.process_symbol(sym, path)
                    if not df_sym.empty:
                        all_dfs.append(df_sym)
                        logger.info(f"Successfully processed {sym} ({len(df_sym)} labeled rows).")
                except Exception as exc:
                    logger.error(f"Symbol {sym} generated an exception: {exc}", exc_info=True)

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

        # Check feature variance and constant columns
        numeric_cols = df_final.select_dtypes(include=[np.number]).columns
        # Exclude metadata / targets
        feature_cols = [c for c in numeric_cols if c not in ["target", "confidence", "window_start", "window_end", "Open", "High", "Low", "Close", "TickVolume", "ema_50", "ema_600", "ema_800"]]

        feature_variances = {}
        constant_columns = []
        for col in feature_cols:
            var = float(df_final[col].var())
            feature_variances[col] = var
            if var == 0 or np.isnan(var) or df_final[col].nunique() <= 1:
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

        # Save files
        print("Saving Dataset...")
        os.makedirs(self.output_dir, exist_ok=True)
        parquet_path = os.path.join(self.output_dir, f"dataset_{version_str}.parquet")
        csv_path = os.path.join(self.output_dir, f"dataset_{version_str}.csv")
        metadata_path = os.path.join(self.output_dir, f"dataset_{version_str}_metadata.json")

        df_final.to_parquet(parquet_path, index=False)
        df_final.to_csv(csv_path, index=False)

        # Construct metadata JSON
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
            "feature_count": len(self.registry.list_enabled()),
            "label_distribution": label_dist,
            "symbol_distribution": symbol_dist,
            "created": datetime.now().strftime("%Y-%m-%d"),
            "generation_date": datetime.now().isoformat(),
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

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

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
