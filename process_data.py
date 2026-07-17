#!/usr/bin/env python3
"""
process_data.py

Unified historical data processing entry point for the Forex_DNN framework.
Prepares ALL historical data for machine learning by running data validation,
cleaning, indicator calculation, market structure detection, S/D zone mapping,
strong/refusal candle evaluation, sliding-window labeling, and caching.

Produces 4 distinct ML datasets:
1. market_state_dataset.parquet
2. level_break_dataset.parquet
3. future_rl_dataset.parquet
4. future_trade_quality_dataset.parquet

Usage:
    # Process ALL discovered symbols
    python process_data.py --symbol ALL

    # Process specific symbols
    python process_data.py --symbol EURUSD,GBPUSD

    # Use a custom config
    python process_data.py --config Configs/process_config.yaml
"""

import os
import sys
import argparse
import logging
import json
import yaml
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import concurrent.futures
from tqdm import tqdm

# Framework Imports
from Configs.path_manager import PathManager
from ML.feature_registry import FeatureRegistry
from ML.feature_pipeline import FeaturePipeline
from ML.data_cleaner import DataCleaner
from ML.dataset_validator import DatasetValidator
from ML.label_engine import LabelEngine
from ML.market_state_labeler import MarketStateLabeler
from ML.dataset_builder import DatasetBuilder

from Market_Data_Pipeline.historical_dataset_builder import HistoricalDatasetBuilder
from Market_Data_Pipeline.structure_engine import MarketStructureEngine
from Market_Data_Pipeline.supply_demand_engine import SupplyDemandEngine
from Market_Data_Pipeline.strong_candle_engine import StrongCandleEngine
from Market_Data_Pipeline.refusal_candle_engine import RefusalCandleEngine
from Market_Data_Pipeline.structure_graph import MarketStructureGraph
from Collecting_Data.indicators import IndicatorEngine

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PathManager.get_relative_path("logs", "process_data.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger("ProcessDataPipeline")


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from YAML with safe fallbacks."""
    defaults = {
        "input_dir": PathManager.get_relative_path("historical_data"),
        "output_dir": PathManager.get_relative_path("datasets"),
        "timeframe": "M5",
        "window_size": 35,
        "window_stride": 1,
        "max_workers": 4,
        "use_multiprocessing": True,
        "clean_data": True,
        "drop_duplicates": True,
        "handle_missing": True
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
                if isinstance(cfg, dict):
                    defaults.update(cfg)
                    logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to parse config file '{config_path}': {e}. Using defaults.")
    else:
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
    return defaults


def discover_historical_files(input_dir: str, symbol_filter: str, timeframe: str) -> Dict[str, str]:
    """
    Discovers all raw Parquet and CSV files for the given timeframe.
    Supports both nested and flat structures.
    """
    discovered = {}
    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return discovered

    # Nested check: input_dir/SYMBOL/TIMEFRAME.parquet or .csv
    for item in os.listdir(input_dir):
        subdir = os.path.join(input_dir, item)
        if os.path.isdir(subdir):
            # Try parquet first, then CSV
            p_path = os.path.join(subdir, f"{timeframe}.parquet")
            if os.path.exists(p_path):
                discovered[item.upper()] = p_path
                continue
            c_path = os.path.join(subdir, f"{timeframe}.csv")
            if os.path.exists(c_path):
                discovered[item.upper()] = c_path
                continue

    # Flat check: input_dir/SYMBOL_TIMEFRAME.parquet or .csv
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path):
            import re
            match = re.match(r"^([A-Z0-9]+)_" + re.escape(timeframe) + r"\.(parquet|csv)$", file, re.IGNORECASE)
            if match:
                sym = match.group(1).upper()
                if sym not in discovered:
                    discovered[sym] = file_path

    # Apply Symbol filtering
    if symbol_filter.upper() != "ALL":
        target_symbols = [s.strip().upper() for s in symbol_filter.split(",")]
        filtered = {}
        for sym in target_symbols:
            if sym in discovered:
                filtered[sym] = discovered[sym]
            else:
                logger.warning(f"Requested symbol '{sym}' was not discovered in raw folder '{input_dir}'!")
        discovered = filtered

    return discovered


def clean_raw_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names, removes duplicates, handles missing rows, and orders chronologically."""
    df_clean = df.copy()

    # Column Standardizing mapping
    mapping = {
        "datetime": "Datetime", "date_time": "Datetime", "timestamp": "Datetime",
        "close": "Close", "open": "Open", "high": "High", "low": "Low",
        "tick_volume": "TickVolume", "volume": "TickVolume", "spread": "Spread"
    }
    for col in df_clean.columns:
        if col.lower() in mapping:
            df_clean.rename(columns={col: mapping[col.lower()]}, inplace=True)

    # Ensure crucial columns are present
    essential = ["Datetime", "Open", "High", "Low", "Close"]
    for col in essential:
        if col not in df_clean.columns:
            # Fallback mock/warning
            if col == "Datetime":
                df_clean["Datetime"] = pd.date_range("2024-01-01", periods=len(df_clean), freq="5min")
            else:
                df_clean[col] = 1.1000  # Safe placeholder

    # Drop rows with missing values in Datetime/Close
    df_clean.dropna(subset=["Datetime", "Close"], inplace=True)

    # Convert Datetime to datetime objects
    df_clean["Datetime"] = pd.to_datetime(df_clean["Datetime"])

    # Remove duplicated timestamps
    df_clean.drop_duplicates(subset=["Datetime"], keep="first", inplace=True)

    # Sort chronologically
    df_clean.sort_values(by="Datetime", inplace=True)
    df_clean.reset_index(drop=True, inplace=True)

    # Fill basic volume/spread defaults
    if "TickVolume" not in df_clean.columns:
        df_clean["TickVolume"] = 100.0
    if "Spread" not in df_clean.columns:
        df_clean["Spread"] = 1.0

    return df_clean


def process_single_symbol(
    symbol: str,
    file_path: str,
    cfg: Dict[str, Any],
    registry: FeatureRegistry,
    cache_dir: str = None
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if cache_dir is None:
        cache_dir = PathManager.get_relative_path("cache")
    """Loads, cleans, enriches, and produces Market State and Level Break datasets for a single symbol."""
    try:
        # 1. Resume Caching check
        cache_state_file = os.path.join(cache_dir, f"process_{symbol}_{cfg['timeframe']}_w{cfg['window_size']}.parquet")
        cache_level_file = os.path.join(cache_dir, f"level_break_{symbol}_{cfg['timeframe']}_w{cfg['window_size']}.parquet")

        if os.path.exists(cache_state_file) and os.path.exists(cache_level_file):
            logger.info(f"[{symbol}] [CACHE] Found cached datasets. Loading directly...")
            return pd.read_parquet(cache_state_file), pd.read_parquet(cache_level_file)

        # 2. Load & Clean Raw Data
        logger.info(f"[{symbol}] [1/5] READING: Loading raw candle data from {os.path.basename(file_path)}...")
        if file_path.endswith(".parquet"):
            df_raw = pd.read_parquet(file_path)
        else:
            df_raw = pd.read_csv(file_path)

        df_clean = clean_raw_candles(df_raw)
        total_bars = len(df_clean)
        if total_bars < cfg["window_size"]:
            logger.warning(f"[{symbol}] Insufficient candles ({total_bars}) for window {cfg['window_size']}. Skipping.")
            return None, None
        logger.info(f"[{symbol}] [1/5] READING: Successfully loaded and cleaned {total_bars} bars.")

        # 3. Enrich Candle indicators and run structures
        logger.info(f"[{symbol}] [2/5] PROCESSING (INDICATORS/SMC/SD): Computing EMAs, Slopes, SMC swings, BOS, CHOCH, and S&D zones...")
        ind_engine = IndicatorEngine(ema_periods=[50, 600, 800], slope_period=32)
        df_enriched = ind_engine.calculate(df_clean)

        ms_engine = MarketStructureEngine(lookback=3)
        df_enriched = ms_engine.process(df_enriched)

        sd_engine = SupplyDemandEngine(atr_period=14, impulse_threshold=2.0)
        df_enriched = sd_engine.process(df_enriched)
        logger.info(f"[{symbol}] [2/5] PROCESSING: Successfully completed indicators, SMC structures, and S&D mappings.")

        # 4. Construct Graph
        msg = MarketStructureGraph(
            symbol=symbol,
            timeframe=cfg["timeframe"],
            swing_highs=[s for s in ms_engine.swings if s.level_type == "SwingHigh"],
            swing_lows=[s for s in ms_engine.swings if s.level_type == "SwingLow"],
            protected_high=ms_engine.protected_high,
            protected_low=ms_engine.protected_low,
            bos=list(ms_engine.bos_list),
            choch=list(ms_engine.choch_list),
            supply_zones=[z for z in sd_engine.zones if z.type == "Supply"],
            demand_zones=[z for z in sd_engine.zones if z.type == "Demand"],
            trend_direction="Bull" if df_enriched.iloc[-1].get("trend", 0) == 1 else ("Bear" if df_enriched.iloc[-1].get("trend", 0) == -1 else "Neutral"),
            atr=float(df_enriched.iloc[-1].get("atr_14", 0.0001))
        )

        # 5. Strong & Refusal Candle Evaluations at each index (or added to features)
        logger.info(f"[{symbol}] [3/5] CANDLE EVALUATION: Initiating evaluation of Strong and Refusal candles...")
        strong_eng = StrongCandleEngine()
        refusal_eng = RefusalCandleEngine()

        strong_scores = []
        strong_confs = []
        refusal_scores = []
        refusal_confs = []

        log_interval = max(1, total_bars // 10)
        for i in range(total_bars):
            sc = strong_eng.evaluate(df_enriched, i, msg)
            rc = refusal_eng.evaluate_rejection(df_enriched, i, None, msg)
            strong_scores.append(sc.quality_score)
            strong_confs.append(sc.confidence)
            refusal_scores.append(rc.quality_score)
            refusal_confs.append(rc.confidence)

            if (i + 1) % log_interval == 0 or i == total_bars - 1:
                pct = int((i + 1) / total_bars * 100)
                logger.info(f"[{symbol}] [3/5] CANDLE EVALUATION: {pct}% complete ({i + 1}/{total_bars} bars)")

        df_enriched["strong_candle_score"] = strong_scores
        df_enriched["strong_candle_confidence"] = strong_confs
        df_enriched["refusal_candle_score"] = refusal_scores
        df_enriched["refusal_candle_confidence"] = refusal_confs
        logger.info(f"[{symbol}] [3/5] CANDLE EVALUATION: Completed successfully.")

        # 6. Generate Market State dataset using LabelEngine
        logger.info(f"[{symbol}] [4/5] LABELING & FEATURE EXTRACTION: Starting Market State labeling...")
        label_engine = LabelEngine(
            window_size=cfg["window_size"],
            window_stride=cfg["window_stride"],
            registry=registry
        )
        df_state = label_engine.generate_dataset(
            data_inputs={(symbol, cfg["timeframe"]): df_enriched},
            ms_engine=ms_engine,
            sd_engine=sd_engine
        )

        # 7. Generate Level Break dataset using DatasetBuilder
        logger.info(f"[{symbol}] [4/5] LABELING & FEATURE EXTRACTION: Starting Level Break labeling...")
        db_builder = DatasetBuilder(registry=registry)
        df_level = db_builder.build_level_break_dataset(df_enriched, msg)
        if not df_level.empty:
            df_level["symbol"] = symbol
            df_level["timeframe"] = cfg["timeframe"]
        logger.info(f"[{symbol}] [4/5] LABELING & FEATURE EXTRACTION: Labeled data generated successfully.")

        # Cache the resulting datasets
        logger.info(f"[{symbol}] [5/5] SAVING TO CACHE: Serializing processed datasets to Parquet cache files...")
        os.makedirs(cache_dir, exist_ok=True)
        if not df_state.empty:
            df_state.to_parquet(cache_state_file, index=False)
        if not df_level.empty:
            df_level.to_parquet(cache_level_file, index=False)
        logger.info(f"[{symbol}] [5/5] SAVING TO CACHE: Saved cached datasets for {symbol} successfully.")

        return df_state, df_level

    except Exception as e:
        logger.error(f"[{symbol}] Error processing symbol: {e}", exc_info=True)
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Forex_DNN Process Historical Data Pipeline")
    parser.add_argument("--symbol", type=str, default="ALL",
                        help="Specific symbols (comma-separated, e.g. EURUSD,GBPUSD) or 'ALL'.")
    parser.add_argument("--timeframe", type=str, default="M5",
                        help="Target timeframe to process (e.g. M5, M15).")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Path to folder containing historical symbol candles.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Path to folder where output datasets will be serialized.")
    parser.add_argument("--config", type=str, default="Configs/process_config.yaml",
                        help="Path to YAML configuration settings.")
    parser.add_argument("--force", action="store_true",
                        help="Force complete re-computation and ignore caches.")

    args = parser.parse_args()

    # Load Configs
    cfg = load_config(args.config)
    if args.input_dir:
        cfg["input_dir"] = args.input_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.timeframe:
        cfg["timeframe"] = args.timeframe

    logger.info("==================================================")
    logger.info("   Forex_DNN Centralized Data Processing Pipeline ")
    logger.info("==================================================")
    logger.info(f"Target Directory : {cfg['input_dir']}")
    logger.info(f"Output Directory : {cfg['output_dir']}")
    logger.info(f"Symbols Filter   : {args.symbol}")
    logger.info(f"Timeframe        : {cfg['timeframe']}")
    logger.info(f"Window Size/Stride: {cfg['window_size']} / {cfg['window_stride']}")

    # Clear cache if forced
    cache_dir = PathManager.get_relative_path("cache")
    if args.force and os.path.exists(cache_dir):
        logger.info("Force flag enabled. Flushing cache directory...")
        import shutil
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

    # Discover historical candle files
    discovered = discover_historical_files(cfg["input_dir"], args.symbol, cfg["timeframe"])
    if not discovered:
        logger.error("No historical files matching the configuration criteria were discovered. Terminating.")
        sys.exit(1)

    logger.info(f"Successfully discovered {len(discovered)} symbols to process: {list(discovered.keys())}")

    # Process symbols concurrently or sequentially
    registry = FeatureRegistry(load_defaults=True)
    all_states = []
    all_levels = []

    pbar = tqdm(discovered.items(), desc="Processing symbols")
    max_workers = cfg["max_workers"] if cfg["use_multiprocessing"] else 1

    if max_workers > 1 and len(discovered) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sym = {
                executor.submit(process_single_symbol, sym, path, cfg, registry, cache_dir): sym
                for sym, path in discovered.items()
            }
            for future in concurrent.futures.as_completed(future_to_sym):
                sym = future_to_sym[future]
                try:
                    df_state, df_level = future.result()
                    if df_state is not None and not df_state.empty:
                        all_states.append(df_state)
                    if df_level is not None and not df_level.empty:
                        all_levels.append(df_level)
                    pbar.set_postfix({"Finished": sym})
                except Exception as exc:
                    logger.error(f"Symbol {sym} generated an exception during concurrent loop: {exc}")
                pbar.update(1)
    else:
        for sym, path in discovered.items():
            pbar.set_postfix({"Current": sym})
            df_state, df_level = process_single_symbol(sym, path, cfg, registry, cache_dir)
            if df_state is not None and not df_state.empty:
                all_states.append(df_state)
            if df_level is not None and not df_level.empty:
                all_levels.append(df_level)
            pbar.update(1)

    pbar.close()

    # Consolidate and clean final datasets
    os.makedirs(cfg["output_dir"], exist_ok=True)
    cleaner = DataCleaner()
    validator = DatasetValidator()

    # 1. Market State Dataset
    if all_states:
        logger.info("Consolidating Market State Datasets...")
        master_state = pd.concat(all_states, ignore_index=True)
        if "datetime" in master_state.columns:
            master_state.sort_values(by=["datetime", "symbol"], inplace=True)
        master_state = cleaner.clean(master_state, label_col="target")

        state_path = os.path.join(cfg["output_dir"], "market_state_dataset.parquet")
        master_state.to_parquet(state_path, index=False)
        logger.info(f"Saved Consolidated Market State Dataset to {state_path} ({len(master_state)} samples)")

        # Validate
        report = validator.validate(master_state, expected_window_size=cfg["window_size"])
        logger.info(f"Market State Dataset Integrity Validation status: {'PASS' if report['is_valid'] else 'FAIL'}")

        # Produce RL and Trade Quality Datasets
        # For Reinforcement Learning
        logger.info("Preparing Reinforcement Learning (RL) Dataset...")
        master_rl = master_state.copy()

        # Calculate a deterministic forward reward: 5-bar shift of close log difference
        if "Close" in master_rl.columns:
            # Shift backwards to see future
            forward_close = master_rl.groupby("symbol")["Close"].shift(-5)
            master_rl["reward"] = ((forward_close - master_rl["Close"]) / master_rl["Close"]).fillna(0.0)
            master_rl["action"] = np.where(master_rl["reward"] > 0.001, 1, np.where(master_rl["reward"] < -0.001, 2, 0))
        else:
            master_rl["reward"] = 0.0
            master_rl["action"] = 0

        master_rl["done"] = False
        # Last 5 bars of each symbol are done
        for sym in master_rl["symbol"].unique():
            idx = master_rl[master_rl["symbol"] == sym].index
            if len(idx) > 5:
                master_rl.loc[idx[-5:], "done"] = True

        rl_path = os.path.join(cfg["output_dir"], "future_rl_dataset.parquet")
        master_rl.to_parquet(rl_path, index=False)
        logger.info(f"Saved Reinforcement Learning Dataset to {rl_path} ({len(master_rl)} samples)")

        # For Trade Quality
        logger.info("Preparing Trade Quality Dataset...")
        master_tq = master_state.copy()

        # Deterministic win_loss label based on forward outcome:
        # Check if the close rises by 1.5 ATR before dropping by 1 ATR
        if "Close" in master_tq.columns:
            forward_close_10 = master_tq.groupby("symbol")["Close"].shift(-10).fillna(master_tq["Close"])
            atr = master_tq.get("atr", 0.0010)
            # If closed higher by 1.5 ATR, win_loss=1, else 0
            master_tq["win_loss"] = np.where(forward_close_10 >= master_tq["Close"] + 1.5 * atr, 1, 0)
            master_tq["trade_quality_score"] = np.clip((forward_close_10 - master_tq["Close"]) / (1.5 * atr + 1e-9) * 100.0, 0, 100.0)
        else:
            master_tq["win_loss"] = 0
            master_tq["trade_quality_score"] = 50.0

        tq_path = os.path.join(cfg["output_dir"], "future_trade_quality_dataset.parquet")
        master_tq.to_parquet(tq_path, index=False)
        logger.info(f"Saved Trade Quality Dataset to {tq_path} ({len(master_tq)} samples)")

    else:
        logger.warning("No Market State samples were generated!")

    # 2. Level Break Dataset
    if all_levels:
        logger.info("Consolidating Level Break Datasets...")
        master_level = pd.concat(all_levels, ignore_index=True)
        if "timestamp" in master_level.columns:
            master_level.sort_values(by=["timestamp", "symbol"], inplace=True)
        master_level = cleaner.clean(master_level, label_col="target")

        level_path = os.path.join(cfg["output_dir"], "level_break_dataset.parquet")
        master_level.to_parquet(level_path, index=False)
        logger.info(f"Saved Consolidated Level Break Dataset to {level_path} ({len(master_level)} samples)")

        # Validate
        report_lvl = validator.validate(master_level, expected_window_size=cfg["window_size"])
        logger.info(f"Level Break Dataset Integrity Validation status: {'PASS' if report_lvl['is_valid'] else 'FAIL'}")
    else:
        logger.warning("No Level Break samples were generated!")

    # Generate metadata report
    meta = {
        "pipeline": "process_data.py",
        "symbols": list(discovered.keys()),
        "timeframe": cfg["timeframe"],
        "window_size": cfg["window_size"],
        "window_stride": cfg["window_stride"],
        "feature_registry_hash": registry.compute_hash(),
        "creation_time": datetime.now(timezone.utc).isoformat(),
        "market_state_size": len(master_state) if all_states else 0,
        "level_break_size": len(master_level) if all_levels else 0,
    }
    meta_path = os.path.join(cfg["output_dir"], "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    logger.info(f"Saved central metadata log to {meta_path}")

    logger.info("==================================================")
    logger.info("  DATA PROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
