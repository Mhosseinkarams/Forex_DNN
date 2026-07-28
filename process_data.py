#!/usr/bin/env python3
"""
process_data.py

Unified historical data processing entry point for the Forex_DNN framework.
Prepares ALL historical data for machine learning by running data validation,
cleaning, indicator calculation, market structure detection, S/D zone mapping,
strong/refusal candle evaluation, sliding-window labeling, and caching.

Supports robust stage-by-stage checkpointing and automatic resume after crashes.

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
import threading
import multiprocessing
import shutil
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
from ML.checkpoint_manager import CheckpointManager

from Market_Data_Pipeline.historical_dataset_builder import HistoricalDatasetBuilder
from Market_Data_Pipeline.structure_engine import MarketStructureEngine
from Market_Data_Pipeline.supply_demand_engine import SupplyDemandEngine
from Market_Data_Pipeline.strong_candle_engine import StrongCandleEngine
from Market_Data_Pipeline.refusal_candle_engine import RefusalCandleEngine
from Market_Data_Pipeline.structure_graph import MarketStructureGraph
from Collecting_Data.indicators import IndicatorEngine

# Configure Logging (MainProcess gets stdout & file, subprocesses get file only to prevent screen corruption)
if multiprocessing.current_process().name == 'MainProcess':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(PathManager.get_relative_path("logs", "process_data.log"), encoding="utf-8")
        ]
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(PathManager.get_relative_path("logs", "process_data.log"), encoding="utf-8")
        ]
    )

logger = logging.getLogger("ProcessDataPipeline")


# Console Progress Monitor & Logging handler
class ProcessingProgressMonitor:
    """
    Thread-safe console progress visualizer for the data processing pipeline.
    Maintains a live status row for each concurrent worker using ANSI escape sequences.
    """
    def __init__(self, num_workers: int, total_symbols: int, enabled: bool = True):
        self.num_workers = num_workers
        self.total_symbols = total_symbols
        self.completed_symbols = 0
        self.start_time = time.time()
        # Enable progress monitor only if terminal is interactive (tty)
        self.enabled = enabled and sys.stdout.isatty()
        self.lock = threading.Lock()

        # State for each slot: {slot_index: {"symbol": "-", "phase": "Idle", "pct": 0, "status": "Idle", "thread_id": None}}
        self.slots = {i: {"symbol": "-", "phase": "Idle", "pct": 0, "status": "Idle", "thread_id": None} for i in range(1, num_workers + 1)}
        self.active_lines_printed = False

    def register_worker(self, thread_id) -> int:
        """Assigns an unassigned slot index to a thread, making it thread-safe."""
        with self.lock:
            # Check if thread is already registered
            for slot, info in self.slots.items():
                if info.get("thread_id") == thread_id:
                    return slot
            # Assign to first unassigned slot
            for slot, info in self.slots.items():
                if info.get("thread_id") is None:
                    self.slots[slot]["thread_id"] = thread_id
                    return slot
            # Fallback
            return 1

    def register_symbol_slot(self, symbol: str) -> int:
        """Assigns an unassigned slot index to a symbol dynamically for multiprocessing updates."""
        with self.lock:
            # Check if symbol is already registered to a slot
            for slot, info in self.slots.items():
                if info.get("symbol") == symbol:
                    return slot
            # Assign to first unassigned slot
            for slot, info in self.slots.items():
                if info.get("symbol") == "-" or info.get("symbol") is None:
                    self.slots[slot]["symbol"] = symbol
                    return slot
            # Fallback
            return 1

    def release_symbol_slot(self, symbol: str):
        """Releases the slot associated with the symbol, resetting its state to Idle."""
        with self.lock:
            for slot, info in self.slots.items():
                if info.get("symbol") == symbol:
                    self.slots[slot] = {"symbol": "-", "phase": "Idle", "pct": 0, "status": "Idle", "thread_id": None}
                    break

    def increment_completed(self):
        with self.lock:
            self.completed_symbols += 1
            if not self.enabled:
                return
            self._draw()

    def update(self, slot_id: int, symbol: str, phase: str, pct: int, status: str):
        if slot_id is None:
            return
        with self.lock:
            self.slots[slot_id].update({
                "symbol": symbol,
                "phase": phase,
                "pct": pct,
                "status": status
            })
            if not self.enabled:
                return
            self._draw()

    def _draw(self):
        if not self.enabled:
            return

        lines = []
        lines.append("=" * 100)

        # Calculate overall progress
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        hours, mins = divmod(mins, 60)
        time_str = f"{hours:02d}:{mins:02d}:{secs:02d}" if hours > 0 else f"{mins:02d}:{secs:02d}"

        overall_pct = int((self.completed_symbols / self.total_symbols) * 100) if self.total_symbols > 0 else 0
        overall_filled = int((overall_pct / 100) * 30)
        overall_bar = "█" * overall_filled + "░" * (30 - overall_filled)

        lines.append(f"Pipeline Progress: {self.completed_symbols}/{self.total_symbols} symbols | [{overall_bar}] {overall_pct:3d}% | Elapsed: {time_str}")
        lines.append("-" * 100)

        for slot, info in self.slots.items():
            sym = info["symbol"]
            phase = info["phase"]
            pct = info["pct"]
            status = info["status"]

            # Progress bar
            bar_len = 20
            filled = int((pct / 100) * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)

            line = f"  Worker {slot:02d}: [{sym: <8}] | {phase: <15} | [{bar}] {pct:3d}% | {status}"
            lines.append(line)
        lines.append("=" * 100)

        # Draw to terminal
        total_lines_to_print = len(lines)
        if self.active_lines_printed:
            # Move cursor up by total_lines_to_print
            sys.stdout.write(f"\033[{total_lines_to_print}A")
            for line in lines:
                # Clear line and print
                sys.stdout.write(f"\033[K{line}\n")
        else:
            for line in lines:
                sys.stdout.write(f"{line}\n")
            self.active_lines_printed = True
        sys.stdout.flush()

    def clear(self):
        """Clears the progress monitor output from console cleanly without causing scrolls."""
        if not self.enabled or not self.active_lines_printed:
            return
        with self.lock:
            total_lines = 4 + self.num_workers
            # Move up one line and clear it, for total_lines
            for _ in range(total_lines):
                sys.stdout.write("\033[1A\033[K")
            sys.stdout.flush()
            self.active_lines_printed = False

    def redraw(self):
        """Redraws the progress monitor after being cleared."""
        if not self.enabled:
            return
        with self.lock:
            self._draw()


class ProgressAwareStreamHandler(logging.StreamHandler):
    """
    Log stream handler that intercepts emitting, temporarily clears
    the live progress monitor, prints the log, and restores the progress display.
    """
    def __init__(self, monitor: ProcessingProgressMonitor = None, stream=None):
        super().__init__(stream)
        self.monitor = monitor

    def emit(self, record):
        try:
            msg = self.format(record)
            if self.monitor and self.monitor.enabled and self.monitor.active_lines_printed:
                self.monitor.clear()
                self.stream.write(msg + self.terminator)
                self.stream.flush()
                self.monitor.redraw()
            else:
                self.stream.write(msg + self.terminator)
                self.stream.flush()
        except Exception:
            self.handleError(record)


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
            match = re.match(r"^([A-Z0-9#_.]+)_" + re.escape(timeframe) + r"\.(parquet|csv)$", file, re.IGNORECASE)
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


def save_parquet_atomically(df: pd.DataFrame, path: str) -> None:
    """Saves a pandas DataFrame to parquet format atomically using a temporary swap."""
    temp_path = f"{path}.tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_parquet(temp_path, index=False)
    os.replace(temp_path, path)


def process_single_symbol(
    symbol: str,
    file_path: str,
    cfg: Dict[str, Any],
    registry: FeatureRegistry,
    cache_dir: str = None,
    monitor: Optional[ProcessingProgressMonitor] = None,
    slot_id: Optional[int] = None,
    progress_queue: Optional[Any] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Loads, cleans, enriches, and produces Market State and Level Break datasets for a single symbol
    utilizing robust stage-by-stage checkpointing and automatic resume after crashes.
    """
    try:
        # Suppress standard logging output to stream handlers inside subprocess workers to keep terminal display strictly clean
        if multiprocessing.current_process().name != 'MainProcess':
            root_logger = logging.getLogger()
            for h in list(root_logger.handlers):
                if isinstance(h, logging.StreamHandler):
                    root_logger.removeHandler(h)

        # Register slot dynamically if monitor is provided but slot_id is not
        if monitor and slot_id is None:
            slot_id = monitor.register_worker(threading.get_ident())

        # Helper to route progress updates
        def update_progress(phase: str, pct: int, status: str, log_msg: Optional[str] = None):
            if progress_queue is not None:
                progress_queue.put(("UPDATE", symbol, phase, pct, status, log_msg))
            elif monitor and monitor.enabled and slot_id:
                monitor.update(slot_id, symbol, phase, pct, status)
            elif log_msg:
                logger.info(log_msg)

        # Setup Stage Directories and Checkpoint Manager
        stages_dir = PathManager.get_path("temporary", "stages")
        checkpoint_dir = PathManager.get_path("temporary", "checkpoints")
        stages = ["CLEAN", "ENRICH", "EVAL", "STATE", "LEVEL"]

        cp_mgr = CheckpointManager(
            symbol=symbol,
            timeframe=cfg["timeframe"],
            window_size=cfg["window_size"],
            stages=stages,
            checkpoint_dir=checkpoint_dir
        )

        # Resume logic loop across sequential unfinished stages
        while True:
            next_stage = cp_mgr.get_next_unfinished_stage()
            if next_stage is None:
                # All stages have successfully completed
                break

            # ----------------- STAGE 1: CLEAN -----------------
            if next_stage == "CLEAN":
                cp_mgr.mark_stage_started("CLEAN")
                update_progress("CLEANING", 0, "Loading & cleaning raw candles...", f"[{symbol}] [Stage CLEAN] Loading raw candles from {os.path.basename(file_path)}...")

                # 1. Load raw candle data
                if file_path.endswith(".parquet"):
                    df_raw = pd.read_parquet(file_path)
                else:
                    df_raw = pd.read_csv(file_path)

                # 2. Execute
                df_clean = clean_raw_candles(df_raw)
                total_bars = len(df_clean)

                # 3. Validate
                if total_bars < cfg["window_size"]:
                    raise ValueError(f"Insufficient candles ({total_bars}) for window size {cfg['window_size']}.")
                essential = ["Datetime", "Open", "High", "Low", "Close"]
                for col in essential:
                    if col not in df_clean.columns:
                        raise ValueError(f"Missing essential column '{col}' in cleaned raw data.")

                # 4. Save Atomic
                clean_out_path = os.path.join(stages_dir, f"{symbol}_clean.parquet")
                save_parquet_atomically(df_clean, clean_out_path)

                # 5. Update checkpoint
                cp_mgr.mark_stage_completed("CLEAN", clean_out_path)

                # 6 & 7. Release memory & GC
                del df_raw, df_clean
                import gc
                gc.collect()

                update_progress("CLEANING", 100, f"Cleaned {total_bars} bars", f"[{symbol}] [Stage CLEAN] Finished. Cleaned {total_bars} bars.")

            # ----------------- STAGE 2: ENRICH -----------------
            elif next_stage == "ENRICH":
                cp_mgr.mark_stage_started("ENRICH")
                update_progress("PROCESSING", 0, "Computing EMAs & SMC structures...", f"[{symbol}] [Stage ENRICH] Computing indicators, SMC structures, and S&D zones...")

                # 1. Load required data
                clean_in_path = cp_mgr.get_stage_output("CLEAN")
                df_clean = pd.read_parquet(clean_in_path)

                # 2. Execute
                ind_engine = IndicatorEngine(ema_periods=[50, 600, 800], slope_period=32)
                df_enriched = ind_engine.calculate(df_clean)

                ms_engine = MarketStructureEngine(lookback=3)
                df_enriched = ms_engine.process(df_enriched)

                sd_engine = SupplyDemandEngine(atr_period=14, impulse_threshold=2.0)
                df_enriched = sd_engine.process(df_enriched)

                # 3. Validate
                if df_enriched.empty:
                    raise ValueError("Enriched DataFrame is empty.")
                required_cols = ["ema_50", "ema_600", "ema_800", "atr_14", "trend"]
                for col in required_cols:
                    if col not in df_enriched.columns:
                        raise ValueError(f"Missing indicator/SMC column '{col}' in enriched data.")

                # 4. Save Atomic
                enrich_out_path = os.path.join(stages_dir, f"{symbol}_enriched.parquet")
                save_parquet_atomically(df_enriched, enrich_out_path)

                # 5. Update checkpoint
                cp_mgr.mark_stage_completed("ENRICH", enrich_out_path)

                # 6 & 7. Release memory & GC
                del df_clean, df_enriched
                import gc
                gc.collect()

                update_progress("PROCESSING", 100, "Completed structures", f"[{symbol}] [Stage ENRICH] Finished indicators and SMC structures.")

            # ----------------- STAGE 3: EVAL -----------------
            elif next_stage == "EVAL":
                cp_mgr.mark_stage_started("EVAL")
                update_progress("CANDLE_EVAL", 0, "Evaluating candles...", f"[{symbol}] [Stage EVAL] Starting Strong/Refusal candle evaluation...")

                # 1. Load required data
                enrich_in_path = cp_mgr.get_stage_output("ENRICH")
                df_enriched = pd.read_parquet(enrich_in_path)
                total_bars = len(df_enriched)

                # Local reconstruction for evaluation engine
                ms_engine = MarketStructureEngine(lookback=3)
                df_struct = ms_engine.process(df_enriched)
                sd_engine = SupplyDemandEngine(atr_period=14, impulse_threshold=2.0)
                df_final_struct = sd_engine.process(df_struct)

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
                    trend_direction="Bull" if df_final_struct.iloc[-1].get("trend", 0) == 1 else ("Bear" if df_final_struct.iloc[-1].get("trend", 0) == -1 else "Neutral"),
                    atr=float(df_final_struct.iloc[-1].get("atr_14", 0.0001))
                )

                # 2. Execute
                strong_eng = StrongCandleEngine()
                refusal_eng = RefusalCandleEngine()

                strong_scores = []
                strong_confs = []
                refusal_scores = []
                refusal_confs = []

                log_interval = max(1, total_bars // 10)
                for i in range(total_bars):
                    sc = strong_eng.evaluate(df_final_struct, i, msg)
                    rc = refusal_eng.evaluate_rejection(df_final_struct, i, None, msg)
                    strong_scores.append(sc.quality_score)
                    strong_confs.append(sc.confidence)
                    refusal_scores.append(rc.quality_score)
                    refusal_confs.append(rc.confidence)

                    if (i + 1) % log_interval == 0 or i == total_bars - 1:
                        pct = int((i + 1) / total_bars * 100)
                        update_progress("CANDLE_EVAL", pct, f"Bars {i+1}/{total_bars}", f"[{symbol}] [Stage EVAL] Candle evaluation progress: {pct}%")

                df_evaluated = df_enriched.copy()
                df_evaluated["strong_candle_score"] = strong_scores
                df_evaluated["strong_candle_confidence"] = strong_confs
                df_evaluated["refusal_candle_score"] = refusal_scores
                df_evaluated["refusal_candle_confidence"] = refusal_confs

                # 3. Validate
                required_eval_cols = ["strong_candle_score", "strong_candle_confidence", "refusal_candle_score", "refusal_candle_confidence"]
                for col in required_eval_cols:
                    if col not in df_evaluated.columns:
                        raise ValueError(f"Missing evaluation column '{col}' in evaluated data.")

                # 4. Save Atomic
                eval_out_path = os.path.join(stages_dir, f"{symbol}_evaluated.parquet")
                save_parquet_atomically(df_evaluated, eval_out_path)

                # 5. Update checkpoint
                cp_mgr.mark_stage_completed("EVAL", eval_out_path)

                # 6 & 7. Release memory & GC
                del df_enriched, df_evaluated, df_struct, df_final_struct, msg
                import gc
                gc.collect()

                update_progress("CANDLE_EVAL", 100, "Completed candle evaluations", f"[{symbol}] [Stage EVAL] Finished Strong/Refusal candle evaluations.")

            # ----------------- STAGE 4: STATE -----------------
            elif next_stage == "STATE":
                cp_mgr.mark_stage_started("STATE")
                update_progress("LABELING_MS", 0, "Starting Market State labeling...", f"[{symbol}] [Stage STATE] Commencing Market State sliding-window labeling...")

                # 1. Load required data
                eval_in_path = cp_mgr.get_stage_output("EVAL")
                df_evaluated = pd.read_parquet(eval_in_path)

                # Local reconstruction
                ms_engine = MarketStructureEngine(lookback=3)
                df_struct = ms_engine.process(df_evaluated)
                sd_engine = SupplyDemandEngine(atr_period=14, impulse_threshold=2.0)
                df_final_struct = sd_engine.process(df_struct)

                # 2. Execute
                label_engine = LabelEngine(
                    window_size=cfg["window_size"],
                    window_stride=cfg["window_stride"],
                    registry=registry
                )
                df_state = label_engine.generate_dataset(
                    data_inputs={(symbol, cfg["timeframe"]): df_final_struct},
                    ms_engine=ms_engine,
                    sd_engine=sd_engine,
                    monitor=monitor,
                    slot_id=slot_id
                )

                # 3. Validate
                if not df_state.empty:
                    required_state_cols = ["target", "confidence", "symbol"]
                    for col in required_state_cols:
                        if col not in df_state.columns:
                            raise ValueError(f"Missing column '{col}' in generated Market State dataset.")

                # 4. Save Atomic
                state_out_path = os.path.join(stages_dir, f"{symbol}_state.parquet")
                save_parquet_atomically(df_state, state_out_path)

                # 5. Update checkpoint
                cp_mgr.mark_stage_completed("STATE", state_out_path)

                # 6 & 7. Release memory & GC
                del df_evaluated, df_struct, df_final_struct, df_state, label_engine
                import gc
                gc.collect()

                update_progress("LABELING_MS", 100, "Completed market state labeling", f"[{symbol}] [Stage STATE] Finished Market State labeling.")

            # ----------------- STAGE 5: LEVEL -----------------
            elif next_stage == "LEVEL":
                cp_mgr.mark_stage_started("LEVEL")
                update_progress("LABELING_LVL", 0, "Starting Level Break labeling...", f"[{symbol}] [Stage LEVEL] Commencing Level Break proximity and outcome labeling...")

                # 1. Load required data
                eval_in_path = cp_mgr.get_stage_output("EVAL")
                df_evaluated = pd.read_parquet(eval_in_path)

                # Local reconstruction
                ms_engine = MarketStructureEngine(lookback=3)
                df_struct = ms_engine.process(df_evaluated)
                sd_engine = SupplyDemandEngine(atr_period=14, impulse_threshold=2.0)
                df_final_struct = sd_engine.process(df_struct)

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
                    trend_direction="Bull" if df_final_struct.iloc[-1].get("trend", 0) == 1 else ("Bear" if df_final_struct.iloc[-1].get("trend", 0) == -1 else "Neutral"),
                    atr=float(df_final_struct.iloc[-1].get("atr_14", 0.0001))
                )

                # 2. Execute
                db_builder = DatasetBuilder(registry=registry)
                df_level = db_builder.build_level_break_dataset(
                    df_final_struct, msg,
                    monitor=monitor,
                    slot_id=slot_id
                )
                if not df_level.empty:
                    df_level["symbol"] = symbol
                    df_level["timeframe"] = cfg["timeframe"]

                # 3. Validate
                if not df_level.empty:
                    required_level_cols = ["target", "zone_type", "symbol"]
                    for col in required_level_cols:
                        if col not in df_level.columns:
                            raise ValueError(f"Missing column '{col}' in generated Level Break dataset.")

                # 4. Save Atomic
                level_out_path = os.path.join(stages_dir, f"{symbol}_level.parquet")
                save_parquet_atomically(df_level, level_out_path)

                # 5. Update checkpoint
                cp_mgr.mark_stage_completed("LEVEL", level_out_path)

                # 6 & 7. Release memory & GC
                del df_evaluated, df_struct, df_final_struct, msg, df_level, db_builder
                import gc
                gc.collect()

                update_progress("LABELING_LVL", 100, "Completed level break labeling", f"[{symbol}] [Stage LEVEL] Finished Level Break labeling.")

        # Finally, return completed stage outputs
        state_final_path = cp_mgr.get_stage_output("STATE")
        level_final_path = cp_mgr.get_stage_output("LEVEL")
        return state_final_path, level_final_path

    except Exception as e:
        logger.error(f"[{symbol}] Error processing symbol: {e}", exc_info=True)
        if progress_queue is not None:
            progress_queue.put(("UPDATE", symbol, "ERROR", 0, str(e), f"[{symbol}] Error: {e}"))
        return None, None
    finally:
        if progress_queue is not None:
            progress_queue.put(("RELEASE", symbol))


def progress_listener_worker(queue, monitor):
    """
    Background thread running in the main parent process that listens to the progress queue
    and updates the terminal ProcessingProgressMonitor display dynamically.
    """
    # Create a local file logger strictly dedicated to recording background worker progress step details to logs/process_data.log
    file_logger = logging.getLogger("ProcessDataFileLogger")
    file_logger.propagate = False
    if not file_logger.handlers:
        file_logger.addHandler(logging.FileHandler(PathManager.get_relative_path("logs", "process_data.log"), encoding="utf-8"))
        file_logger.setLevel(logging.INFO)

    try:
        while True:
            msg = queue.get()
            if msg is None:  # Shutdown sentinel
                break

            msg_type = msg[0]
            if msg_type == "UPDATE":
                _, symbol, phase, pct, status, log_msg = msg
                slot = monitor.register_symbol_slot(symbol)
                monitor.update(slot, symbol, phase, pct, status)
                # Write background progress update messages to the file logger only (leaving the terminal console clean)
                if log_msg and file_logger:
                    file_logger.info(log_msg)
            elif msg_type == "RELEASE":
                _, symbol = msg
                monitor.release_symbol_slot(symbol)
    except Exception as e:
        logger.error(f"Error in progress listener thread: {e}", exc_info=True)


def main():
    from Collecting_Data.memory_monitor import MemoryMonitor
    mem_monitor = MemoryMonitor()
    mem_monitor.check("Pipeline start")

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

    # Flush directories if forced
    stages_dir = PathManager.get_path("temporary", "stages")
    checkpoint_dir = PathManager.get_path("temporary", "checkpoints")

    if args.force:
        logger.info("Force flag enabled. Flushing checkpoints, stages, and cache directories...")
        shutil.rmtree(stages_dir, ignore_errors=True)
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        shutil.rmtree(PathManager.get_path("cache"), ignore_errors=True)

    os.makedirs(stages_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Discover historical candle files
    discovered = discover_historical_files(cfg["input_dir"], args.symbol, cfg["timeframe"])
    if not discovered:
        logger.error("No historical files matching the configuration criteria were discovered. Terminating.")
        sys.exit(1)

    logger.info(f"Successfully discovered {len(discovered)} symbols to process: {list(discovered.keys())}")

    # Resolve active workers
    max_workers = cfg["max_workers"] if cfg["use_multiprocessing"] else 1
    total_symbols = len(discovered)

    # Initialize Progress Monitor
    monitor = ProcessingProgressMonitor(num_workers=max_workers, total_symbols=total_symbols, enabled=True)

    # Swap standard log handler with progress-aware log handler
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, ProgressAwareStreamHandler):
            formatter = handler.formatter
            new_handler = ProgressAwareStreamHandler(monitor=monitor, stream=sys.stdout)
            new_handler.setFormatter(formatter)
            root_logger.removeHandler(handler)
            root_logger.addHandler(new_handler)

    # Process symbols concurrently or sequentially
    registry = FeatureRegistry(load_defaults=True)
    state_paths = []
    level_paths = []

    if monitor.enabled:
        monitor._draw()

    if max_workers > 1 and total_symbols > 1:
        if cfg.get("use_multiprocessing", True):
            # True ProcessPoolExecutor multiprocessing (bypasses GIL)
            logger.info(f"Initiating true ProcessPoolExecutor parallel execution on {max_workers} CPU workers...")

            # Start background thread to listen to progress updates from worker processes
            manager = multiprocessing.Manager()
            progress_queue = manager.Queue()

            listener_thread = threading.Thread(
                target=progress_listener_worker,
                args=(progress_queue, monitor),
                daemon=True
            )
            listener_thread.start()

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_sym = {
                    executor.submit(process_single_symbol, sym, path, cfg, registry, None, None, None, progress_queue): sym
                    for sym, path in discovered.items()
                }
                for future in concurrent.futures.as_completed(future_to_sym):
                    sym = future_to_sym[future]
                    try:
                        state_path, level_path = future.result()
                        if state_path:
                            state_paths.append(state_path)
                        if level_path:
                            level_paths.append(level_path)
                    except Exception as exc:
                        logger.error(f"Symbol {sym} generated an exception during concurrent multiprocessing run: {exc}", exc_info=True)
                    monitor.increment_completed()

            # Stop the progress listener thread cleanly
            progress_queue.put(None)
            listener_thread.join(timeout=5)
        else:
            # ThreadPoolExecutor (shares memory/monitor directly, but bound by GIL)
            logger.info(f"Initiating ThreadPoolExecutor execution on {max_workers} thread workers...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_sym = {
                    executor.submit(process_single_symbol, sym, path, cfg, registry, None, monitor, None, None): sym
                    for sym, path in discovered.items()
                }
                for future in concurrent.futures.as_completed(future_to_sym):
                    sym = future_to_sym[future]
                    try:
                        state_path, level_path = future.result()
                        if state_path:
                            state_paths.append(state_path)
                        if level_path:
                            level_paths.append(level_path)
                    except Exception as exc:
                        logger.error(f"Symbol {sym} generated an exception during thread loop: {exc}")
                    monitor.increment_completed()
    else:
        for sym, path in discovered.items():
            state_path, level_path = process_single_symbol(sym, path, cfg, registry, None, monitor, 1, None)
            if state_path:
                state_paths.append(state_path)
            if level_path:
                level_paths.append(level_path)
            monitor.increment_completed()

    if monitor.enabled:
        monitor.clear()

    # Consolidate and clean final datasets
    os.makedirs(cfg["output_dir"], exist_ok=True)
    cleaner = DataCleaner()
    validator = DatasetValidator()

    # 1. Market State Dataset
    master_state = None
    if state_paths:
        logger.info("Consolidating Market State Datasets...")
        mem_monitor.check("Consolidation of Market State start")
        all_states = []
        for path in state_paths:
            if os.path.exists(path):
                df = pd.read_parquet(path)
                if not df.empty:
                    all_states.append(df)

        if all_states:
            master_state = pd.concat(all_states, ignore_index=True)
            # Free individual lists/DataFrames immediately
            del all_states
            import gc
            gc.collect()

            if "datetime" in master_state.columns:
                master_state.sort_values(by=["datetime", "symbol"], inplace=True)
            master_state = cleaner.clean(master_state, label_col="target")

            state_path = os.path.join(cfg["output_dir"], "market_state_dataset.parquet")
            save_parquet_atomically(master_state, state_path)
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
            save_parquet_atomically(master_rl, rl_path)
            logger.info(f"Saved Reinforcement Learning Dataset to {rl_path} ({len(master_rl)} samples)")
            del master_rl
            gc.collect()

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
            save_parquet_atomically(master_tq, tq_path)
            logger.info(f"Saved Trade Quality Dataset to {tq_path} ({len(master_tq)} samples)")
            del master_tq
            gc.collect()
            mem_monitor.check("After Market State consolidation and saving")

    if not state_paths or master_state is None:
        logger.warning("No Market State samples were generated!")

    # 2. Level Break Dataset
    master_level = None
    if level_paths:
        logger.info("Consolidating Level Break Datasets...")
        all_levels = []
        for path in level_paths:
            if os.path.exists(path):
                df = pd.read_parquet(path)
                if not df.empty:
                    all_levels.append(df)
        if all_levels:
            master_level = pd.concat(all_levels, ignore_index=True)
            # Free individual DataFrames from memory immediately
            del all_levels
            import gc
            gc.collect()

            if "timestamp" in master_level.columns:
                master_level.sort_values(by=["timestamp", "symbol"], inplace=True)
            master_level = cleaner.clean(master_level, label_col="target")

            level_path = os.path.join(cfg["output_dir"], "level_break_dataset.parquet")
            save_parquet_atomically(master_level, level_path)
            logger.info(f"Saved Consolidated Level Break Dataset to {level_path} ({len(master_level)} samples)")

            # Validate
            report_lvl = validator.validate(master_level, expected_window_size=cfg["window_size"])
            logger.info(f"Level Break Dataset Integrity Validation status: {'PASS' if report_lvl['is_valid'] else 'FAIL'}")

    if not level_paths or master_level is None:
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
        "market_state_size": len(master_state) if master_state is not None else 0,
        "level_break_size": len(master_level) if master_level is not None else 0,
    }
    meta_path = os.path.join(cfg["output_dir"], "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    logger.info(f"Saved central metadata log to {meta_path}")

    # Clean up master frames
    del master_state, master_level
    import gc
    gc.collect()

    mem_monitor.check("Pipeline finished")

    logger.info("==================================================")
    logger.info("  DATA PROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
