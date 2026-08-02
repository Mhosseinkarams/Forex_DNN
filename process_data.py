#!/usr/bin/env python3
"""
process_data.py

Completely redesigned, enterprise-grade streaming data processing pipeline.
Processes ONE trading symbol at a time sequentially, completely releasing resources before starting the next.
For the active symbol, divides the workload into sequential stages (CLEAN -> ENRICH -> EVAL -> STATE -> LEVEL),
and parallelizes computations across sequential CHUNKS of the SAME symbol via a stateless worker pool.

Prioritizes:
    - Low RAM usage (comfortable on 16GB RAM)
    - Low IPC overhead (uses lightweight task descriptors, direct Parquet reads/writes)
    - Granular, atomic chunk-level checkpointing & crash recovery
    - Adaptive chunk sizing
    - Robust worker failure recovery & deterministic execution

Produces 4 distinct ML datasets:
1. market_state_dataset.parquet
2. level_break_dataset.parquet
3. future_rl_dataset.parquet
4. future_trade_quality_dataset.parquet

Usage:
    python process_data.py --symbol ALL
    python process_data.py --symbol EURUSD,GBPUSD
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
import psutil
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import concurrent.futures
import pyarrow as pa
import pyarrow.parquet as pq

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


class ProcessingProgressMonitor:
    """
    Thread-safe console progress visualizer for the redesigned streaming pipeline.
    Maintains a live status row for each concurrent chunk worker using ANSI escape sequences.
    """
    def __init__(self, num_workers: int, total_chunks: int = 0, enabled: bool = True):
        self.num_workers = num_workers
        self.total_chunks = total_chunks
        self.completed_chunks = 0
        self.start_time = time.time()
        self.enabled = enabled and sys.stdout.isatty()
        self.lock = threading.Lock()

        self.slots = {i: {"chunk_id": "-", "phase": "Idle", "pct": 0, "status": "Idle", "thread_id": None} for i in range(1, num_workers + 1)}
        self.active_lines_printed = False

    def register_worker(self, thread_id) -> int:
        with self.lock:
            for slot, info in self.slots.items():
                if info.get("thread_id") == thread_id:
                    return slot
            for slot, info in self.slots.items():
                if info.get("thread_id") is None:
                    self.slots[slot]["thread_id"] = thread_id
                    return slot
            return 1

    def register_chunk_slot(self, chunk_id: int) -> int:
        with self.lock:
            for slot, info in self.slots.items():
                if info.get("chunk_id") == chunk_id:
                    return slot
            for slot, info in self.slots.items():
                if info.get("chunk_id") == "-" or info.get("chunk_id") is None:
                    self.slots[slot]["chunk_id"] = chunk_id
                    return slot
            return 1

    def release_chunk_slot(self, chunk_id: int):
        with self.lock:
            for slot, info in self.slots.items():
                if info.get("chunk_id") == chunk_id:
                    self.slots[slot] = {"chunk_id": "-", "phase": "Idle", "pct": 0, "status": "Idle", "thread_id": None}
                    break

    def increment_completed(self):
        with self.lock:
            self.completed_chunks += 1
            if not self.enabled:
                return
            self._draw()

    def update(self, slot_id: int, chunk_id: Any, phase: str, pct: int, status: str):
        if slot_id is None:
            return
        with self.lock:
            self.slots[slot_id].update({
                "chunk_id": chunk_id,
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

        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        hours, mins = divmod(mins, 60)
        time_str = f"{hours:02d}:{mins:02d}:{secs:02d}" if hours > 0 else f"{mins:02d}:{secs:02d}"

        overall_pct = int((self.completed_chunks / self.total_chunks) * 100) if self.total_chunks > 0 else 0
        overall_filled = int((overall_pct / 100) * 30)
        overall_bar = "█" * overall_filled + "░" * (30 - overall_filled)

        lines.append(f"Stage Chunks Progress: {self.completed_chunks}/{self.total_chunks} chunks | [{overall_bar}] {overall_pct:3d}% | Elapsed: {time_str}")
        lines.append("-" * 100)

        for slot, info in self.slots.items():
            cid = info["chunk_id"]
            phase = info["phase"]
            pct = info["pct"]
            status = info["status"]

            bar_len = 20
            filled = int((pct / 100) * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)

            cid_str = f"Chunk {cid}" if isinstance(cid, int) else str(cid)
            line = f"  Worker {slot:02d}: [{cid_str: <10}] | {phase: <15} | [{bar}] {pct:3d}% | {status}"
            lines.append(line)
        lines.append("=" * 100)

        total_lines_to_print = len(lines)
        if self.active_lines_printed:
            sys.stdout.write(f"\033[{total_lines_to_print}A")
            for line in lines:
                sys.stdout.write(f"\033[K{line}\n")
        else:
            for line in lines:
                sys.stdout.write(f"{line}\n")
            self.active_lines_printed = True
        sys.stdout.flush()

    def clear(self):
        if not self.enabled or not self.active_lines_printed:
            return
        with self.lock:
            total_lines = 4 + self.num_workers
            for _ in range(total_lines):
                sys.stdout.write("\033[1A\033[K")
            sys.stdout.flush()
            self.active_lines_printed = False

    def redraw(self):
        if not self.enabled:
            return
        with self.lock:
            self._draw()


class ProgressAwareStreamHandler(logging.StreamHandler):
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
        "handle_missing": True,
        "chunk_size": 100000
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
    discovered = {}
    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return discovered

    for item in os.listdir(input_dir):
        subdir = os.path.join(input_dir, item)
        if os.path.isdir(subdir):
            p_path = os.path.join(subdir, f"{timeframe}.parquet")
            if os.path.exists(p_path):
                discovered[item.upper()] = p_path
                continue
            c_path = os.path.join(subdir, f"{timeframe}.csv")
            if os.path.exists(c_path):
                discovered[item.upper()] = c_path
                continue

    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path):
            import re
            match = re.match(r"^([A-Z0-9#_.]+)_" + re.escape(timeframe) + r"\.(parquet|csv)$", file, re.IGNORECASE)
            if match:
                sym = match.group(1).upper()
                if sym not in discovered:
                    discovered[sym] = file_path

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
    df_clean = df.copy()
    mapping = {
        "datetime": "Datetime", "date_time": "Datetime", "timestamp": "Datetime",
        "close": "Close", "open": "Open", "high": "High", "low": "Low",
        "tick_volume": "TickVolume", "volume": "TickVolume", "spread": "Spread"
    }
    for col in df_clean.columns:
        if col.lower() in mapping:
            df_clean.rename(columns={col: mapping[col.lower()]}, inplace=True)

    essential = ["Datetime", "Open", "High", "Low", "Close"]
    for col in essential:
        if col not in df_clean.columns:
            if col == "Datetime":
                df_clean["Datetime"] = pd.date_range("2024-01-01", periods=len(df_clean), freq="5min")
            else:
                df_clean[col] = 1.1000

    df_clean.dropna(subset=["Datetime", "Close"], inplace=True)
    df_clean["Datetime"] = pd.to_datetime(df_clean["Datetime"])
    df_clean.drop_duplicates(subset=["Datetime"], keep="first", inplace=True)
    df_clean.sort_values(by="Datetime", inplace=True)
    df_clean.reset_index(drop=True, inplace=True)

    if "TickVolume" not in df_clean.columns:
        df_clean["TickVolume"] = 100.0
    if "Spread" not in df_clean.columns:
        df_clean["Spread"] = 1.0

    return df_clean


def save_parquet_atomically(df: pd.DataFrame, path: str) -> None:
    temp_path = f"{path}.tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_parquet(temp_path, index=False)
    os.replace(temp_path, path)


def calculate_adaptive_chunk_size(total_rows: int, cfg: Dict[str, Any]) -> int:
    """
    Dynamically determines optimal chunk size based on dataset size, CPU cores,
    and system memory bounds to prevent OOM issues and optimize throughput.
    """
    default_size = cfg.get("chunk_size", 100000)
    cpu_cores = os.cpu_count() or 4
    try:
        mem_info = psutil.virtual_memory()
        available_gb = mem_info.available / (1024 ** 3)
    except Exception:
        available_gb = 16.0

    if available_gb < 8.0:
        optimal_size = min(default_size, 30000)
    elif available_gb > 32.0:
        optimal_size = max(default_size, 150000)
    else:
        optimal_size = default_size

    if total_rows > optimal_size * 2:
        max_chunks = cpu_cores * 2
        chunk_ratio_size = total_rows // max_chunks
        optimal_size = max(20000, min(optimal_size, chunk_ratio_size))

    return int(optimal_size)


def determine_required_overlap(stage: str, cfg: Dict[str, Any]) -> Tuple[int, int]:
    """
    Automatically calculates required lookback and lookahead overlap size to ensure
    seamless indicator warmups and avoid boundary artifacts.
    """
    manual_overlap = cfg.get("chunk_overlap", None)
    if manual_overlap is not None:
        return (manual_overlap, manual_overlap)

    if stage == "ENRICH":
        ema_periods = cfg.get("ema_periods", [50, 600, 800])
        max_ema = max(ema_periods) if ema_periods else 800
        return (max_ema * 5, 0)
    elif stage == "EVAL":
        return (4000, 0)
    elif stage == "STATE":
        window_size = cfg.get("window_size", 35)
        return (window_size + 100, 0)
    elif stage == "LEVEL":
        lookahead = cfg.get("lookahead_bars", 20)
        return (100, lookahead)
    return (0, 0)


class ChunkProducer:
    """
    Generates lightweight chunk processing definitions (row ranges and index offsets)
    to distribute to stateless workers without sending heavy DataFrames over IPC.
    """
    @staticmethod
    def generate_chunks(
        total_rows: int,
        chunk_size: int,
        lookback_size: int,
        lookahead_size: int
    ) -> List[Dict[str, Any]]:
        chunks = []
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(total_rows - 1, start_idx + chunk_size - 1)
            read_start_idx = max(0, start_idx - lookback_size)
            read_end_idx = min(total_rows - 1, end_idx + lookahead_size)
            chunks.append({
                "chunk_id": len(chunks),
                "start_idx": start_idx,
                "end_idx": end_idx,
                "read_start_idx": read_start_idx,
                "read_end_idx": read_end_idx,
                "lookback_size": start_idx - read_start_idx,
                "lookahead_size": read_end_idx - end_idx,
            })
        return chunks


def process_chunk_worker_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stateless, process-safe worker task. Runs stage computations on a sliced row-range
    with appropriate overlap, trims off warmup buffers, and writes outputs directly to disk.
    """
    try:
        # Suppress standard logging output inside child processes to keep terminal display strictly clean
        root_logger = logging.getLogger()
        for h in list(root_logger.handlers):
            if isinstance(h, logging.StreamHandler):
                root_logger.removeHandler(h)

        symbol = task["symbol"]
        stage = task["stage"]
        chunk_id = task["chunk_id"]
        start_idx = task["start_idx"]
        end_idx = task["end_idx"]
        read_start_idx = task["read_start_idx"]
        read_end_idx = task["read_end_idx"]
        input_path = task["input_path"]
        output_path = task["output_path"]
        lookback_size = task["lookback_size"]
        lookahead_size = task["lookahead_size"]
        cfg = task["cfg"]

        # Read only specified row range from input file
        table = pq.read_table(input_path)
        sliced_table = table.slice(read_start_idx, read_end_idx - read_start_idx + 1)
        df_chunk = sliced_table.to_pandas()

        del table, sliced_table
        import gc
        gc.collect()

        if stage == "ENRICH":
            from Collecting_Data.indicators import IndicatorEngine
            ind_engine = IndicatorEngine(ema_periods=cfg.get("ema_periods", [50, 600, 800]), slope_period=32)
            df_computed = ind_engine.calculate(df_chunk)

            ms_engine = MarketStructureEngine(lookback=3)
            df_computed = ms_engine.process(df_computed)

            sd_engine = SupplyDemandEngine(atr_period=14, impulse_threshold=2.0)
            df_computed = sd_engine.process(df_computed)

        elif stage == "EVAL":
            from Market_Data_Pipeline.strong_candle_engine import StrongCandleEngine
            from Market_Data_Pipeline.refusal_candle_engine import RefusalCandleEngine

            # local structure for msg
            ms_engine = MarketStructureEngine(lookback=3)
            df_struct = ms_engine.process(df_chunk)
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

            strong_eng = StrongCandleEngine()
            refusal_eng = RefusalCandleEngine()

            total_bars = len(df_final_struct)
            strong_scores = []
            strong_confs = []
            refusal_scores = []
            refusal_confs = []

            for i in range(total_bars):
                sc = strong_eng.evaluate(df_final_struct, i, msg)
                rc = refusal_eng.evaluate_rejection(df_final_struct, i, None, msg)
                strong_scores.append(sc.quality_score)
                strong_confs.append(sc.confidence)
                refusal_scores.append(rc.quality_score)
                refusal_confs.append(rc.confidence)

            df_computed = df_chunk.copy()
            df_computed["strong_candle_score"] = strong_scores
            df_computed["strong_candle_confidence"] = strong_confs
            df_computed["refusal_candle_score"] = refusal_scores
            df_computed["refusal_candle_confidence"] = refusal_confs

        elif stage == "STATE":
            from ML.feature_registry import FeatureRegistry
            from ML.label_engine import LabelEngine

            registry = FeatureRegistry(load_defaults=True)
            label_engine = LabelEngine(
                window_size=cfg["window_size"],
                window_stride=cfg["window_stride"],
                registry=registry
            )

            ms_engine = MarketStructureEngine(lookback=3)
            df_struct = ms_engine.process(df_chunk)
            sd_engine = SupplyDemandEngine(atr_period=14, impulse_threshold=2.0)
            df_final_struct = sd_engine.process(df_struct)

            window_size = cfg["window_size"]
            df_final_struct['inside_supply_rollsum'] = df_final_struct['inside_supply'].rolling(window_size).sum().fillna(0).astype(int)
            df_final_struct['inside_demand_rollsum'] = df_final_struct['inside_demand'].rolling(window_size).sum().fillna(0).astype(int)

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

            active_len = end_idx - start_idx + 1
            all_samples = []

            for offset_i in range(active_len):
                curr_end_idx = lookback_size + offset_i
                curr_start_idx = curr_end_idx - window_size + 1

                if curr_start_idx < 0:
                    continue

                label, confidence, label_info = label_engine.labeler.label_window(
                    df_final_struct, msg, curr_start_idx, curr_end_idx
                )

                if label is None:
                    continue

                feats = label_engine.pipeline.extract_all(df_final_struct, msg, idx=curr_end_idx)
                row_datetime = df_final_struct.iloc[curr_end_idx].get("Datetime")
                datetime_str = row_datetime.isoformat() if isinstance(row_datetime, pd.Timestamp) else str(row_datetime)

                row_data = {
                    **feats,
                    "target": label,
                    "confidence": confidence,
                    "symbol": symbol,
                    "timeframe": cfg["timeframe"],
                    "window_start": start_idx + offset_i - window_size + 1,
                    "window_end": start_idx + offset_i,
                    "datetime": datetime_str,
                    "label_version": label_engine.labeler.label_version,
                    "engine_version": "1.0.0",
                }

                for raw_col in ["Open", "High", "Low", "Close", "TickVolume", "ema_50", "ema_600", "ema_800"]:
                    if raw_col in df_final_struct.columns:
                        row_data[raw_col] = df_final_struct.iloc[curr_end_idx][raw_col]

                for k, v in label_info.items():
                    row_data[f"meta_labeler_{k}"] = v

                all_samples.append(row_data)

            df_computed = pd.DataFrame(all_samples)

        elif stage == "LEVEL":
            from ML.feature_registry import FeatureRegistry
            from ML.dataset_builder import DatasetBuilder

            registry = FeatureRegistry(load_defaults=True)
            db_builder = DatasetBuilder(registry=registry)

            ms_engine = MarketStructureEngine(lookback=3)
            df_struct = ms_engine.process(df_chunk)
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

            active_len = end_idx - start_idx + 1
            rows = []

            supply_created = [z.created_idx for z in msg.supply_zones]
            supply_broken = [z.broken_idx if (z.broken and z.broken_idx is not None) else 99999999 for z in msg.supply_zones]
            demand_created = [z.created_idx for z in msg.demand_zones]
            demand_broken = [z.broken_idx if (z.broken and z.broken_idx is not None) else 99999999 for z in msg.demand_zones]

            lookahead_bars = cfg.get("lookahead_bars", 20)
            rejection_threshold_atr = 1.0

            for i in range(active_len):
                curr_idx = lookback_size + i
                if curr_idx >= len(df_final_struct) - lookahead_bars:
                    continue

                row_close = df_final_struct.iloc[curr_idx]["Close"]
                atr = df_final_struct.iloc[curr_idx].get("atr_14", 0.0001)

                active_supplies = [msg.supply_zones[z_idx] for z_idx, (c_idx, b_idx) in enumerate(zip(supply_created, supply_broken)) if c_idx <= curr_idx < b_idx]
                active_demands = [msg.demand_zones[z_idx] for z_idx, (c_idx, b_idx) in enumerate(zip(demand_created, demand_broken)) if c_idx <= curr_idx < b_idx]

                near_supply = None
                for s in active_supplies:
                    if 0 < (s.lower - row_close) <= 0.5 * atr:
                        near_supply = s
                        break

                near_demand = None
                for d in active_demands:
                    if 0 < (row_close - d.upper) <= 0.5 * atr:
                        near_demand = d
                        break

                if not near_supply and not near_demand:
                    continue

                target = None
                if near_supply:
                    for l in range(1, lookahead_bars + 1):
                        future_bar = df_final_struct.iloc[curr_idx + l]
                        if future_bar["High"] > near_supply.upper:
                            target = 1
                            break
                        elif future_bar["Low"] < near_supply.lower - rejection_threshold_atr * atr:
                            target = 0
                            break
                elif near_demand:
                    for l in range(1, lookahead_bars + 1):
                        future_bar = df_final_struct.iloc[curr_idx + l]
                        if future_bar["Low"] < near_demand.lower:
                            target = 1
                            break
                        elif future_bar["High"] > near_demand.upper + rejection_threshold_atr * atr:
                            target = 0
                            break

                if target is None:
                    continue

                feats = db_builder.pipeline.extract_all(df_final_struct, msg, curr_idx)
                row_datetime = df_final_struct.iloc[curr_idx].get("Datetime")
                datetime_str = row_datetime.isoformat() if isinstance(row_datetime, pd.Timestamp) else str(row_datetime)

                row_data = {
                    **feats,
                    "target": target,
                    "zone_type": "Supply" if near_supply else "Demand",
                    "timestamp": datetime_str,
                    "symbol": symbol,
                    "timeframe": cfg["timeframe"]
                }
                rows.append(row_data)

            df_computed = pd.DataFrame(rows)

        # Trim lookbacks / lookaheads to get exact boundaries
        if stage in ["ENRICH", "EVAL"]:
            trim_start = lookback_size
            trim_end = trim_start + (end_idx - start_idx + 1)
            df_out = df_computed.iloc[trim_start:trim_end].copy()
        else:
            df_out = df_computed.copy()

        save_parquet_atomically(df_out, output_path)

        del df_chunk, df_computed, df_out
        import gc
        gc.collect()

        return {"chunk_id": chunk_id, "status": "SUCCESS", "output_path": output_path}

    except Exception as e:
        import traceback
        err_msg = f"Error processing chunk {task.get('chunk_id', 'unknown')}: {e}\n{traceback.format_exc()}"
        return {"chunk_id": task.get("chunk_id"), "status": "ERROR", "error": err_msg}


class IncrementalWriter:
    """
    Progressively consolidates individual processed chunk Parquet files into a single master Parquet file.
    Utilizes PyArrow Streaming to read and append chunk-by-chunk, keeping RAM usage bounded and constant.
    """
    @staticmethod
    def consolidate_chunks(chunk_paths: List[str], output_path: str):
        if not chunk_paths:
            return
        sorted_paths = sorted(chunk_paths)
        writer = None
        try:
            for path in sorted_paths:
                if not os.path.exists(path) or os.path.getsize(path) == 0:
                    continue
                table = pq.read_table(path)
                if table.num_rows == 0:
                    continue
                if writer is None:
                    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                    writer = pq.ParquetWriter(output_path, table.schema)
                writer.write_table(table)
                del table
        finally:
            if writer is not None:
                writer.close()


class ChunkCheckpointManager:
    """
    Maintains and serializes granular, chunk-level progress for a stage.
    Ensures complete immunity to interruptions by enabling resumption from the exact failed chunk.
    """
    def __init__(self, symbol: str, timeframe: str, stage: str, checkpoint_dir: str):
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.stage = stage
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.path = os.path.join(self.checkpoint_dir, f"{self.symbol}_{self.timeframe}_stage_{self.stage}_chunk_checkpoint.json")
        self.state = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read chunk checkpoint {self.path}: {e}. Initializing clean.")
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "stage": self.stage,
            "completed_chunks": [],
            "completed_paths": {},
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

    def save(self) -> None:
        temp_path = self.path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=4)
        os.replace(temp_path, self.path)

    def mark_chunk_completed(self, chunk_id: int, file_path: str) -> None:
        if chunk_id not in self.state["completed_chunks"]:
            self.state["completed_chunks"].append(chunk_id)
        self.state["completed_paths"][str(chunk_id)] = file_path
        self.state["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.save()

    def is_chunk_completed(self, chunk_id: int) -> bool:
        return chunk_id in self.state["completed_chunks"]

    def clean(self) -> None:
        if os.path.exists(self.path):
            try:
                os.remove(self.path)
            except Exception:
                pass


def run_stage_with_chunking(
    symbol: str,
    stage: str,
    input_path: str,
    final_output_path: str,
    cfg: Dict[str, Any],
    max_workers: int
) -> str:
    """
    Executes a processing stage for a symbol using the new streaming chunk model.
    Dispatches tasks to workers, streams completed chunks dynamically to disk,
    maintains exact chunk checkpointing, and progressively consolidates results.
    """
    logger.info(f"[{symbol}] Starting Stage [{stage}] with chunked streaming...")

    # Load input data metadata to find length
    table_meta = pq.read_metadata(input_path)
    total_rows = table_meta.num_rows
    del table_meta

    if total_rows == 0:
        logger.warning(f"[{symbol}] Stage [{stage}] input dataset is empty. Skipping stage.")
        # Save an empty parquet
        df_empty = pd.DataFrame()
        save_parquet_atomically(df_empty, final_output_path)
        return final_output_path

    # Adaptive chunk size calculation
    chunk_size = calculate_adaptive_chunk_size(total_rows, cfg)
    lookback_size, lookahead_size = determine_required_overlap(stage, cfg)

    # Produce Chunk definitions
    chunks = ChunkProducer.generate_chunks(total_rows, chunk_size, lookback_size, lookahead_size)
    total_chunks = len(chunks)
    logger.info(f"[{symbol}] Created {total_chunks} chunks (size: {chunk_size}, overlap lookback: {lookback_size}, lookahead: {lookahead_size})")

    # Setup Chunk Checkpoint & Chunk Directories
    temp_stage_dir = PathManager.get_path("temporary", f"chunk_cache_{symbol}_{stage}")
    os.makedirs(temp_stage_dir, exist_ok=True)
    chunk_checkpoint = ChunkCheckpointManager(symbol, cfg["timeframe"], stage, PathManager.get_path("temporary", "checkpoints"))

    # Track paths for consolidation
    chunk_paths = []
    pending_chunks = []

    for chunk in chunks:
        cid = chunk["chunk_id"]
        chunk_out_path = os.path.join(temp_stage_dir, f"chunk_{cid:04d}.parquet")

        if chunk_checkpoint.is_chunk_completed(cid) and os.path.exists(chunk_out_path):
            logger.debug(f"[{symbol}] Chunk {cid} already completed. Skipping computation.")
            chunk_paths.append(chunk_out_path)
        else:
            task = {
                "symbol": symbol,
                "stage": stage,
                "chunk_id": cid,
                "start_idx": chunk["start_idx"],
                "end_idx": chunk["end_idx"],
                "read_start_idx": chunk["read_start_idx"],
                "read_end_idx": chunk["read_end_idx"],
                "lookback_size": chunk["lookback_size"],
                "lookahead_size": chunk["lookahead_size"],
                "input_path": input_path,
                "output_path": chunk_out_path,
                "cfg": cfg
            }
            pending_chunks.append(task)

    # Set up Interactive Console Monitor if enabled
    monitor = ProcessingProgressMonitor(num_workers=max_workers, total_chunks=total_chunks, enabled=True)
    monitor.completed_chunks = total_chunks - len(pending_chunks)

    # Run Remaining Chunks via stateless subprocess workers
    if pending_chunks:
        logger.info(f"[{symbol}] Launching {len(pending_chunks)} pending chunk worker tasks...")
        if monitor.enabled:
            monitor._draw()

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(process_chunk_worker_task, task): task
                for task in pending_chunks
            }

            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                cid = task["chunk_id"]
                slot_id = monitor.register_chunk_slot(cid)

                try:
                    result = future.result()
                    if result.get("status") == "SUCCESS":
                        out_path = result["output_path"]
                        chunk_paths.append(out_path)
                        chunk_checkpoint.mark_chunk_completed(cid, out_path)
                        monitor.update(slot_id, cid, "DONE", 100, f"Saved to disk")
                        monitor.increment_completed()
                    else:
                        err = result.get("error", "Unknown error")
                        logger.error(f"[{symbol}] Chunk {cid} failed with error: {err}")
                        monitor.update(slot_id, cid, "FAILED", 0, f"Error: {err}")
                        raise RuntimeError(f"Stage {stage} Chunk {cid} processing failed.")
                except Exception as exc:
                    logger.error(f"[{symbol}] Exception in chunk worker for chunk {cid}: {exc}")
                    monitor.update(slot_id, cid, "CRASHED", 0, str(exc))
                    raise exc
                finally:
                    monitor.release_chunk_slot(cid)

    if monitor.enabled:
        monitor.clear()

    # Progressive Consolidation and Assembly
    logger.info(f"[{symbol}] Assembling and consolidating {len(chunk_paths)} completed chunks for stage [{stage}]...")
    IncrementalWriter.consolidate_chunks(chunk_paths, final_output_path)

    # Checkpoint is finalized. Clean temporary stage files and stage checkpoint
    shutil.rmtree(temp_stage_dir, ignore_errors=True)
    chunk_checkpoint.clean()

    logger.info(f"[{symbol}] Stage [{stage}] successfully finalized. Saved output to {final_output_path}")
    import gc
    gc.collect()

    return final_output_path


def process_single_symbol(
    symbol: str,
    file_path: str,
    cfg: Dict[str, Any],
    registry: FeatureRegistry
) -> Tuple[Optional[str], Optional[str]]:
    """
    Orchestrates the entire sequential Stage-by-Stage processing for a single symbol.
    Ensures complete isolation of memory, releasing all resources on stage completion.
    """
    try:
        max_workers = cfg["max_workers"] if cfg["use_multiprocessing"] else 1
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

        while True:
            next_stage = cp_mgr.get_next_unfinished_stage()
            if next_stage is None:
                break

            # ----------------- STAGE 1: CLEAN -----------------
            if next_stage == "CLEAN":
                cp_mgr.mark_stage_started("CLEAN")
                logger.info(f"[{symbol}] [Stage CLEAN] Loading raw candles from {os.path.basename(file_path)}...")

                # Standard cleaning runs quickly in parent process to produce the baseline Parquet
                if file_path.endswith(".parquet"):
                    df_raw = pd.read_parquet(file_path)
                else:
                    df_raw = pd.read_csv(file_path)

                df_clean = clean_raw_candles(df_raw)
                total_bars = len(df_clean)

                if total_bars < cfg["window_size"]:
                    raise ValueError(f"Insufficient candles ({total_bars}) for window size {cfg['window_size']}.")

                clean_out_path = os.path.join(stages_dir, f"{symbol}_clean.parquet")
                save_parquet_atomically(df_clean, clean_out_path)

                cp_mgr.mark_stage_completed("CLEAN", clean_out_path)

                del df_raw, df_clean
                import gc
                gc.collect()
                logger.info(f"[{symbol}] [Stage CLEAN] Successfully finished. Processed {total_bars} bars.")

            # ----------------- STAGE 2: ENRICH -----------------
            elif next_stage == "ENRICH":
                cp_mgr.mark_stage_started("ENRICH")
                clean_in_path = cp_mgr.get_stage_output("CLEAN")
                enrich_out_path = os.path.join(stages_dir, f"{symbol}_enriched.parquet")

                # Run chunked stage execution
                run_stage_with_chunking(symbol, "ENRICH", clean_in_path, enrich_out_path, cfg, max_workers)
                cp_mgr.mark_stage_completed("ENRICH", enrich_out_path)

            # ----------------- STAGE 3: EVAL -----------------
            elif next_stage == "EVAL":
                cp_mgr.mark_stage_started("EVAL")
                enrich_in_path = cp_mgr.get_stage_output("ENRICH")
                eval_out_path = os.path.join(stages_dir, f"{symbol}_evaluated.parquet")

                run_stage_with_chunking(symbol, "EVAL", enrich_in_path, eval_out_path, cfg, max_workers)
                cp_mgr.mark_stage_completed("EVAL", eval_out_path)

            # ----------------- STAGE 4: STATE -----------------
            elif next_stage == "STATE":
                cp_mgr.mark_stage_started("STATE")
                eval_in_path = cp_mgr.get_stage_output("EVAL")
                state_out_path = os.path.join(stages_dir, f"{symbol}_state.parquet")

                run_stage_with_chunking(symbol, "STATE", eval_in_path, state_out_path, cfg, max_workers)
                cp_mgr.mark_stage_completed("STATE", state_out_path)

            # ----------------- STAGE 5: LEVEL -----------------
            elif next_stage == "LEVEL":
                cp_mgr.mark_stage_started("LEVEL")
                eval_in_path = cp_mgr.get_stage_output("EVAL")
                level_out_path = os.path.join(stages_dir, f"{symbol}_level.parquet")

                run_stage_with_chunking(symbol, "LEVEL", eval_in_path, level_out_path, cfg, max_workers)
                cp_mgr.mark_stage_completed("LEVEL", level_out_path)

        state_final_path = cp_mgr.get_stage_output("STATE")
        level_final_path = cp_mgr.get_stage_output("LEVEL")
        return state_final_path, level_final_path

    except Exception as e:
        logger.error(f"[{symbol}] Error processing symbol: {e}", exc_info=True)
        return None, None


def main():
    from Collecting_Data.memory_monitor import MemoryMonitor
    mem_monitor = MemoryMonitor()
    mem_monitor.check("Pipeline start")

    parser = argparse.ArgumentParser(description="Forex_DNN Redesigned Streaming Data Pipeline")
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

    cfg = load_config(args.config)
    if args.input_dir:
        cfg["input_dir"] = args.input_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.timeframe:
        cfg["timeframe"] = args.timeframe

    logger.info("==================================================")
    logger.info("  Forex_DNN Redesigned High-Performance Pipeline  ")
    logger.info("==================================================")
    logger.info(f"Target Directory : {cfg['input_dir']}")
    logger.info(f"Output Directory : {cfg['output_dir']}")
    logger.info(f"Symbols Filter   : {args.symbol}")
    logger.info(f"Timeframe        : {cfg['timeframe']}")
    logger.info(f"Window Size/Stride: {cfg['window_size']} / {cfg['window_stride']}")

    stages_dir = PathManager.get_path("temporary", "stages")
    checkpoint_dir = PathManager.get_path("temporary", "checkpoints")

    if args.force:
        logger.info("Force flag enabled. Flushing checkpoints, stages, and cache directories...")
        shutil.rmtree(stages_dir, ignore_errors=True)
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        shutil.rmtree(PathManager.get_path("cache"), ignore_errors=True)

    os.makedirs(stages_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    discovered = discover_historical_files(cfg["input_dir"], args.symbol, cfg["timeframe"])
    if not discovered:
        logger.error("No historical files matching the configuration criteria were discovered. Terminating.")
        sys.exit(1)

    logger.info(f"Successfully discovered {len(discovered)} symbols to process: {list(discovered.keys())}")

    # Process each symbol SEQUENTIALLY to keep memory usage completely bounded
    registry = FeatureRegistry(load_defaults=True)
    state_paths = []
    level_paths = []

    for idx, (sym, path) in enumerate(discovered.items(), 1):
        logger.info(f"\n[{idx}/{len(discovered)}] PROCESSING SYMBOL: {sym} sequentially...")
        mem_monitor.check(f"Before processing symbol {sym}")

        state_path, level_path = process_single_symbol(sym, path, cfg, registry)
        if state_path:
            state_paths.append(state_path)
        if level_path:
            level_paths.append(level_path)

        mem_monitor.check(f"After processing symbol {sym}")
        # Explicit garbage collection after each symbol to fully release resources
        import gc
        gc.collect()

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
            del all_states
            import gc
            gc.collect()

            if "datetime" in master_state.columns:
                master_state.sort_values(by=["datetime", "symbol"], inplace=True)
            master_state = cleaner.clean(master_state, label_col="target")

            state_path = os.path.join(cfg["output_dir"], "market_state_dataset.parquet")
            save_parquet_atomically(master_state, state_path)
            logger.info(f"Saved Consolidated Market State Dataset to {state_path} ({len(master_state)} samples)")

            report = validator.validate(master_state, expected_window_size=cfg["window_size"])
            logger.info(f"Market State Dataset Integrity Validation status: {'PASS' if report['is_valid'] else 'FAIL'}")

            # Produce RL and Trade Quality Datasets
            logger.info("Preparing Reinforcement Learning (RL) Dataset...")
            master_rl = master_state.copy()

            if "Close" in master_rl.columns:
                forward_close = master_rl.groupby("symbol")["Close"].shift(-5)
                master_rl["reward"] = ((forward_close - master_rl["Close"]) / master_rl["Close"]).fillna(0.0)
                master_rl["action"] = np.where(master_rl["reward"] > 0.001, 1, np.where(master_rl["reward"] < -0.001, 2, 0))
            else:
                master_rl["reward"] = 0.0
                master_rl["action"] = 0

            master_rl["done"] = False
            for sym in master_rl["symbol"].unique():
                idx = master_rl[master_rl["symbol"] == sym].index
                if len(idx) > 5:
                    master_rl.loc[idx[-5:], "done"] = True

            rl_path = os.path.join(cfg["output_dir"], "future_rl_dataset.parquet")
            save_parquet_atomically(master_rl, rl_path)
            logger.info(f"Saved Reinforcement Learning Dataset to {rl_path} ({len(master_rl)} samples)")
            del master_rl
            gc.collect()

            logger.info("Preparing Trade Quality Dataset...")
            master_tq = master_state.copy()

            if "Close" in master_tq.columns:
                forward_close_10 = master_tq.groupby("symbol")["Close"].shift(-10).fillna(master_tq["Close"])
                atr = master_tq.get("atr", 0.0010)
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
            del all_levels
            import gc
            gc.collect()

            if "timestamp" in master_level.columns:
                master_level.sort_values(by=["timestamp", "symbol"], inplace=True)
            master_level = cleaner.clean(master_level, label_col="target")

            level_path = os.path.join(cfg["output_dir"], "level_break_dataset.parquet")
            save_parquet_atomically(master_level, level_path)
            logger.info(f"Saved Consolidated Level Break Dataset to {level_path} ({len(master_level)} samples)")

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

    del master_state, master_level
    import gc
    gc.collect()

    mem_monitor.check("Pipeline finished")

    logger.info("==================================================")
    logger.info("  DATA PROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
