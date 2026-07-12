"""
historical_data_collector.py
----------------------------
Official historical data downloader for the Forex_DNN framework.
Features:
- Decoupled provider architecture (BaseDataProvider, MT5DataProvider, and others).
- Symbol discovery with automatic visible symbol selection and filtering.
- Chunked downloading with configurable size and retries.
- Pause/resume capability using download_state.json.
- Incremental process with strict memory management.
- Data validation (missing timestamps, duplicates, invalid OHLC, negative volume).
- Consolidated download report and metadata generation.
- Concurrency with configurable workers.
"""

import os
import sys
import json
import time
import logging
import argparse
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

# Try to import MetaTrader5
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("HistoricalDataCollector")


# ── Timeframe Helpers ─────────────────────────────────────────────────────────
TIMEFRAME_DURATIONS = {
    "M1": timedelta(minutes=1),
    "M5": timedelta(minutes=5),
    "M15": timedelta(minutes=15),
    "M30": timedelta(minutes=30),
    "H1": timedelta(hours=1),
    "H4": timedelta(hours=4),
    "D1": timedelta(days=1),
}


# ── Providers Abstraction ─────────────────────────────────────────────────────

class BaseDataProvider(ABC):
    """
    Abstract Base Class for historical data providers.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Initialize connection to the provider."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the provider."""
        pass

    @abstractmethod
    def get_visible_symbols(self, filter_type: str = "all") -> List[str]:
        """
        Retrieve visible/available symbols from the provider.
        Filters can be: 'forex', 'metals', 'indices', 'crypto', 'all'
        """
        pass

    @abstractmethod
    def download_chunk(
        self, symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Download historical bars for a specific chunk.
        Returned columns must be: ['Datetime', 'Open', 'High', 'Low', 'Close', 'TickVolume', 'Spread']
        Datetime must be a UTC-localized pandas DatetimeIndex or Series.
        """
        pass

    @abstractmethod
    def get_broker_name(self) -> str:
        """Return the name of the broker/provider."""
        pass

    @abstractmethod
    def get_timezone(self) -> str:
        """Return the timezone of the provider data (e.g. UTC, GMT+2, etc.)."""
        pass


class MT5DataProvider(BaseDataProvider):
    """
    Concrete implementation of BaseDataProvider for MetaTrader 5.
    """

    def __init__(self, login: Optional[int] = None, password: Optional[str] = None, server: Optional[str] = None):
        self.login = login
        self.password = password
        self.server = server
        self.connected = False
        self._broker = "Unknown_MT5_Broker"

    def connect(self) -> bool:
        if mt5 is None:
            logger.error("MetaTrader5 library is not installed or not supported on this platform.")
            return False

        # If credentials are not explicitly passed, try to load from credentials.json
        if not self.login:
            try:
                from Collecting_Data.auth import load_credentials
                creds = load_credentials("credentials.json")
                self.login = creds.get("login")
                self.password = creds.get("password")
                self.server = creds.get("server")
            except Exception as e:
                logger.warning(f"Could not load credentials.json: {e}. Attempting default terminal init.")

        if self.login and self.password and self.server:
            if not mt5.initialize(login=self.login, password=self.password, server=self.server):
                logger.error(f"MT5 initialization failed with credentials: {mt5.last_error()}")
                return False
        else:
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed with default terminal: {mt5.last_error()}")
                return False

        self.connected = True
        terminal_info = mt5.terminal_info()
        if terminal_info:
            self._broker = getattr(terminal_info, "company", "MT5_Broker")
        logger.info(f"MT5 Provider connected to broker: {self._broker}")
        return True

    def disconnect(self) -> None:
        if self.connected and mt5 is not None:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 Provider disconnected.")

    def get_visible_symbols(self, filter_type: str = "all") -> List[str]:
        if not self.connected or mt5 is None:
            logger.error("MT5 Provider not connected.")
            return []

        symbols = mt5.symbols_get()
        if symbols is None:
            logger.error(f"Failed to get symbols from MT5: {mt5.last_error()}")
            return []

        visible_symbols = []
        filter_type = filter_type.lower()

        for s in symbols:
            name = s.name.upper()
            path = getattr(s, "path", "").lower()

            # Classify symbol based on path/name
            is_forex = "forex" in path or "fx" in path or s.currency_base != ""
            is_metal = "metal" in path or any(m in name for m in ["XAU", "XAG", "GOLD", "SILVER"])
            is_index = "indices" in path or "index" in path or any(idx in name for idx in ["FDAX", "YM", "US30", "SPX", "NAS100", "DE30", "GER30", "US500", "EU50"])
            is_crypto = "crypto" in path or "coin" in path or any(cry in name for cry in ["BTC", "ETH", "LTC", "XRP"])

            if filter_type == "forex" and not is_forex:
                continue
            elif filter_type == "metals" and not is_metal:
                continue
            elif filter_type == "indices" and not is_index:
                continue
            elif filter_type == "crypto" and not is_crypto:
                continue

            # Select symbol to ensure it can be visible/subscribed in Market Watch
            if mt5.symbol_select(s.name, True):
                visible_symbols.append(s.name)

        return sorted(visible_symbols)

    def download_chunk(
        self, symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime
    ) -> Optional[pd.DataFrame]:
        if not self.connected or mt5 is None:
            logger.error("MT5 Provider not connected.")
            return None

        # Convert timeframe string to MT5 constant
        tf_mapping = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        mt5_tf = tf_mapping.get(timeframe)
        if mt5_tf is None:
            logger.error(f"Unsupported timeframe: {timeframe}")
            return None

        # Call copy_rates_range
        logger.debug(f"Calling copy_rates_range for {symbol} ({start_dt} to {end_dt})")
        rates = mt5.copy_rates_range(symbol, mt5_tf, start_dt, end_dt)

        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)

        # Rename columns to standard schema
        rename_map = {
            "time": "Datetime",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tick_volume": "TickVolume",
            "spread": "Spread",
        }
        available_cols = [c for c in rename_map if c in df.columns]
        df = df[available_cols].rename(columns=rename_map)

        # Re-index to ensure schema correctness
        expected_cols = ["Open", "High", "Low", "Close", "TickVolume", "Spread"]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = np.nan

        # Convert time to datetime index/series
        df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s", utc=True)
        df = df[["Datetime", "Open", "High", "Low", "Close", "TickVolume", "Spread"]]

        return df

    def get_broker_name(self) -> str:
        return self._broker

    def get_timezone(self) -> str:
        return "UTC"


class MockDataProvider(BaseDataProvider):
    """
    Mock Data Provider for testing and validation.
    Generates deterministic bars or returns pre-defined chunks.
    """

    def __init__(self, symbols: List[str] = None):
        self._symbols = symbols or ["EURUSD", "GBPUSD"]
        self.connected = False

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.connected = False

    def get_visible_symbols(self, filter_type: str = "all") -> List[str]:
        return self._symbols

    def download_chunk(
        self, symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime
    ) -> Optional[pd.DataFrame]:
        # Generate dummy 5-minute candles between start_dt and end_dt
        if start_dt >= end_dt:
            return None

        tf_dur = TIMEFRAME_DURATIONS.get(timeframe, timedelta(minutes=5))
        timestamps = []
        curr = start_dt
        while curr < end_dt:
            timestamps.append(curr)
            curr += tf_dur

        if not timestamps:
            return None

        # Build synthetic DataFrame
        n = len(timestamps)
        np.random.seed(42 + n) # deterministic seed
        opens = np.linspace(1.1000, 1.1050, n) + np.random.normal(0, 0.001, n)
        highs = opens + np.random.uniform(0.0005, 0.0015, n)
        lows = opens - np.random.uniform(0.0005, 0.0015, n)
        closes = opens + np.random.normal(0, 0.0005, n)
        closes = np.clip(closes, lows, highs)

        df = pd.DataFrame({
            "Datetime": pd.to_datetime(timestamps, utc=True),
            "Open": opens.astype(float),
            "High": highs.astype(float),
            "Low": lows.astype(float),
            "Close": closes.astype(float),
            "TickVolume": np.random.randint(10, 150, n).astype(int),
            "Spread": np.ones(n, dtype=int)
        })
        return df

    def get_broker_name(self) -> str:
        return "MockBroker"

    def get_timezone(self) -> str:
        return "UTC"


# ── HistoricalDataCollector Orchestrator ──────────────────────────────────────

class HistoricalDataCollector:
    """
    Orchestrates historical data downloading, validation, state resume, and parallelization.
    """

    def __init__(
        self,
        provider: BaseDataProvider,
        output_dir: str = "HistoricalData",
        chunk_days: int = 180,
        max_retries: int = 3,
        retry_delay: int = 5,
        num_workers: int = 4,
    ):
        self.provider = provider
        self.output_dir = output_dir
        self.chunk_days = chunk_days
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.num_workers = num_workers
        self.state_file = os.path.join(output_dir, "download_state.json")
        self.report_file = os.path.join(output_dir, "download_report.csv")

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load download state: {e}. Starting fresh state.")
        return {}

    def _save_state(self) -> None:
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save download state: {e}")

    def update_state(self, symbol: str, timeframe: str, last_dt: datetime, chunk_idx: int) -> None:
        key = f"{symbol}_{timeframe}"
        self.state[key] = {
            "last_downloaded_datetime": last_dt.isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "chunk_index": chunk_idx,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_state()

    def get_resume_info(self, symbol: str, timeframe: str, default_start: datetime) -> Tuple[datetime, int]:
        key = f"{symbol}_{timeframe}"
        if key in self.state:
            info = self.state[key]
            try:
                last_dt = datetime.fromisoformat(info["last_downloaded_datetime"])
                chunk_idx = info.get("chunk_index", 0)
                logger.info(f"Resuming {symbol} {timeframe} from {last_dt} (chunk index {chunk_idx})")
                return last_dt, chunk_idx
            except Exception as e:
                logger.warning(f"Error parsing resume info for {key}: {e}. Restarting from default start.")
        return default_start, 0

    # ── Validation Engine ─────────────────────────────────────────────────────

    @staticmethod
    def validate_chunk(df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """
        Validates missing timestamps, duplicates, invalid OHLC, negative volume, timezone consistency.
        """
        report = {
            "missing_timestamps": 0,
            "duplicates": 0,
            "invalid_ohlc": 0,
            "negative_volume": 0,
            "timezone_consistent": True,
            "errors": []
        }

        if df.empty:
            return report

        # Timezone check
        has_tz = df["Datetime"].dt.tz is not None
        if not has_tz:
            report["timezone_consistent"] = False
            report["errors"].append("Missing timezone localization.")
        else:
            # Check if all are UTC or consistent
            tz_names = df["Datetime"].dt.tz.zone if hasattr(df["Datetime"].dt.tz, "zone") else str(df["Datetime"].dt.tz)
            if "UTC" not in tz_names.upper() and "+00:00" not in tz_names:
                report["errors"].append(f"Non-UTC timezone detected: {tz_names}")

        # Duplicate check
        dup_count = df.duplicated(subset=["Datetime"]).sum()
        report["duplicates"] = int(dup_count)
        if dup_count > 0:
            report["errors"].append(f"Found {dup_count} duplicate timestamps.")

        # Invalid OHLC check
        # High must be max, Low must be min, all values > 0
        invalid_mask = (
            (df["High"] < df["Low"]) |
            (df["High"] < df["Open"]) |
            (df["High"] < df["Close"]) |
            (df["Low"] > df["Open"]) |
            (df["Low"] > df["Close"]) |
            (df["Open"] <= 0) |
            (df["High"] <= 0) |
            (df["Low"] <= 0) |
            (df["Close"] <= 0)
        )
        invalid_count = invalid_mask.sum()
        report["invalid_ohlc"] = int(invalid_count)
        if invalid_count > 0:
            report["errors"].append(f"Found {invalid_count} bars with invalid OHLC boundaries or <= 0 values.")

        # Negative volume
        neg_vol_count = (df["TickVolume"] < 0).sum()
        report["negative_volume"] = int(neg_vol_count)
        if neg_vol_count > 0:
            report["errors"].append(f"Found {neg_vol_count} bars with negative volume.")

        # Missing timestamps check (Continuity check)
        tf_dur = TIMEFRAME_DURATIONS.get(timeframe)
        if tf_dur and len(df) > 1:
            diff_seconds = df["Datetime"].diff().dt.total_seconds()
            expected_seconds = tf_dur.total_seconds()

            # Detect where gaps exist (greater than 1.5 times the timeframe)
            gaps_mask = diff_seconds > (expected_seconds * 1.5)
            # Ignore weekend gaps (gaps around 45 hours or more are typical weekend closures)
            weekend_seconds = 45 * 3600

            missing_count = 0
            for sec in diff_seconds[gaps_mask]:
                if sec < weekend_seconds:
                    # Estimate number of missing bars
                    missing_bars = int(sec / expected_seconds) - 1
                    if missing_bars > 0:
                        missing_count += missing_bars

            report["missing_timestamps"] = missing_count
            if missing_count > 0:
                report["errors"].append(f"Detected approximately {missing_count} missing candles (excluding weekends).")

        return report

    # ── Downloading and Merging ───────────────────────────────────────────────

    def download_symbol(self, symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime) -> Dict[str, Any]:
        """
        Download complete available history for a single symbol in chunks, validate, and save.
        """
        t_start = time.perf_counter()
        logger.info(f"Starting download for {symbol} {timeframe}")

        # Determine start date (support resume)
        current_start, chunk_idx = self.get_resume_info(symbol, timeframe, start_dt)
        if current_start >= end_dt:
            logger.info(f"Symbol {symbol} {timeframe} is already up to date.")
            return {
                "Symbol": symbol,
                "Bars": 0,
                "Start": current_start.isoformat(),
                "End": end_dt.isoformat(),
                "Status": "SKIPPED",
                "Elapsed Time": 0.0,
                "Errors": 0
            }

        # Calculate chunks of date ranges
        chunk_ranges = []
        temp_dt = current_start
        while temp_dt < end_dt:
            next_dt = min(temp_dt + timedelta(days=self.chunk_days), end_dt)
            chunk_ranges.append((temp_dt, next_dt))
            temp_dt = next_dt

        total_chunks = len(chunk_ranges)
        logger.info(f"{symbol} download split into {total_chunks} chunks.")

        symbol_dir = os.path.join(self.output_dir, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        chunk_files = []

        total_bars_downloaded = 0
        total_missing = 0
        total_duplicates = 0
        total_errors_logged = 0

        # Download chunks sequentially
        for idx, (c_start, c_end) in enumerate(chunk_ranges):
            actual_chunk_idx = chunk_idx + idx
            logger.info(f"Downloading {symbol} {timeframe} - Chunk {actual_chunk_idx + 1}/{total_chunks + chunk_idx}")

            df_chunk = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    df_chunk = self.provider.download_chunk(symbol, timeframe, c_start, c_end)
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt} failed for chunk {actual_chunk_idx + 1}: {e}")
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay)
                    else:
                        logger.error(f"Failed to download chunk {actual_chunk_idx + 1} after {self.max_retries} attempts.")

            if df_chunk is None or df_chunk.empty:
                logger.warning(f"No bars returned for {symbol} in range {c_start} to {c_end}.")
                continue

            # Validate chunk
            val_report = self.validate_chunk(df_chunk, timeframe)
            total_missing += val_report["missing_timestamps"]
            total_duplicates += val_report["duplicates"]
            if val_report["errors"]:
                total_errors_logged += len(val_report["errors"])
                for err in val_report["errors"]:
                    logger.warning(f"[{symbol} {timeframe} Chunk {actual_chunk_idx + 1}] Validation: {err}")

            # Save chunk incrementally to prevent massive RAM usage
            chunk_file = os.path.join(symbol_dir, f"chunk_{actual_chunk_idx}.parquet")
            try:
                df_chunk.to_parquet(chunk_file, index=False)
                chunk_files.append(chunk_file)
                total_bars_downloaded += len(df_chunk)
                logger.info(f"Saved chunk {actual_chunk_idx + 1} with {len(df_chunk)} bars.")
            except Exception as e:
                logger.error(f"Failed to write incremental chunk file {chunk_file}: {e}")
                return {
                    "Symbol": symbol,
                    "Bars": 0,
                    "Start": start_dt.isoformat(),
                    "End": end_dt.isoformat(),
                    "Status": "FAILED",
                    "Elapsed Time": time.perf_counter() - t_start,
                    "Errors": 1
                }

            # Update resume state
            self.update_state(symbol, timeframe, c_end, actual_chunk_idx + 1)

        # Merge and finalize
        # Discover all chunk parquet files in the symbol directory (including previously interrupted ones)
        import glob
        discovered_chunks = sorted(
            glob.glob(os.path.join(symbol_dir, "chunk_*.parquet")),
            key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
        )

        if not discovered_chunks:
            logger.warning(f"No historical data chunks found for {symbol}.")
            return {
                "Symbol": symbol,
                "Bars": 0,
                "Start": start_dt.isoformat(),
                "End": end_dt.isoformat(),
                "Status": "NO_DATA",
                "Elapsed Time": time.perf_counter() - t_start,
                "Errors": 0
            }

        logger.info(f"Merging {len(discovered_chunks)} chunks for {symbol}...")
        merged_df_list = []
        for cf in discovered_chunks:
            try:
                merged_df_list.append(pd.read_parquet(cf))
            except Exception as e:
                logger.error(f"Failed to read chunk file {cf} during merge: {e}")

        # Let's see if we have existing data to merge as well
        final_csv_path = os.path.join(symbol_dir, f"{timeframe}.csv")
        final_parquet_path = os.path.join(symbol_dir, f"{timeframe}.parquet")

        if os.path.exists(final_parquet_path):
            try:
                logger.info(f"Loading existing historical Parquet for merging: {final_parquet_path}")
                existing_df = pd.read_parquet(final_parquet_path)
                merged_df_list.insert(0, existing_df)
            except Exception as e:
                logger.warning(f"Could not load existing Parquet: {e}")

        if not merged_df_list:
            return {
                "Symbol": symbol,
                "Bars": 0,
                "Start": start_dt.isoformat(),
                "End": end_dt.isoformat(),
                "Status": "FAILED",
                "Elapsed Time": time.perf_counter() - t_start,
                "Errors": 1
            }

        full_df = pd.concat(merged_df_list, ignore_index=True)

        # Remove duplicates
        initial_len = len(full_df)
        full_df.drop_duplicates(subset=["Datetime"], keep="first", inplace=True)
        duplicates_removed = initial_len - len(full_df)

        # Sort chronologically
        full_df.sort_values(by="Datetime", inplace=True)
        full_df.reset_index(drop=True, inplace=True)

        # Validate Continuity
        final_val_report = self.validate_chunk(full_df, timeframe)
        if final_val_report["errors"]:
            logger.info(f"Final dataset validation report contains {len(final_val_report['errors'])} notes.")

        # Save finalized formats
        try:
            full_df.to_parquet(final_parquet_path, index=False)
            full_df.to_csv(final_csv_path, index=False)
            logger.info(f"Saved finalized dataset to Parquet & CSV for {symbol}. Total Bars: {len(full_df)}")
        except Exception as e:
            logger.error(f"Failed to save final merged files: {e}")
            return {
                "Symbol": symbol,
                "Bars": len(full_df),
                "Start": start_dt.isoformat(),
                "End": end_dt.isoformat(),
                "Status": "WRITE_ERROR",
                "Elapsed Time": time.perf_counter() - t_start,
                "Errors": 1
            }

        # Clear chunk files to save disk space
        for cf in discovered_chunks:
            try:
                os.remove(cf)
            except OSError as e:
                logger.warning(f"Failed to delete temp chunk file {cf}: {e}")

        # Metadata generation
        metadata = {
            "symbol": symbol,
            "timeframe": timeframe,
            "download_date": datetime.now(timezone.utc).isoformat(),
            "first_candle": full_df["Datetime"].min().isoformat() if not full_df.empty else None,
            "last_candle": full_df["Datetime"].max().isoformat() if not full_df.empty else None,
            "number_of_bars": len(full_df),
            "duplicates_removed": int(duplicates_removed + total_duplicates),
            "missing_candles": int(final_val_report["missing_timestamps"]),
            "broker": self.provider.get_broker_name(),
            "timezone": self.provider.get_timezone()
        }

        metadata_file = os.path.join(symbol_dir, "metadata.json")
        try:
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save metadata.json: {e}")

        elapsed_time = time.perf_counter() - t_start
        logger.info(f"Completed download & process for {symbol} {timeframe} in {elapsed_time:.2f} seconds.")

        return {
            "Symbol": symbol,
            "Bars": len(full_df),
            "Start": full_df["Datetime"].min().isoformat() if not full_df.empty else start_dt.isoformat(),
            "End": full_df["Datetime"].max().isoformat() if not full_df.empty else end_dt.isoformat(),
            "Status": "SUCCESS",
            "Elapsed Time": elapsed_time,
            "Errors": total_errors_logged
        }

    def download_multiple_symbols(
        self, symbols: List[str], timeframe: str, start_dt: datetime, end_dt: datetime
    ) -> List[Dict[str, Any]]:
        """Download multiple symbols concurrently using ThreadPoolExecutor."""
        reports = []

        logger.info(f"Starting parallel download for {len(symbols)} symbols with {self.num_workers} workers.")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_symbol = {
                executor.submit(self.download_symbol, sym, timeframe, start_dt, end_dt): sym
                for sym in symbols
            }

            for future in as_completed(future_to_symbol):
                sym = future_to_symbol[future]
                try:
                    rep = future.result()
                    reports.append(rep)
                except Exception as e:
                    logger.error(f"Exception raised during download of {sym}: {e}", exc_info=True)
                    reports.append({
                        "Symbol": sym,
                        "Bars": 0,
                        "Start": start_dt.isoformat(),
                        "End": end_dt.isoformat(),
                        "Status": "EXCEPTION",
                        "Elapsed Time": 0.0,
                        "Errors": 1
                    })

        # Save unified report
        df_report = pd.DataFrame(reports)
        df_report.to_csv(self.report_file, index=False)
        logger.info(f"Consolidated download report written to {self.report_file}")

        return reports


# ── Command Line Interface ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Forex_DNN Historical Data Collector")
    parser.add_argument("--timeframe", type=str, default="M5", choices=["M1", "M5", "M15", "M30", "H1", "H4", "D1"], help="Target timeframe (default: M5)")
    parser.add_argument("--start", type=str, default="2010-06-01", help="Start date in YYYY-MM-DD format (default: 2010-06-01)")
    parser.add_argument("--end", type=str, default="auto", help="End date in YYYY-MM-DD or 'auto' for last completed candle (default: auto)")
    parser.add_argument("--symbols", type=str, default="all", help="Comma-separated list of symbols or 'all' (default: all)")
    parser.add_argument("--filter", type=str, default="all", choices=["forex", "metals", "indices", "crypto", "all"], help="Symbol category filter if --symbols is 'all' (default: all)")
    parser.add_argument("--format", type=str, default="parquet", choices=["parquet", "csv", "both"], help="Output format (default: parquet)")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent downloader threads (default: 4)")
    parser.add_argument("--chunk-days", type=int, default=180, help="Chunk size in days (default: 180)")
    parser.add_argument("--output-dir", type=str, default="HistoricalData", help="Target output directory (default: HistoricalData)")

    args = parser.parse_args()

    # Parse dates
    try:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception as e:
        logger.error(f"Invalid start date format: {args.start}. Use YYYY-MM-DD.")
        sys.exit(1)

    if args.end.lower() == "auto":
        end_dt = datetime.now(timezone.utc)
    else:
        try:
            end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception as e:
            logger.error(f"Invalid end date format: {args.end}. Use YYYY-MM-DD.")
            sys.exit(1)

    # Instantiate provider
    # Default is MT5, fallback to Mock if mt5 library not found (or for quick non-Windows test)
    if mt5 is None:
        logger.warning("MT5 library not found on this system. Falling back to MockDataProvider for demonstration/testing.")
        provider = MockDataProvider()
    else:
        provider = MT5DataProvider()

    logger.info("Initializing provider...")
    if not provider.connect():
        logger.error("Could not connect to data provider. Exiting.")
        sys.exit(1)

    try:
        # Resolve symbols list
        if args.symbols.lower() == "all":
            symbols = provider.get_visible_symbols(filter_type=args.filter)
            logger.info(f"Discovered {len(symbols)} symbols matching filter '{args.filter}'")
        else:
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

        if not symbols:
            logger.error("No symbols resolved. Exiting.")
            provider.disconnect()
            sys.exit(0)

        # Run Collector
        collector = HistoricalDataCollector(
            provider=provider,
            output_dir=args.output_dir,
            chunk_days=args.chunk_days,
            num_workers=args.workers
        )

        collector.download_multiple_symbols(symbols, args.timeframe, start_dt, end_dt)

    finally:
        provider.disconnect()


if __name__ == "__main__":
    main()
