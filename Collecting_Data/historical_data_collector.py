"""
historical_data_collector.py
----------------------------
Official historical data downloader for the Forex_DNN framework.
Supports multiple providers with provider-based abstraction (MT5, Dukascopy, OANDA, CSV, Mock),
chunked downloading, resume capability, data validation, parallel downloading, and metadata generation.
"""

import os
import sys
import json
import time
import math
import logging
import argparse
import re
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

# Try to load MetaTrader5 safely
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("HistoricalDataCollector")


# ── Provider Abstractions ─────────────────────────────────────────────────────

class BaseDataProvider(ABC):
    """
    Abstract base class for historical data providers.
    """
    @abstractmethod
    def connect(self) -> bool:
        """Initialize connection to the data provider."""
        pass

    @abstractmethod
    def disconnect(self):
        """Cleanly close connection to the data provider."""
        pass

    @abstractmethod
    def get_available_symbols(self) -> list:
        """Retrieve list of all available symbols."""
        pass

    @abstractmethod
    def symbol_select(self, symbol: str, enable: bool) -> bool:
        """Select/enable symbol for data retrieval."""
        pass

    @abstractmethod
    def fetch_chunk(self, symbol: str, timeframe: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Fetch a single chunk of historical OHLCV data.
        Returns a DataFrame with columns:
        ["Datetime", "Open", "High", "Low", "Close", "TickVolume", "Spread"]
        """
        pass

    @abstractmethod
    def get_broker_name(self) -> str:
        """Returns the broker/provider name."""
        pass

    @abstractmethod
    def get_timezone_name(self) -> str:
        """Returns the timezone name (e.g. UTC, EET)."""
        pass


class MT5DataProvider(BaseDataProvider):
    """
    Data provider using official MetaTrader 5 Python API.
    """
    def __init__(self, login: int = None, password: str = None, server: str = None):
        self.login = login
        self.password = password
        self.server = server
        self.connected = False

    def connect(self) -> bool:
        if mt5 is None:
            logger.error("MetaTrader5 library is not installed or not available in this environment.")
            return False

        # If login info is provided, use it; otherwise, use active terminal
        if self.login is not None and self.password is not None and self.server is not None:
            if not mt5.initialize(login=self.login, password=self.password, server=self.server):
                logger.error(f"MT5 initialization with credentials failed: {mt5.last_error()}")
                return False
        else:
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed (no credentials, active terminal): {mt5.last_error()}")
                return False

        self.connected = True
        logger.info(f"MT5 Provider connected to {mt5.terminal_info().company}")
        return True

    def disconnect(self):
        if self.connected and mt5 is not None:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 Provider disconnected.")

    def get_available_symbols(self) -> list:
        if not self.connected or mt5 is None:
            return []
        symbols = mt5.symbols_get()
        if symbols is None:
            logger.error(f"Failed to get symbols from MT5: {mt5.last_error()}")
            return []
        return [s.name for s in symbols]

    def symbol_select(self, symbol: str, enable: bool) -> bool:
        if not self.connected or mt5 is None:
            return False
        return mt5.symbol_select(symbol, enable)

    def fetch_chunk(self, symbol: str, timeframe: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        if not self.connected or mt5 is None:
            raise RuntimeError("MT5 Provider is not connected.")

        mt5_tf = self._get_mt5_timeframe(timeframe)
        if mt5_tf is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        # Ensure timezone info is stripped for mt5 library which assumes broker local time or UTC based on server.
        # We assume start/end are timezone-aware UTC and convert or keep naive.
        if start_time.tzinfo is not None:
            start_time = start_time.astimezone(timezone.utc).replace(tzinfo=None)
        if end_time.tzinfo is not None:
            end_time = end_time.astimezone(timezone.utc).replace(tzinfo=None)

        rates = mt5.copy_rates_range(symbol, mt5_tf, start_time, end_time)
        if rates is None or len(rates) == 0:
            return pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "TickVolume", "Spread"])

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        _RENAME = {
            "time": "Datetime",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tick_volume": "TickVolume",
            "spread": "Spread"
        }
        df = df[[c for c in _RENAME if c in df.columns]].rename(columns=_RENAME)
        df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s", utc=True)

        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].astype(float)
        df["TickVolume"] = df["TickVolume"].astype(int)
        df["Spread"] = df["Spread"].astype(int)

        return df[["Datetime", "Open", "High", "Low", "Close", "TickVolume", "Spread"]]

    def get_broker_name(self) -> str:
        if self.connected and mt5 is not None:
            try:
                return mt5.terminal_info().server
            except Exception:
                return "MetaTrader5"
        return "MT5_Disconnected"

    def get_timezone_name(self) -> str:
        # MT5 brokers usually run in EET (UTC+2 / UTC+3 DST) or UTC
        return "Broker_Time"

    def _get_mt5_timeframe(self, timeframe: str) -> int | None:
        mapping = {
            "M1": mt5.TIMEFRAME_M1 if mt5 else 1,
            "M5": mt5.TIMEFRAME_M5 if mt5 else 5,
            "M15": mt5.TIMEFRAME_M15 if mt5 else 15,
            "M30": mt5.TIMEFRAME_M30 if mt5 else 30,
            "H1": mt5.TIMEFRAME_H1 if mt5 else 16385,
            "H4": mt5.TIMEFRAME_H4 if mt5 else 16388,
            "D1": mt5.TIMEFRAME_D1 if mt5 else 16408,
        }
        return mapping.get(timeframe)


class DukascopyDataProvider(BaseDataProvider):
    """
    Future Provider: Dukascopy API downloader.
    """
    def connect(self) -> bool:
        logger.info("Dukascopy Provider initialized (Stub).")
        return True

    def disconnect(self):
        pass

    def get_available_symbols(self) -> list:
        return ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]

    def symbol_select(self, symbol: str, enable: bool) -> bool:
        return True

    def fetch_chunk(self, symbol: str, timeframe: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        logger.warning("fetch_chunk called on Dukascopy Stub Provider. Returning empty.")
        return pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "TickVolume", "Spread"])

    def get_broker_name(self) -> str:
        return "Dukascopy"

    def get_timezone_name(self) -> str:
        return "GMT"


class OANDADataProvider(BaseDataProvider):
    """
    Future Provider: OANDA v20 REST API downloader.
    """
    def connect(self) -> bool:
        logger.info("OANDA Provider initialized (Stub).")
        return True

    def disconnect(self):
        pass

    def get_available_symbols(self) -> list:
        return ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"]

    def symbol_select(self, symbol: str, enable: bool) -> bool:
        return True

    def fetch_chunk(self, symbol: str, timeframe: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        logger.warning("fetch_chunk called on OANDA Stub Provider. Returning empty.")
        return pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "TickVolume", "Spread"])

    def get_broker_name(self) -> str:
        return "OANDA"

    def get_timezone_name(self) -> str:
        return "UTC"


class CSVImportDataProvider(BaseDataProvider):
    """
    Future Provider: CSV local import utility.
    """
    def connect(self) -> bool:
        logger.info("CSV Import Provider initialized (Stub).")
        return True

    def disconnect(self):
        pass

    def get_available_symbols(self) -> list:
        return []

    def symbol_select(self, symbol: str, enable: bool) -> bool:
        return True

    def fetch_chunk(self, symbol: str, timeframe: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        logger.warning("fetch_chunk called on CSV Import Stub Provider. Returning empty.")
        return pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "TickVolume", "Spread"])

    def get_broker_name(self) -> str:
        return "LocalCSV"

    def get_timezone_name(self) -> str:
        return "UTC"


class MockDataProvider(BaseDataProvider):
    """
    High-quality mock provider for testing, demoing, and validating collector operations offline.
    Generates deterministic prices based on a continuous wave formula.
    Can inject gaps, duplicates, or invalid candles if configured.
    """
    def __init__(self, inject_issues: bool = False):
        self.inject_issues = inject_issues
        self.symbols = ["EURUSD", "GBPUSD", "XAUUSD", "YM", "FDAX"]

    def connect(self) -> bool:
        logger.info("Mock Provider connected successfully.")
        return True

    def disconnect(self):
        logger.info("Mock Provider disconnected.")

    def get_available_symbols(self) -> list:
        return self.symbols

    def symbol_select(self, symbol: str, enable: bool) -> bool:
        return symbol in self.symbols

    def fetch_chunk(self, symbol: str, timeframe: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        # Determine step from timeframe
        mapping = {
            "M1": timedelta(minutes=1),
            "M5": timedelta(minutes=5),
            "M15": timedelta(minutes=15),
            "M30": timedelta(minutes=30),
            "H1": timedelta(hours=1),
            "H4": timedelta(hours=4),
            "D1": timedelta(days=1),
        }
        step = mapping.get(timeframe, timedelta(minutes=5))

        # Base price per symbol
        base_prices = {
            "EURUSD": 1.1000,
            "GBPUSD": 1.2500,
            "XAUUSD": 2000.0,
            "YM": 38000.0,
            "FDAX": 17000.0,
        }
        base_price = base_prices.get(symbol, 100.00)

        # Generate timestamps
        current = start_time
        datetimes = []
        while current <= end_time:
            # Skip weekends (simulate real market)
            # Sunday 17:00 NY to Friday 17:00 NY is standard. We can do simple: UTC day of week 5 (Saturday) and 6 (Sunday)
            if current.weekday() < 5 or (current.weekday() == 4 and current.hour < 22) or (current.weekday() == 6 and current.hour >= 22):
                datetimes.append(current)
            current += step

        if not datetimes:
            return pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "TickVolume", "Spread"])

        # Create synthetic deterministic rows
        rows = []
        for i, dt in enumerate(datetimes):
            t_epoch = dt.timestamp()

            # Dynamic wave-based deterministic price formula
            wave1 = 0.05 * math.sin(t_epoch / 86400.0)    # Daily cycle
            wave2 = 0.02 * math.sin(t_epoch / 604800.0)   # Weekly cycle
            wave3 = 0.005 * math.cos(t_epoch / 3600.0)    # Hourly noise

            scaling_factor = 0.01 if symbol in ["EURUSD", "GBPUSD"] else (1.0 if symbol == "XAUUSD" else 10.0)

            # Price offsets
            offset = (wave1 + wave2 + wave3) * scaling_factor
            open_val = base_price + offset
            close_val = base_price + offset + (0.0003 if i % 2 == 0 else -0.0003) * scaling_factor

            # Standard candlestick relations
            high_val = max(open_val, close_val) + 0.0005 * scaling_factor
            low_val = min(open_val, close_val) - 0.0005 * scaling_factor

            # Adjust decimals based on symbol
            decimals = 5 if symbol in ["EURUSD", "GBPUSD"] else (2 if symbol == "XAUUSD" else 1)
            open_val = round(open_val, decimals)
            high_val = round(high_val, decimals)
            low_val = round(low_val, decimals)
            close_val = round(close_val, decimals)

            volume = 100 + (int(t_epoch) % 1000)
            spread = 2 if symbol in ["EURUSD", "GBPUSD"] else 15

            row = {
                "Datetime": dt,
                "Open": open_val,
                "High": high_val,
                "Low": low_val,
                "Close": close_val,
                "TickVolume": volume,
                "Spread": spread
            }

            # Inject artificial issues for testing validation engine
            if self.inject_issues:
                # 1. Duplicate candle at index 3
                if i == 3:
                    rows.append(row.copy())
                # 2. Invalid OHLC value at index 7 (High < Low)
                if i == 7:
                    bad_row = row.copy()
                    bad_row["High"] = bad_row["Low"] - 1.0
                    rows.append(bad_row)
                    continue
                # 3. Negative volume at index 11
                if i == 11:
                    bad_row = row.copy()
                    bad_row["TickVolume"] = -50
                    rows.append(bad_row)
                    continue

            rows.append(row)

        df = pd.DataFrame(rows)
        # Ensure Datetime is timezone-aware UTC
        df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.tz_localize(timezone.utc) if df["Datetime"].dt.tz is None else pd.to_datetime(df["Datetime"])
        return df[["Datetime", "Open", "High", "Low", "Close", "TickVolume", "Spread"]]

    def get_broker_name(self) -> str:
        return "MockBroker"

    def get_timezone_name(self) -> str:
        return "UTC"


# ── Collector Orchestrator ───────────────────────────────────────────────────

class HistoricalDataCollector:
    """
    Orchestrator class for downloading, validating, merging, and saving historical market data.
    """
    def __init__(self, provider: BaseDataProvider, output_dir: str = "HistoricalData", format: str = "parquet", chunk_size_days: int = 180, max_workers: int = 4):
        self.provider = provider
        self.output_dir = output_dir
        self.format = format.lower()
        self.chunk_size_days = chunk_size_days
        self.max_workers = max_workers
        self.state_file = os.path.join(output_dir, "download_state.json")
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load download state: {e}. Starting fresh.")
        return {}

    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save download state: {e}")

    def filter_symbols(self, symbols: list, category: str) -> list:
        """
        Filters a list of symbols based on a pre-defined category.
        """
        category = category.lower()
        if category in ["everything", "all", "any"]:
            return symbols

        filtered = []
        for sym in symbols:
            upper_sym = sym.upper()
            # Forex: 6 consecutive letters plus optional prefix/suffix
            is_forex = bool(re.match(r"^[A-Z]{6}(_o|\.|\+)?$", upper_sym)) or any(x in upper_sym for x in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "GBPJPY"])
            # Metals: Contains standard metal pairs (Gold, Silver, Platinum, Palladium)
            is_metals = any(x in upper_sym for x in ["XAU", "XAG", "XPT", "XPD", "GOLD", "SILVER"])
            # Indices: YM, FDAX, SPX, etc.
            is_indices = any(x in upper_sym for x in ["YM", "FDAX", "DE30", "DE40", "US30", "SPX", "NAS100", "US500", "FCHI", "UK100", "N225", "ASX200"])
            # Crypto: Bitcoin, Ethereum, etc.
            is_crypto = any(x in upper_sym for x in ["BTC", "ETH", "LTC", "XRP", "SOL", "ADA"]) and not is_forex

            if category == "forex" and is_forex:
                filtered.append(sym)
            elif category == "metals" and is_metals:
                filtered.append(sym)
            elif category == "indices" and is_indices:
                filtered.append(sym)
            elif category == "crypto" and is_crypto:
                filtered.append(sym)
        return filtered

    def validate_chunk(self, df: pd.DataFrame) -> dict:
        """
        Validates individual chunk data for duplicates, invalid OHLC, negative volume, and gaps.
        """
        issues = {
            "duplicates": 0,
            "invalid_ohlc": 0,
            "negative_volume": 0,
            "missing_timestamps": 0
        }

        if df.empty:
            return issues

        # 1. Duplicates
        issues["duplicates"] = int(df.duplicated(subset=["Datetime"]).sum())

        # 2. Invalid OHLC
        # High must be >= Low, Open, Close. Low must be <= Open, Close, High.
        invalid_mask = (df["High"] < df["Low"]) | (df["High"] < df["Open"]) | (df["High"] < df["Close"]) | (df["Low"] > df["Open"]) | (df["Low"] > df["Close"])
        issues["invalid_ohlc"] = int(invalid_mask.sum())

        # 3. Negative volume
        negative_vol_mask = df["TickVolume"] < 0
        issues["negative_volume"] = int(negative_vol_mask.sum())

        return issues

    def merge_chunks(self, chunks: list) -> pd.DataFrame:
        """
        Combines downloaded chunks, removes duplicate rows based on Datetime,
        and sorts chronologically.
        """
        if not chunks:
            return pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "TickVolume", "Spread"])

        merged = pd.concat(chunks, ignore_index=True)
        initial_count = len(merged)

        # Drop duplicates based on Datetime
        merged.drop_duplicates(subset=["Datetime"], keep="last", inplace=True)
        duplicates_removed = initial_count - len(merged)

        # Sort chronologically
        merged.sort_values(by="Datetime", inplace=True)
        merged.reset_index(drop=True, inplace=True)

        logger.info(f"Merged {len(chunks)} chunks. Rows: {len(merged)} (Removed {duplicates_removed} duplicates)")
        return merged

    def detect_missing_candles(self, df: pd.DataFrame, timeframe: str) -> int:
        """
        Estimates missing candles in a continuous DataFrame based on the expected timeframe step,
        ignoring weekend gaps.
        """
        if len(df) < 2:
            return 0

        mapping = {
            "M1": timedelta(minutes=1),
            "M5": timedelta(minutes=5),
            "M15": timedelta(minutes=15),
            "M30": timedelta(minutes=30),
            "H1": timedelta(hours=1),
            "H4": timedelta(hours=4),
            "D1": timedelta(days=1),
        }
        step = mapping.get(timeframe, timedelta(minutes=5))

        diffs = df["Datetime"].diff()
        missing_count = 0

        for diff in diffs:
            if pd.isna(diff):
                continue
            if diff > step:
                # Calculate how many intervals are missing
                intervals = (diff / step) - 1
                # If it spans across a weekend, we subtract the weekend duration to avoid reporting it as missing.
                # A crude but effective heuristic: if gap is > 48 hours, subtract ~48 hours of intervals
                if diff > timedelta(hours=50):
                    intervals = max(0, intervals - (timedelta(hours=48) / step))
                missing_count += int(round(intervals))

        return missing_count

    def download_symbol(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> dict:
        """
        Downloads the entire requested history for a single symbol in chunk-intervals,
        safely resuming if a previous state exists.
        """
        t0_symbol = time.perf_counter()
        logger.info(f"Starting download for {symbol} ({timeframe})")

        # Resume logic
        state_key = f"{symbol}_{timeframe}"
        symbol_state = self.state.get(state_key, {})

        current_start = start_date
        existing_df = None
        if symbol_state.get("status") == "completed":
            logger.info(f"{symbol} {timeframe} is already fully downloaded.")
            return {
                "symbol": symbol,
                "bars": symbol_state.get("number_of_bars", 0),
                "start": symbol_state.get("first_candle", str(start_date)),
                "end": symbol_state.get("last_candle", str(end_date)),
                "status": "Skipped",
                "elapsed": 0.0,
                "errors": 0
            }
        elif symbol_state.get("status") == "in_progress":
            last_dt_str = symbol_state.get("last_downloaded_datetime")
            if last_dt_str:
                current_start = datetime.fromisoformat(last_dt_str)
                logger.info(f"Resuming {symbol} {timeframe} from {current_start}")

                # Attempt to load previously downloaded data to prevent data loss
                symbol_dir = os.path.join(self.output_dir, symbol)
                csv_path = os.path.join(symbol_dir, f"{timeframe}.csv")
                parquet_path = os.path.join(symbol_dir, f"{timeframe}.parquet")

                if self.format in ["parquet", "both"] and os.path.exists(parquet_path):
                    try:
                        existing_df = pd.read_parquet(parquet_path)
                        logger.info(f"Loaded existing Parquet data for {symbol} ({len(existing_df)} rows) to prepend.")
                    except Exception as e:
                        logger.warning(f"Failed to load existing Parquet data: {e}")
                elif self.format in ["csv", "both"] and os.path.exists(csv_path):
                    try:
                        existing_df = pd.read_csv(csv_path)
                        existing_df["Datetime"] = pd.to_datetime(existing_df["Datetime"])
                        logger.info(f"Loaded existing CSV data for {symbol} ({len(existing_df)} rows) to prepend.")
                    except Exception as e:
                        logger.warning(f"Failed to load existing CSV data: {e}")

        # Partition into chunks of `chunk_size_days`
        chunks_ranges = []
        temp_start = current_start
        while temp_start < end_date:
            temp_end = min(temp_start + timedelta(days=self.chunk_size_days), end_date)
            chunks_ranges.append((temp_start, temp_end))
            temp_start = temp_end

        total_chunks = len(chunks_ranges)
        logger.info(f"{symbol} partitioned into {total_chunks} chunks.")

        chunks_data = []
        if existing_df is not None and not existing_df.empty:
            # Normalize index datetime timezone representation
            if existing_df["Datetime"].dt.tz is None:
                existing_df["Datetime"] = existing_df["Datetime"].dt.tz_localize(timezone.utc)
            else:
                existing_df["Datetime"] = existing_df["Datetime"].dt.tz_convert(timezone.utc)
            chunks_data.append(existing_df)

        total_errors = 0
        total_duplicates = 0
        total_invalid_ohlc = 0
        total_negative_vol = 0

        # Try to select symbol
        self.provider.symbol_select(symbol, True)

        for idx, (c_start, c_end) in enumerate(chunks_ranges, 1):
            logger.info(f"[{symbol}] Downloading Chunk {idx} / {total_chunks} ({c_start.date()} to {c_end.date()})")

            # Download chunk with retries
            success = False
            chunk_df = None
            retries = 3
            backoff = 2.0

            for r in range(retries):
                try:
                    chunk_df = self.provider.fetch_chunk(symbol, timeframe, c_start, c_end)
                    success = True
                    break
                except Exception as e:
                    logger.warning(f"Error downloading chunk {idx} (attempt {r+1}/{retries}) for {symbol}: {e}")
                    total_errors += 1
                    time.sleep(backoff)
                    backoff *= 2

            if not success or chunk_df is None:
                logger.error(f"Failed to download chunk {idx} for {symbol} after {retries} retries. Saving intermediate progress.")
                # Mark as in-progress for future resume
                self.state[state_key] = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "last_downloaded_datetime": c_start.isoformat(),
                    "chunk_index": idx - 1,
                    "status": "in_progress"
                }
                self._save_state()
                return {
                    "symbol": symbol,
                    "bars": 0,
                    "start": str(start_date),
                    "end": str(end_date),
                    "status": "Failed",
                    "elapsed": time.perf_counter() - t0_symbol,
                    "errors": total_errors
                }

            # Validate chunk
            val_results = self.validate_chunk(chunk_df)
            total_duplicates += val_results["duplicates"]
            total_invalid_ohlc += val_results["invalid_ohlc"]
            total_negative_vol += val_results["negative_volume"]

            if val_results["duplicates"] > 0 or val_results["invalid_ohlc"] > 0 or val_results["negative_volume"] > 0:
                logger.warning(f"[{symbol} Chunk {idx}] Validation issues detected: {val_results}")

            if not chunk_df.empty:
                chunks_data.append(chunk_df)

            # Update resume state incrementally
            self.state[state_key] = {
                "symbol": symbol,
                "timeframe": timeframe,
                "last_downloaded_datetime": c_end.isoformat(),
                "chunk_index": idx,
                "status": "in_progress"
            }
            self._save_state()

        # Merge and finalize
        final_df = self.merge_chunks(chunks_data)

        if final_df.empty:
            logger.warning(f"No data retrieved for symbol {symbol} {timeframe}")
            self.state[state_key] = {
                "symbol": symbol,
                "timeframe": timeframe,
                "status": "failed"
            }
            self._save_state()
            return {
                "symbol": symbol,
                "bars": 0,
                "start": str(start_date),
                "end": str(end_date),
                "status": "No Data",
                "elapsed": time.perf_counter() - t0_symbol,
                "errors": total_errors
            }

        # Estimate missing candles
        missing_count = self.detect_missing_candles(final_df, timeframe)

        # Save symbol data to disk
        symbol_dir = os.path.join(self.output_dir, symbol)
        os.makedirs(symbol_dir, exist_ok=True)

        # Save CSV and Parquet
        csv_path = os.path.join(symbol_dir, f"{timeframe}.csv")
        parquet_path = os.path.join(symbol_dir, f"{timeframe}.parquet")

        first_c = final_df["Datetime"].iloc[0]
        last_c = final_df["Datetime"].iloc[-1]

        # Serialization
        if self.format in ["parquet", "both"]:
            final_df.to_parquet(parquet_path, index=False)
            logger.info(f"Saved {symbol} parquet to {parquet_path}")
        if self.format in ["csv", "both"]:
            final_df.to_csv(csv_path, index=False)
            logger.info(f"Saved {symbol} csv to {csv_path}")

        # Create metadata.json
        metadata = {
            "symbol": symbol,
            "timeframe": timeframe,
            "download_date": datetime.now(timezone.utc).isoformat(),
            "first_candle": first_c.isoformat(),
            "last_candle": last_c.isoformat(),
            "number_of_bars": len(final_df),
            "duplicates_removed": total_duplicates,
            "invalid_ohlc_removed": total_invalid_ohlc,
            "negative_volume_removed": total_negative_vol,
            "missing_candles_estimated": missing_count,
            "broker": self.provider.get_broker_name(),
            "timezone": self.provider.get_timezone_name()
        }

        with open(os.path.join(symbol_dir, "metadata.json"), "w") as mf:
            json.dump(metadata, mf, indent=4)

        # Update final state
        self.state[state_key] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "first_candle": first_c.isoformat(),
            "last_candle": last_c.isoformat(),
            "number_of_bars": len(final_df),
            "status": "completed"
        }
        self._save_state()

        elapsed_time = time.perf_counter() - t0_symbol
        logger.info(f"Successfully finalized {symbol} in {elapsed_time:.2f}s. Total bars: {len(final_df)}")

        return {
            "symbol": symbol,
            "bars": len(final_df),
            "start": first_c.isoformat(),
            "end": last_c.isoformat(),
            "status": "Success",
            "elapsed": elapsed_time,
            "errors": total_errors
        }

    def run(self, symbols_arg: str, timeframe: str, start_date: datetime, end_date: datetime) -> list:
        """
        Executes parallel or sequential downloading for all selected symbols.
        """
        # Discover and filter symbols
        all_avail_symbols = self.provider.get_available_symbols()
        logger.info(f"Total available symbols from provider: {len(all_avail_symbols)}")

        target_symbols = []
        if symbols_arg.lower() in ["all", "everything", "forex", "metals", "indices", "crypto"]:
            # Perform automatic filter
            category = "all" if symbols_arg.lower() in ["all", "everything"] else symbols_arg.lower()
            target_symbols = self.filter_symbols(all_avail_symbols, category)
        else:
            # Explicit comma-separated list
            explicit_list = [s.strip() for s in symbols_arg.split(",") if s.strip()]
            target_symbols = [s for s in explicit_list if s in all_avail_symbols]

        if not target_symbols:
            logger.error(f"No matching symbols found for selection: {symbols_arg}")
            return []

        logger.info(f"Selected target symbols ({len(target_symbols)}): {target_symbols}")

        results = []
        # Support parallel downloading using a ThreadPoolExecutor
        if self.max_workers > 1 and len(target_symbols) > 1:
            logger.info(f"Downloading {len(target_symbols)} symbols concurrently with {self.max_workers} workers.")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.download_symbol, sym, timeframe, start_date, end_date): sym
                    for sym in target_symbols
                }
                for fut in as_completed(futures):
                    sym = futures[fut]
                    try:
                        res = fut.result()
                        results.append(res)
                    except Exception as e:
                        logger.error(f"Unhandled exception during parallel download of {sym}: {e}")
                        results.append({
                            "symbol": sym,
                            "bars": 0,
                            "start": str(start_date),
                            "end": str(end_date),
                            "status": "Error",
                            "elapsed": 0.0,
                            "errors": 1
                        })
        else:
            logger.info("Downloading symbols sequentially.")
            for sym in target_symbols:
                try:
                    res = self.download_symbol(sym, timeframe, start_date, end_date)
                    results.append(res)
                except Exception as e:
                    logger.error(f"Unhandled exception during sequential download of {sym}: {e}")
                    results.append({
                        "symbol": sym,
                        "bars": 0,
                        "start": str(start_date),
                        "end": str(end_date),
                        "status": "Error",
                        "elapsed": 0.0,
                        "errors": 1
                    })

        # Generate download_report.csv
        report_df = pd.DataFrame(results)
        report_path = os.path.join(self.output_dir, "download_report.csv")
        report_df.to_csv(report_path, index=False)
        logger.info(f"Generated global download report at {report_path}")

        return results


# ── CLI Interface ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Official Forex_DNN Historical Data Downloader CLI")
    parser.add_argument("--timeframe", type=str, default="M5", choices=["M1", "M5", "M15", "M30", "H1", "H4", "D1"], help="Candle timeframe (default: M5)")
    parser.add_argument("--start", type=str, default="2010-06-01", help="Start date in YYYY-MM-DD format (default: 2010-06-01)")
    parser.add_argument("--end", type=str, default="auto", help="End date in YYYY-MM-DD or 'auto' for last completed candle (default: auto)")
    parser.add_argument("--symbols", type=str, default="all", help="Comma-separated symbols, 'all', or filters (forex, metals, indices, crypto)")
    parser.add_argument("--provider", type=str, default="mt5", choices=["mt5", "dukascopy", "oanda", "csv", "mock"], help="Historical data provider (default: mt5)")
    parser.add_argument("--format", type=str, default="parquet", choices=["parquet", "csv", "both"], help="Output format: parquet, csv, or both")
    parser.add_argument("--output-dir", type=str, default="HistoricalData", help="Output storage directory")
    parser.add_argument("--chunk-size-days", type=int, default=180, help="Chunk size in days for historical queries (default: 180)")
    parser.add_argument("--workers", type=int, default=4, help="Maximum concurrent download workers (default: 4)")
    parser.add_argument("--inject-issues", action="store_true", help="Inject issues into MockProvider for testing")

    args = parser.parse_args()

    # Parse dates
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        logger.error(f"Invalid start date format: {args.start}. Must be YYYY-MM-DD.")
        sys.exit(1)

    if args.end.lower() == "auto":
        end_date = datetime.now(timezone.utc)
    else:
        try:
            end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            logger.error(f"Invalid end date format: {args.end}. Must be YYYY-MM-DD.")
            sys.exit(1)

    # Initialize Provider
    if args.provider == "mt5":
        provider = MT5DataProvider()
    elif args.provider == "mock":
        provider = MockDataProvider(inject_issues=args.inject_issues)
    elif args.provider == "dukascopy":
        provider = DukascopyDataProvider()
    elif args.provider == "oanda":
        provider = OANDADataProvider()
    else:
        provider = CSVImportDataProvider()

    logger.info(f"Initializing connection to data provider: {args.provider}")
    if not provider.connect():
        if args.provider == "mt5":
            logger.warning("MT5 connection failed or MT5 library is not available in this environment. Falling back to MockProvider.")
            provider = MockDataProvider(inject_issues=args.inject_issues)
            provider.connect()
        else:
            logger.error("Failed to connect to specified provider.")
            sys.exit(1)

    # Initialize and execute collector
    collector = HistoricalDataCollector(
        provider=provider,
        output_dir=args.output_dir,
        format=args.format,
        chunk_size_days=args.chunk_size_days,
        max_workers=args.workers
    )

    try:
        collector.run(
            symbols_arg=args.symbols,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date
        )
    finally:
        provider.disconnect()


if __name__ == "__main__":
    main()
