"""
data_feed.py
------------
Unified MT5 data retrieval module for both live trading and backtesting.
Note: get_available_symbols() should be called first to verify symbol names on a new broker.

Usage:
    feed = MT5DataFeed(login=123, password="x", server="Broker-Live")

    # Live mode - retrieves latest N candles, monitors health
    df = feed.get_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M1, live=True)

    # Backtest mode - retrieves maximum available or date-bounded history
    df = feed.get_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M1, live=False)
    df = feed.get_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M1, live=False,
                       date_from=datetime(2022,1,1), date_to=datetime(2023,1,1))

Schema returned (always identical regardless of mode):
    Datetime, Open, High, Low, Close, TickVolume, Spread

Spread is in points (integer) as provided by MT5 rates struct.
All other indicators are calculated downstream — not here.
"""

import time
import logging
from datetime import datetime, timezone, timedelta
from enum import Enum

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

try:
    from auth import load_credentials
except ImportError:
    from Collecting_Data.auth import load_credentials

import os

# ── Logging setup ─────────────────────────────────────────────────────────────
logger = logging.getLogger("DataFeed")


# ── Health states ──────────────────────────────────────────────────────────────
class FeedHealth(Enum):
    HEALTHY     = "HEALTHY"      # data fresh, latency acceptable
    DEGRADED    = "DEGRADED"     # connected but slow or stale — do not open new positions
    DISCONNECTED = "DISCONNECTED" # reconnection required


# ── Thresholds (tune per broker/VPS environment) ───────────────────────────────
LATENCY_WARN_MS      = 50    # API exec time: log warning above this
LATENCY_PAUSE_MS     = 200   # API exec time: set DEGRADED above this
DATA_STALE_S         = 10    # last tick age in seconds: set DEGRADED above this
RECONNECT_MAX_TRIES  = 5
RECONNECT_BACKOFF_S  = 5     # doubles each failed attempt


# ── Column schema ──────────────────────────────────────────────────────────────
SCHEMA = ["Datetime", "Open", "High", "Low", "Close", "TickVolume", "Spread"]

# MT5 rates struct → our schema
_RENAME = {
    "time":        "Datetime",
    "open":        "Open",
    "high":        "High",
    "low":         "Low",
    "close":       "Close",
    "tick_volume": "TickVolume",
    "spread":      "Spread",
}


class MT5DataFeed:
    """
    Single class for all MT5 data needs.
    Handles connection lifecycle, health monitoring, and data retrieval.
    live=True  → live feed with health checks
    live=False → historical/backtest fetch, no health monitoring
    """

    def __init__(self, login: int = None, password: str = None, server: str = None):
        creds = load_credentials(path="credentials.json")
        self.login    = creds["login"]
        self.password =  creds["password"]
        self.server   =  creds["server"]

        self._health  = FeedHealth.DISCONNECTED
        self._connected = False

    # ── Connection management ──────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Purpose:
            Initializes the MetaTrader 5 connection using the stored credentials.
            Ensures the terminal is ready for data retrieval and trading.

        Returns:
            bool: True if connection is established successfully, False otherwise.

        Side Effects:
            Sets internal _health to HEALTHY on success, DISCONNECTED on failure.
            Sets _connected to True on success.

        Example:
            >>> feed = MT5DataFeed()
            >>> if feed.connect():
            ...     print("Ready to trade")
        """
        if not mt5.initialize(login=self.login, password=self.password, server=self.server):
            logger.error(f"MT5 init failed: {mt5.last_error()}")
            self._health = FeedHealth.DISCONNECTED
            self._connected = False
            return False

        logger.info(f"MT5 connected — {mt5.terminal_info()} / {mt5.version()}")
        self._connected = True
        self._health = FeedHealth.HEALTHY
        return True

    def disconnect(self):
        """
        Purpose:
            Shuts down the MetaTrader 5 terminal connection cleanly.
            Should be called during application shutdown to release resources.

        Side Effects:
            Sets _connected to False and _health to DISCONNECTED.
        """
        mt5.shutdown()
        self._connected = False
        self._health = FeedHealth.DISCONNECTED
        logger.info("MT5 disconnected.")

    def reconnect(self) -> bool:
        """
        Purpose:
            Attempts to re-establish the MT5 connection using exponential backoff.
            Used to recover from transient network or terminal issues.

        Returns:
            bool: True if reconnection succeeds within the maximum number of attempts.

        Notes:
            - Max attempts: 5.
            - Backoff: Starts at 5s, doubles each attempt, capped at 60s.
        """
        self.disconnect()
        wait = RECONNECT_BACKOFF_S

        for attempt in range(1, RECONNECT_MAX_TRIES + 1):
            logger.warning(f"Reconnection attempt {attempt}/{RECONNECT_MAX_TRIES} — waiting {wait}s")
            time.sleep(wait)

            if self.connect():
                logger.info("Reconnection successful.")
                return True

            wait = min(wait * 2, 60)  # cap at 60s

        logger.error("All reconnection attempts failed. Feed is DISCONNECTED.")
        self._health = FeedHealth.DISCONNECTED
        return False

    # ── Utility methods ────────────────────────────────────────────────────────

    def get_available_symbols(self) -> list[str]:
        """
        Purpose:
            Retrieves the list of all symbols available on the current broker account.
            Useful for verifying symbol naming conventions (e.g., '_o' suffix).

        Returns:
            list[str]: Sorted list of symbol names.

        Exceptions:
            Logs an error and returns an empty list if MT5 query fails.
        """
        symbols = mt5.symbols_get()
        if symbols is None:
            logger.error(f"Failed to get symbols: {mt5.last_error()}")
            return []
        
        symbol_names = sorted([s.name for s in symbols])
        logger.info(f"Retrieved {len(symbol_names)} symbols from broker.")
        return symbol_names

    def get_symbol_info(self, symbol: str) -> dict | None:
        """
        Purpose:
            Retrieves detailed specifications for a specific symbol.
            Used for calculating lot sizes and placing correct orders.

        Arguments:
            symbol (str): Target symbol name.

        Returns:
            dict | None: Dictionary with keys (digits, point, volume_min, volume_step,
                        volume_max, trade_contract_size) or None if symbol not found.
        """
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Symbol {symbol} not found.")
            return None
        
        return {
            "digits": info.digits,
            "point": info.point,
            "volume_min": info.volume_min,
            "volume_step": info.volume_step,
            "volume_max": info.volume_max,
            "trade_contract_size": info.trade_contract_size
        }

    def get_history_depth(self, symbol: str, timeframe: int) -> datetime | None:
        """
        Purpose:
            Quickly identifies the earliest available historical bar for a symbol.

        Arguments:
            symbol (str): Target symbol.
            timeframe (int): MT5 timeframe constant.

        Returns:
            datetime | None: UTC datetime of the oldest bar or None on failure.
        """
        # Use a large enough count to find the broker's typical history depth
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 99999)
        if rates is None or len(rates) == 0:
            logger.error(f"Could not retrieve history depth for {symbol}: {mt5.last_error()}")
            return None
        
        earliest_ts = rates[0]['time']
        earliest_dt = datetime.fromtimestamp(earliest_ts, tz=timezone.utc)
        logger.info(f"History depth for {symbol} TF={timeframe}: {earliest_dt}")
        return earliest_dt

    # ── Health monitoring ──────────────────────────────────────────────────────

    @property
    def health(self) -> FeedHealth:
        return self._health

    def check_health(self, symbol: str, num_samples: int = 5) -> FeedHealth:
        """
        Purpose:
            Measures MetaTrader 5 API execution time and market data freshness.
            Ensures that the trading strategy is not acting on stale or delayed data.

        Arguments:
            symbol (str): The symbol to check (e.g., "EURUSD_o").
            num_samples (int): Number of tick requests to average. Defaults to 5.

        Returns:
            FeedHealth: HEALTHY, DEGRADED, or DISCONNECTED.

        Notes:
            - LATENCY_WARN_MS (50ms): Threshold for warning logs.
            - LATENCY_PAUSE_MS (200ms): Threshold for marking feed DEGRADED.
            - DATA_STALE_S (10s): Threshold for tick age before marking DEGRADED.
        """
        if not self._connected:
            self._health = FeedHealth.DISCONNECTED
            return self._health

        exec_times = []
        tick_ages  = []

        for _ in range(num_samples):
            t0   = time.perf_counter()
            tick = mt5.symbol_info_tick(symbol)
            exec_ms = (time.perf_counter() - t0) * 1000

            if tick is None:
                logger.warning(f"symbol_info_tick returned None for {symbol}")
                self._health = FeedHealth.DISCONNECTED
                return self._health

            tick_age_s = time.time() - tick.time
            exec_times.append(exec_ms)
            tick_ages.append(tick_age_s)

        avg_exec_ms  = np.mean(exec_times)
        avg_age_s    = np.mean(tick_ages)

        logger.debug(f"Health check — avg exec: {avg_exec_ms:.1f}ms | avg tick age: {avg_age_s:.1f}s")

        if avg_exec_ms > LATENCY_WARN_MS:
            logger.warning(f"API latency elevated: {avg_exec_ms:.1f}ms")

        if avg_exec_ms > LATENCY_PAUSE_MS or avg_age_s > DATA_STALE_S:
            logger.warning(
                f"Feed DEGRADED — exec: {avg_exec_ms:.1f}ms, tick age: {avg_age_s:.1f}s"
            )
            self._health = FeedHealth.DEGRADED
        else:
            self._health = FeedHealth.HEALTHY

        return self._health

    def wait_for_healthy(self, symbol: str, timeout_s: int = 120) -> bool:
        """
        Purpose:
            Blocks execution until the data feed returns to a HEALTHY state or a timeout occurs.
            Crucial for ensuring the strategy doesn't execute during market gaps or high latency.

        Arguments:
            symbol (str): Symbol to monitor for health.
            timeout_s (int): Maximum time to wait in seconds. Defaults to 120.

        Returns:
            bool: True if the feed is healthy within the timeout, False otherwise.

        Side Effects:
            Will attempt automatic reconnection if the state is DISCONNECTED.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            state = self.check_health(symbol)
            if state == FeedHealth.HEALTHY:
                return True
            if state == FeedHealth.DISCONNECTED:
                if not self.reconnect():
                    return False
            logger.info(f"Feed {state.value} — retrying in 5s")
            time.sleep(5)

        logger.error(f"Feed did not recover within {timeout_s}s timeout.")
        return False

    # ── Data retrieval ─────────────────────────────────────────────────────────

    def get_ohlcv(self, symbol: str, timeframe_str: str, count: int = 1000) -> pd.DataFrame | None:
        """
        Purpose:
            High-level wrapper to retrieve the latest OHLCV candles using string timeframes.

        Arguments:
            symbol (str): The symbol to fetch (e.g., "EURUSD_o").
            timeframe_str (str): Target timeframe ("M1", "M5", "M15", "H1", etc.).
            count (int): Number of candles to retrieve. Defaults to 1000.

        Returns:
            pd.DataFrame | None: DataFrame with standard schema or None on failure.

        Example:
            >>> df = feed.get_ohlcv("EURUSD_o", "M5", count=500)
            >>> print(df.iloc[-1]['Close'])
        """
        mapping = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        tf = mapping.get(timeframe_str)
        if tf is None:
            logger.error(f"Invalid timeframe string: {timeframe_str}")
            return None

        return self.get_data(symbol, tf, live=True, count=count)

    def get_data(
        self,
        symbol:    str,
        timeframe: int,
        live:      bool = True,
        count:     int  = 500,
        date_from: datetime = None,
        date_to:   datetime = None,
        probe_max: bool = False,
    ) -> pd.DataFrame | None:
        """
        Purpose:
            The primary, unified data retrieval method for the framework.
            Handles both live polling and historical range fetching.

        Arguments:
            symbol (str): Target symbol.
            timeframe (int): MT5 timeframe constant (e.g., mt5.TIMEFRAME_M5).
            live (bool): If True, performs health checks and fetches latest data.
            count (int): Number of bars to fetch (for relative requests).
            date_from (datetime): Start of range (for historical requests).
            date_to (datetime): End of range (for historical requests).
            probe_max (bool): If True, binary searches for max retrievable bars.

        Returns:
            pd.DataFrame | None: Standard schema DataFrame or None if unhealthy/failed.

        Schema:
            Datetime (UTC), Open, High, Low, Close, TickVolume, Spread.

        Notes:
            In live mode, if the feed is DEGRADED or DISCONNECTED, this returns None.
            The strategy loop is expected to handle this by waiting.
        """
        if not self._connected:
            logger.error("Not connected. Call connect() first.")
            return None

        if not mt5.symbol_select(symbol, True):
            logger.error(f"Cannot select symbol {symbol}: {mt5.last_error()}")
            return None

        # ── Live mode ──────────────────────────────────────────────────────────
        if live:
            state = self.check_health(symbol)
            if state != FeedHealth.HEALTHY:
                logger.warning(f"Live fetch blocked — feed is {state.value}")
                return None

            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None:
                logger.error(f"copy_rates_from_pos failed: {mt5.last_error()}")
                return None

            return self._to_dataframe(rates)

        # ── Backtest mode ──────────────────────────────────────────────────────
        if probe_max:
            return self._probe_max_bars(symbol, timeframe)

        if date_from is not None:
            return self._fetch_date_range(symbol, timeframe, date_from, date_to)

        # Default: fetch `count` bars from current position
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            logger.error(f"Historical fetch failed: {mt5.last_error()}")
            return None

        return self._to_dataframe(rates)

    # ── Backtest helpers ───────────────────────────────────────────────────────

    def _fetch_date_range(
        self,
        symbol:    str,
        timeframe: int,
        date_from: datetime,
        date_to:   datetime = None,
    ) -> pd.DataFrame | None:
        """
        Fetch bars between date_from and date_to.
        If date_to is None, fetches from date_from to now.
        Handles naive datetimes by assuming UTC.
        """
        if date_from.tzinfo is None:
            date_from = date_from.replace(tzinfo=timezone.utc)

        if date_to is None:
            date_to = datetime.now(timezone.utc)
        elif date_to.tzinfo is None:
            date_to = date_to.replace(tzinfo=timezone.utc)

        rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)

        if rates is None or len(rates) == 0:
            logger.error(
                f"No data for {symbol} {date_from} → {date_to}: {mt5.last_error()}"
            )
            return None

        df = self._to_dataframe(rates)
        
        # Validation: Check if returned data starts reasonably close to requested date_from
        earliest_returned = df['Datetime'].min()
        if earliest_returned > date_from + timedelta(days=7):
            logger.error(
                f"Broker lacks data for requested period for {symbol}. "
                f"Requested start: {date_from}, Earliest available: {earliest_returned}."
            )
            return None

        logger.info(
            f"Fetched {len(df)} bars for {symbol} "
            f"({df['Datetime'].iloc[0]} → {df['Datetime'].iloc[-1]})"
        )
        return df

    def _probe_max_bars(self, symbol: str, timeframe: int) -> pd.DataFrame | None:
        """
        Binary search for maximum retrievable bars from broker.
        Can take 10-30 seconds — use once at session start, cache the result.
        Returns full DataFrame of all available bars.
        """
        logger.info(f"Probing max available bars for {symbol} TF={timeframe}...")

        low, high = 1, 10_000_000
        best_rates = None

        while low <= high:
            mid   = (low + high) // 2
            if mid <= 0:
                low = 1
                continue

            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, mid)

            if rates is None:
                high = mid - 1
                continue

            # Keep the result that gives us the most bars
            if best_rates is None or len(rates) > len(best_rates):
                best_rates = rates

            returned = len(rates)
            if returned == mid:
                low = mid + 1
            else:
                # When broker returns fewer than requested, the ceiling is somewhere below mid
                high = mid - 1

        if best_rates is None:
            logger.error("Could not retrieve any bars during max probe.")
            return None

        df = self._to_dataframe(best_rates)
        logger.info(
            f"Max available: {len(df)} bars for {symbol} "
            f"({df['Datetime'].iloc[0]} → {df['Datetime'].iloc[-1]})"
        )
        return df

    # ── Resampling ─────────────────────────────────────────────────────────────

    @staticmethod
    def resample(df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
        """
        Purpose:
            Aggregates low-timeframe data (typically M1) into higher timeframes.
            Useful for multi-timeframe analysis without multiple broker requests.

        Arguments:
            df (pd.DataFrame): Source DataFrame in standard schema.
            timeframe_minutes (int): Target timeframe in minutes (5, 15, 60, etc.).

        Returns:
            pd.DataFrame: Resampled DataFrame in standard schema.

        Example:
            >>> m5_df = MT5DataFeed.resample(m1_df, 5)

        Notes:
            Drops incomplete candles automatically (rows with any NaN in OHLC).
        """
        df = df.copy()
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime")

        rule = f"{timeframe_minutes}min"

        resampled = df.resample(rule, closed="left", label="left").agg({
            "Open":        "first",
            "High":        "max",
            "Low":         "min",
            "Close":       "last",
            "TickVolume":  "sum",
            "Spread":      "first",
        })

        # Drop incomplete candles (any NaN in OHLC)
        resampled.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

        return resampled.reset_index()

    # ── Schema conversion ──────────────────────────────────────────────────────

    @staticmethod
    def _to_dataframe(rates) -> pd.DataFrame:
        """Convert MT5 rates struct to standard schema DataFrame."""
        df = pd.DataFrame(rates)

        # Keep only columns we care about — MT5 may include 'real_volume' etc.
        cols_available = [c for c in _RENAME if c in df.columns]
        df = df[cols_available].rename(columns=_RENAME)

        # Validation: Verify all six expected columns are present after rename
        expected_after_rename = ["Open", "High", "Low", "Close", "TickVolume", "Spread"]
        missing = [col for col in expected_after_rename if col not in df.columns]
        if missing:
            logger.error(f"Missing expected columns after rename: {missing}")
            # Return empty DataFrame with correct schema
            return pd.DataFrame(columns=SCHEMA)

        df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s", utc=True)

        # Ensure correct dtypes
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].astype(float)
        df["TickVolume"] = df["TickVolume"].astype(int)
        df["Spread"]     = df["Spread"].astype(int)

        # Enforce column order
        return df[SCHEMA].reset_index(drop=True)
