import os
import json
import logging
import threading
import time
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Optional MT5 import for environments where it's not installed (e.g. Linux CI)
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

# Modules
from Collecting_Data.indicators import IndicatorEngine

logger = logging.getLogger("MMStrategy")

class MMStrategy:
    def __init__(
        self,
        data_feed,                    # DataFeed instance
        send_order,                   # SendOrder instance
        trading_journal,              # TradingJournal instance
        drawdown_manager,             # DrawdownManager instance
        symbols: list[str],           # e.g. ["EURUSD_o", "GBPUSD_o"]
        poll_interval_seconds: float = 5.0,
        swing_lookback: int = 10,
        max_sl_pips: int = 25,
        m5_slope_threshold: float = 0.1,
        m15_slope_threshold: float = 0.1,
        price_to_fast_atr_threshold: float = 1.5,
        fast_to_slow_atr_threshold: float = 3.0,
        reversal_ema_sep_threshold: float = 9.0,
        state_file: str = "mm_strategy_state.json",
    ):
        self.data_feed = data_feed
        self.send_order = send_order
        self.trading_journal = trading_journal
        self.drawdown_manager = drawdown_manager
        self.symbols = symbols
        self.poll_interval_seconds = poll_interval_seconds
        self.swing_lookback = swing_lookback
        self.max_sl_pips = max_sl_pips
        self.m5_slope_threshold = m5_slope_threshold
        self.m15_slope_threshold = m15_slope_threshold
        self.price_to_fast_atr_threshold = price_to_fast_atr_threshold
        self.fast_to_slow_atr_threshold = fast_to_slow_atr_threshold
        self.reversal_ema_sep_threshold = reversal_ema_sep_threshold
        self.state_file = state_file

        self.engine_m5 = IndicatorEngine(ema_periods=[50, 600], slope_period=32)
        self.engine_m15 = IndicatorEngine(ema_periods=[50, 800], slope_period=32)

        self.last_bar_time = {} # symbol -> timeframe -> timestamp
        self.signal_history = {} # symbol -> timeframe -> list of dicts
        self._bar_counters = {} # symbol -> timeframe -> int

        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

        self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    self.last_bar_time = json.load(f)
                logger.info(f"Loaded state from {self.state_file}")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")

    def _save_state(self):
        try:
            temp_file = self.state_file + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(self.last_bar_time, f, indent=4)
            os.replace(temp_file, self.state_file)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def start(self) -> None:
        """Start background polling loop."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("MMStrategy is already running.")
            return
        
        # Reset signal history on start
        self.signal_history = {s: {"M5": [], "M15": []} for s in self.symbols}
        self._bar_counters = {s: {"M5": 0, "M15": 0} for s in self.symbols}
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("MMStrategy started.")

    def stop(self) -> None:
        """Stop polling loop cleanly."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("MMStrategy stopped.")

    def _poll_loop(self):
        while not self._stop_event.is_set():
            try:
                self._poll_cycle()
            except Exception as e:
                logger.error(f"Error in poll cycle: {e}", exc_info=True)
            
            time.sleep(self.poll_interval_seconds)

    def _poll_cycle(self):
        for symbol in self.symbols:
            for timeframe, fast_p, slow_p in [("M5", 50, 600), ("M15", 50, 800)]:
                df_raw = self.data_feed.get_ohlcv(symbol, timeframe)
                if df_raw is None or len(df_raw) < (slow_p + 50):
                    logger.warning(f"Insufficient data for {symbol} {timeframe}")
                    continue
                
                engine = self.engine_m5 if timeframe == "M5" else self.engine_m15
                df = engine.calculate(df_raw)
                
                if not self._is_new_bar(symbol, timeframe, df):
                    continue
                
                # New bar detected
                self._bar_counters[symbol][timeframe] += 1
                logger.info(f"New bar detected for {symbol} {timeframe}: {df.iloc[-1]['Datetime']}")
                
                self._check_and_submit_signal(symbol, timeframe, df, fast_p, slow_p)
                self._save_state()

    def _is_new_bar(self, symbol, timeframe, df):
        current_bar_time = str(df.iloc[-1]["Datetime"])
        with self._lock:
            if symbol not in self.last_bar_time:
                self.last_bar_time[symbol] = {}
            
            if timeframe not in self.last_bar_time[symbol]:
                self.last_bar_time[symbol][timeframe] = current_bar_time
                return False # First time seeing this, don't trigger signal yet
            
            if self.last_bar_time[symbol][timeframe] != current_bar_time:
                self.last_bar_time[symbol][timeframe] = current_bar_time
                return True
        
        return False

    def _check_and_submit_signal(self, symbol, timeframe, df, fast_p, slow_p):
        # Index -1 is forming bar, -2 is last closed bar
        idx_closed = -2
        idx_forming = -1
        
        bar_closed = df.iloc[idx_closed]
        bar_forming = df.iloc[idx_forming]
        
        # Columns
        ema_fast_col = f"ema_{fast_p}"
        ema_slow_col = f"ema_{slow_p}"
        slope_col = f"ema_slope_{slow_p}"
        cross_fast_col = f"cross_ema_{fast_p}"
        dist_fast_col = f"dist_ema_{fast_p}"
        atr_col = "atr_14"

        # Values from forming bar (index -1)
        ema_fast_val = bar_forming[ema_fast_col]
        ema_slow_val = bar_forming[ema_slow_col]
        slope_val = bar_forming[slope_col]
        atr_val = bar_forming[atr_col]
        dist_fast_val = bar_forming[dist_fast_col]
        ema_sep_atr = abs(ema_fast_val - ema_slow_val) / (atr_val + 1e-9)

        # Priority: HR -> STD -> REV
        
        # 1. High-Risk
        hr_dir = self._evaluate_high_risk(bar_closed, ema_fast_val, ema_slow_val, slope_val, cross_fast_col)
        if hr_dir:
            self._process_signal(symbol, timeframe, "high_risk", hr_dir, df)
            return

        # 2. Standard
        std_dir = self._evaluate_standard(bar_closed, ema_fast_val, ema_slow_val, slope_val, ema_sep_atr, dist_fast_val)
        if std_dir:
            self._process_signal(symbol, timeframe, "standard", std_dir, df)
            return

        # 3. Reversal
        rev_dir = self._evaluate_reversal(bar_closed, ema_fast_val, ema_slow_val, ema_sep_atr, cross_fast_col)
        if rev_dir:
            self._process_signal(symbol, timeframe, "reversal", rev_dir, df)
            return

    def _evaluate_high_risk(self, bar_closed, ema_fast_val, ema_slow_val, slope_val, cross_fast_col):
        # 1. Previous candle crosses through fast EMA
        cross = bar_closed[cross_fast_col]
        if cross == 0: return None
        
        # 2. Cross direction aligns with slow EMA trend
        direction = 0
        if cross == 1 and ema_fast_val > ema_slow_val:
            direction = 1
        elif cross == -1 and ema_fast_val < ema_slow_val:
            direction = -1
        else:
            return None
        
        # 3. Previous candle body percentage
        if bar_closed["body_pct"] < 0.70: return None
        
        # 4. Previous candle size vs average
        if bar_closed["body_vs_avg"] < 1.2: return None
        
        # 5. Slow EMA slope
        if slope_val < 0.1: return None
        
        return direction

    def _evaluate_standard(self, bar_closed, ema_fast_val, ema_slow_val, slope_val, ema_sep_atr, dist_fast_val):
        # 1. EMA alignment
        direction = 0
        if ema_fast_val > ema_slow_val:
            direction = 1
        elif ema_fast_val < ema_slow_val:
            direction = -1
        else:
            return None
            
        # 2. Price proximity to fast EMA (ATR-dynamic)
        if abs(dist_fast_val) >= self.price_to_fast_atr_threshold: return None
        
        # 3. EMA separation (ATR-dynamic)
        if ema_sep_atr >= self.fast_to_slow_atr_threshold: return None
        
        # 4. Previous candle body percentage
        if bar_closed["body_pct"] < 0.60: return None
        
        # 5. Previous candle size vs average
        if bar_closed["body_vs_avg"] <= 1.0: return None
        
        # 6. Slow EMA slope
        if slope_val < 0.1: return None
        
        # 7. Direction match (candle confirms EMA direction)
        if bar_closed["candle_direction"] != direction: return None
        
        return direction

    def _evaluate_reversal(self, bar_closed, ema_fast_val, ema_slow_val, ema_sep_atr, cross_fast_col):
        # 1. EMA separation is large
        if ema_sep_atr < self.reversal_ema_sep_threshold: return None
        
        # 2. Previous candle crosses through fast EMA
        cross = bar_closed[cross_fast_col]
        if cross == 0: return None
        
        # 3. Cross direction is OPPOSITE to slow EMA trend
        direction = 0
        if cross == 1 and ema_fast_val < ema_slow_val:
            direction = 1
        elif cross == -1 and ema_fast_val > ema_slow_val:
            direction = -1
        else:
            return None
            
        # 4. Previous candle body percentage
        if bar_closed["body_pct"] < 0.80: return None
        
        # 5. Previous candle size vs average
        if bar_closed["body_vs_avg"] < 1.5: return None
        
        return direction

    def _process_signal(self, symbol, timeframe, signal_type, direction, df):
        # bar_timestamp is the Datetime of the signal bar (bar[-2])
        bar_timestamp = str(df.iloc[-2]["Datetime"])
        
        # Default trade parameters
        if signal_type == "standard":
            tp_level = 2
            stage = "multi"
            risk_pct_default = 0.01
        elif signal_type == "high_risk":
            tp_level = 1
            stage = "single"
            risk_pct_default = 0.005
        else: # reversal
            tp_level = 1
            stage = "single"
            risk_pct_default = 0.003

        # SL Calculation
        sl_price = self._calculate_sl(symbol, direction, df)
        
        # Live ask/bid for entry_price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to fetch tick for {symbol}")
            return
        entry_price = float(tick.ask if direction == 1 else tick.bid)
        
        # Distance metrics for ML
        extra_fields = self._get_signal_distances(symbol, timeframe, signal_type, direction)
        
        # Add indicator values to extra_fields
        idx_forming = -1
        idx_closed = -2
        fast_p = 50
        slow_p = 600 if timeframe == "M5" else 800
        atr_val = float(df.iloc[idx_forming]["atr_14"])
        ema_fast_val = float(df.iloc[idx_forming][f"ema_{fast_p}"])
        ema_slow_val = float(df.iloc[idx_forming][f"ema_{slow_p}"])
        slope_val = float(df.iloc[idx_forming][f"ema_slope_{slow_p}"])
        
        extra_fields.update({
            "ema_fast": ema_fast_val,
            "ema_slow": ema_slow_val,
            "atr": atr_val,
            "body_pct": float(df.iloc[idx_closed]["body_pct"]),
            "body_vs_avg": float(df.iloc[idx_closed]["body_vs_avg"]),
            "ema_slope": slope_val,
            "ema_separation_atr": abs(ema_fast_val - ema_slow_val) / (atr_val + 1e-9),
            "risk_pct_default": risk_pct_default,
        })
        
        trading_allowed = self.drawdown_manager.trading_allowed()
        if not trading_allowed:
            extra_fields["blocked_by_drawdown"] = True
            logger.info(f"Signal {signal_type} {direction} for {symbol} BLOCKED by drawdown")
        
        # Log to journal
        signal_id = self.trading_journal.log_signal(
            signal_type=signal_type,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_level=tp_level,
            stage=stage,
            strategy="mm",
            signal_category=signal_type,
            bar_timestamp=bar_timestamp,
            extra_fields=extra_fields
        )
        
        # Update history
        self._update_signal_history(symbol, timeframe, signal_type, direction)
        
        if trading_allowed:
            logger.info(f"Submitting {signal_type} {direction} for {symbol} {timeframe}")
            res = self.send_order.execute(
                symbol=symbol,
                direction=direction,
                entry_price=0.0, # Market order
                sl_price=sl_price,
                tp_level=tp_level,
                stage=stage,
                strategy="mm",
                signal_category=signal_type,
                signal_id=signal_id
            )
            logger.info(f"Order result for {symbol}: {res.get('success')} - {res.get('reason')}")

    def _calculate_sl(self, symbol, direction, df):
        # SL is based on swing high/low over a lookback (default 10 bars)
        # Using 10 closed bars before the forming bar: df.iloc[-11:-1]
        lookback_df = df.iloc[-11:-1]
        
        if direction == 1:
            sl_price = lookback_df["Low"].min()
        else:
            sl_price = lookback_df["High"].max()
            
        # Entry price for SL calculation (live)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None: return float(sl_price)
        entry_price = tick.ask if direction == 1 else tick.bid
        
        # Max SL distance cap
        info = mt5.symbol_info(symbol)
        if info is None: return float(sl_price)
        
        pip_size = info.point * 10
        max_sl_dist = self.max_sl_pips * pip_size
        
        current_dist = abs(entry_price - sl_price)
        if current_dist > max_sl_dist:
            sl_price = entry_price - direction * max_sl_dist
            logger.warning(f"SL capped for {symbol} at {self.max_sl_pips} pips")
            
        # Minimum SL distance
        stops_level_price = info.trade_stops_level * info.point
        if abs(entry_price - sl_price) < stops_level_price:
            sl_price = entry_price - direction * (stops_level_price + info.point)
            
        return float(sl_price)

    def _get_signal_distances(self, symbol, timeframe, signal_type, direction):
        with self._lock:
            history = self.signal_history.get(symbol, {}).get(timeframe, [])
            count = self._bar_counters[symbol][timeframe]
        
        def find_last(type_filter=None, dir_filter=None):
            for sig in reversed(history):
                if type_filter and sig["type"] != type_filter: continue
                if dir_filter and sig["direction"] != dir_filter: continue
                return sig
            return None

        last_std = find_last(type_filter="standard", dir_filter=direction)
        last_hr = find_last(type_filter="high_risk", dir_filter=direction)
        last_any = find_last(dir_filter=direction)
        
        dist_std = count - last_std["count"] if last_std else -1
        dist_hr = count - last_hr["count"] if last_hr else -1
        dist_any = count - last_any["count"] if last_any else -1

        return {
            f"bars_since_last_standard_{direction}": dist_std,
            f"bars_since_last_high_risk_{direction}": dist_hr,
            "bars_since_last_any_signal": dist_any
        }

    def _update_signal_history(self, symbol, timeframe, signal_type, direction):
        with self._lock:
            count = self._bar_counters[symbol][timeframe]
            self.signal_history[symbol][timeframe].append({
                "type": signal_type,
                "direction": direction,
                "count": count
            })

if __name__ == "__main__":
    import unittest
    from unittest.mock import MagicMock, patch
    import sys

    # Mock MT5 for testing
    mock_mt5 = MagicMock()
    sys.modules["MetaTrader5"] = mock_mt5
    import MetaTrader5 as mt5

    class TestMMStrategy(unittest.TestCase):
        def setUp(self):
            self.data_feed = MagicMock()
            self.send_order = MagicMock()
            self.trading_journal = MagicMock()
            self.drawdown_manager = MagicMock()
            self.symbols = ["EURUSD_o"]
            self.state_file = "test_mm_state.json"
            
            self.strategy = MMStrategy(
                self.data_feed,
                self.send_order,
                self.trading_journal,
                self.drawdown_manager,
                self.symbols,
                state_file=self.state_file
            )
            # Initialize internal structures normally done in start()
            self.strategy.signal_history = {s: {"M5": [], "M15": []} for s in self.symbols}
            self.strategy._bar_counters = {s: {"M5": 0, "M15": 0} for s in self.symbols}
            
            # Default MT5 mocks
            mt5.symbol_info_tick.return_value = MagicMock(ask=1.1000, bid=1.0990)
            mt5.symbol_info.return_value = MagicMock(point=0.00001, trade_stops_level=0)
            self.drawdown_manager.trading_allowed.return_value = True

        def tearDown(self):
            if os.path.exists(self.state_file):
                os.remove(self.state_file)

        def make_df(self, n_bars=850, ema_fast_above_slow=True, bullish_candles=True):
            prices = 1.1000 + np.linspace(0, 0.01 if ema_fast_above_slow else -0.01, n_bars)
            df = pd.DataFrame({
                "Datetime": pd.date_range("2024-01-01", periods=n_bars, freq="5min"),
                "Open": prices,
                "High": prices + 0.0005,
                "Low": prices - 0.0005,
                "Close": prices,
                "TickVolume": 100,
                "Spread": 1
            })
            if bullish_candles:
                df["Open"] = df["Close"] - 0.0007
                df["High"] = df["Close"] + 0.0001
                df["Low"] = df["Open"] - 0.0001
            else:
                df["Open"] = df["Close"] + 0.0007
                df["High"] = df["Open"] + 0.0001
                df["Low"] = df["Close"] - 0.0001
            return df

        def test_standard_buy_signal(self):
            # Test Case 1 & 2: Standard BUY/SELL
            df_raw = self.make_df(ema_fast_above_slow=True, bullish_candles=True)
            self.strategy._is_new_bar = MagicMock(return_value=True)
            self.send_order.execute.return_value = {"success": True}
            
            with patch.object(self.strategy, '_evaluate_standard', return_value=1):
                self.strategy._check_and_submit_signal("EURUSD_o", "M5", self.strategy.engine_m5.calculate(df_raw), 50, 600)
                
            self.trading_journal.log_signal.assert_called_once()
            self.assertEqual(self.trading_journal.log_signal.call_args[1]["signal_type"], "standard")
            self.assertEqual(self.trading_journal.log_signal.call_args[1]["direction"], 1)
            self.send_order.execute.assert_called_once()
            self.assertEqual(self.send_order.execute.call_args[1]["tp_level"], 2)
            self.assertEqual(self.send_order.execute.call_args[1]["stage"], "multi")

        def test_high_risk_buy_signal(self):
            # Test Case 3: High-Risk BUY
            df_raw = self.make_df()
            self.strategy._is_new_bar = MagicMock(return_value=True)
            
            with patch.object(self.strategy, '_evaluate_high_risk', return_value=1):
                self.strategy._check_and_submit_signal("EURUSD_o", "M5", self.strategy.engine_m5.calculate(df_raw), 50, 600)
                
            self.assertEqual(self.trading_journal.log_signal.call_args[1]["signal_type"], "high_risk")
            self.assertEqual(self.send_order.execute.call_args[1]["tp_level"], 1)
            self.assertEqual(self.send_order.execute.call_args[1]["stage"], "single")

        def test_reversal_sell_signal(self):
            # Test Case 4: Reversal SELL
            df_raw = self.make_df()
            self.strategy._is_new_bar = MagicMock(return_value=True)
            
            with patch.object(self.strategy, '_evaluate_reversal', return_value=-1):
                self.strategy._check_and_submit_signal("EURUSD_o", "M5", self.strategy.engine_m5.calculate(df_raw), 50, 600)
                
            self.assertEqual(self.trading_journal.log_signal.call_args[1]["signal_type"], "reversal")
            self.assertEqual(self.trading_journal.log_signal.call_args[1]["direction"], -1)

        def test_signal_priority(self):
            # Test Case 5: Signal priority
            df_raw = self.make_df()
            self.strategy._is_new_bar = MagicMock(return_value=True)
            
            with patch.object(self.strategy, '_evaluate_high_risk', return_value=1), \
                 patch.object(self.strategy, '_evaluate_standard', return_value=1), \
                 patch.object(self.strategy, '_evaluate_reversal', return_value=1):
                self.strategy._check_and_submit_signal("EURUSD_o", "M5", self.strategy.engine_m5.calculate(df_raw), 50, 600)
                
            self.trading_journal.log_signal.assert_called_once()
            self.assertEqual(self.trading_journal.log_signal.call_args[1]["signal_type"], "high_risk")

        def test_no_signal_ema_alignment(self):
            # Test Case 6: No signal when EMA alignment fails (candle direction mismatch)
            df_raw = self.make_df(ema_fast_above_slow=True, bullish_candles=False) # Bearish candle in Bullish trend
            self.strategy._is_new_bar = MagicMock(return_value=True)
            
            self.strategy._check_and_submit_signal("EURUSD_o", "M5", self.strategy.engine_m5.calculate(df_raw), 50, 600)
            self.trading_journal.log_signal.assert_not_called()

        def test_no_signal_repeated_bar(self):
            # Test Case 7: No signal on repeated bar
            df_raw = self.make_df()
            self.data_feed.get_ohlcv.return_value = df_raw
            
            self.strategy._poll_cycle() # First call initializes
            self.strategy._poll_cycle() # Second call same bar
            self.trading_journal.log_signal.assert_not_called()

        def test_drawdown_blocked(self):
            # Test Case 8: Drawdown blocked
            df_raw = self.make_df()
            self.strategy._is_new_bar = MagicMock(return_value=True)
            self.drawdown_manager.trading_allowed.return_value = False
            
            with patch.object(self.strategy, '_evaluate_standard', return_value=1):
                self.strategy._check_and_submit_signal("EURUSD_o", "M5", self.strategy.engine_m5.calculate(df_raw), 50, 600)
                
            self.trading_journal.log_signal.assert_called_once()
            self.assertTrue(self.trading_journal.log_signal.call_args[1]["extra_fields"]["blocked_by_drawdown"])
            self.send_order.execute.assert_not_called()

        def test_sl_capped(self):
            # Test Case 9: SL capped at 25 pips
            df_raw = self.make_df()
            df_raw.loc[df_raw.index[-10:-1], "Low"] = 1.0000 
            sl = self.strategy._calculate_sl("EURUSD_o", 1, df_raw)
            self.assertAlmostEqual(sl, 1.0975) # 1.1000 - 25 * 0.0001

        def test_m15_slope_800(self):
            # Test Case 10: M15 uses ema_slope_800
            df_raw = self.make_df()
            df = self.strategy.engine_m15.calculate(df_raw)
            self.assertIn("ema_slope_800", df.columns)

        def test_signal_distance_tracking(self):
            # Test Case 11: Signal distance tracking
            self.strategy._bar_counters["EURUSD_o"]["M5"] = 100
            self.strategy.signal_history["EURUSD_o"]["M5"] = [{"type": "standard", "direction": 1, "count": 90}]
            dists = self.strategy._get_signal_distances("EURUSD_o", "M5", "standard", 1)
            self.assertEqual(dists["bars_since_last_standard_1"], 10)

        def test_extra_fields_logged(self):
            # Test Case 12: extra_fields logged correctly
            df_raw = self.make_df()
            self.strategy._is_new_bar = MagicMock(return_value=True)
            with patch.object(self.strategy, '_evaluate_standard', return_value=1):
                self.strategy._check_and_submit_signal("EURUSD_o", "M5", self.strategy.engine_m5.calculate(df_raw), 50, 600)
            
            fields = self.trading_journal.log_signal.call_args[1]["extra_fields"]
            required = ["ema_fast", "ema_slow", "atr", "body_pct", "body_vs_avg", "ema_slope", 
                        "ema_separation_atr", "risk_pct_default", "bars_since_last_standard_1", 
                        "bars_since_last_high_risk_1", "bars_since_last_any_signal"]
            for r in required:
                self.assertIn(r, fields)

    unittest.main()
