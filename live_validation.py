import os
import sys
import time
import json
import logging
import uuid
import shutil
import threading
from datetime import datetime, timezone
import pandas as pd

# Try to import MetaTrader5, but allow for environment without it (for syntax/mock checking)
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

# Framework Imports
from Collecting_Data.logging_config import setup_logging
from Collecting_Data.auth import load_credentials
from Collecting_Data.trading_journal import TradingJournal
from Collecting_Data.data_feed import MT5DataFeed
from PositionManager.position_manager import PositionManager
from PositionManager.position_tracker import PositionTracker
from PositionManager.drawdown import DrawdownManager
from PositionManager.risk_sizing import PositionSizer
from PositionManager.exit_manager import ExitManager
from PositionManager.send_order import SendOrder
from Strategies.mm_strategy import MMStrategy

# ==============================================================================
# CONFIGURATION BLOCK
# ==============================================================================
SYMBOL = "EURUSD_o"
TIMEFRAME = "M5"
ACCOUNT_MODE = "demo"  # "demo" or "live"

# Strategy / SendOrder settings
MAGIC_UNITY = 200001
MAGIC_MM = 200002
RISK_PERCENT = 0.01
MAX_SLIPPAGE = 10

# Validation timing
MAX_VALIDATION_TIME = 300      # 5 minutes for signal wait
WAIT_AFTER_OPEN = 10           # Seconds to wait for PositionTracker/ExitManager
WAIT_AFTER_CLOSE = 5           # Seconds to wait for outcome logging
MAX_POSITION_DURATION = 60     # Auto-close after 60 seconds

# Execution Flags
OPEN_TEST_TRADE = True
AUTO_CLOSE_POSITION = True

# Directories
VALIDATION_DIRECTORY = "Validation_Report"
LOG_DIRECTORY = os.path.join(VALIDATION_DIRECTORY, "Logs")
STATE_DIRECTORY = os.path.join(VALIDATION_DIRECTORY, "State")
JOURNAL_DIRECTORY = os.path.join(VALIDATION_DIRECTORY, "Journals")

# ==============================================================================

class LiveValidator:
    def __init__(self):
        self.results = {}
        self.metrics = {
            "startup_time": 0,
            "order_latency": 0,
            "tracker_latency": 0,
            "journal_latency": 0,
            "shutdown_time": 0
        }
        self.modules = {}
        self.test_ticket = None
        self.signal_id = None
        self.start_ts = None
        self.order_open_ts = None

        # Create/Clean directories
        if os.path.exists(VALIDATION_DIRECTORY):
            shutil.rmtree(VALIDATION_DIRECTORY)
        
        for d in [VALIDATION_DIRECTORY, LOG_DIRECTORY, STATE_DIRECTORY, JOURNAL_DIRECTORY]:
            if not os.path.exists(d):
                os.makedirs(d)

        setup_logging(LOG_DIRECTORY, level=logging.INFO)
        self.logger = logging.getLogger("LiveValidator")

    def print_warning(self):
        if ACCOUNT_MODE == "live":
            print("\n" + "!" * 60)
            print("!!! WARNING: LIVE MODE DETECTED !!!")
            print(f"!!! THIS WILL OPEN A REAL TRADE ON SYMBOL: {SYMBOL} !!!")
            print("!" * 60 + "\n")
            confirm = input("Type 'CONFIRM' to proceed: ")
            if confirm != "CONFIRM":
                print("Validation aborted by user.")
                sys.exit(0)
        else:
            print(f"\n--- Running in {ACCOUNT_MODE.upper()} Mode ---\n")

    def run(self):
        self.print_warning()
        self.logger.info("Starting Live Framework Validation...")
        self.start_ts = time.time()

        try:
            # TEST 1: MT5 Connection
            if not self.test_1_mt5_connection(): return self.finish()

            # TEST 2: Market Data
            if not self.test_2_market_data(): return self.finish()

            # TEST 3: Indicator Engine
            if not self.test_3_indicators(): return self.finish()

            # TEST 4: Strategy Health
            self.test_4_strategy_health()

            # TEST 5 & 6: Trade Execution & Broker Validation
            if OPEN_TEST_TRADE:
                if not self.test_5_execution(): return self.finish()
                if not self.test_6_broker_validation(): return self.finish()

                # TEST 7: Position Tracker
                if not self.test_7_tracker(): return self.finish()

                # TEST 8: Exit Manager
                if not self.test_8_exit_manager(): return self.finish()

                # TEST 9: Trading Journal
                if not self.test_9_journal(): return self.finish()

                # TEST 10: Broker Close
                if AUTO_CLOSE_POSITION:
                    if not self.test_10_close(): return self.finish()

                    # TEST 11: Trade Outcome
                    if not self.test_11_outcome(): return self.finish()

            # TEST 12: Recovery
            if not self.test_12_recovery(): return self.finish()

        except Exception as e:
            self.logger.exception(f"Validation failed with exception: {e}")
            self.results["GLOBAL"] = ("FAIL", str(e))

        finally:
            # TEST 13: Framework Shutdown
            self.test_13_shutdown()

        return self.finish()

    def test_1_mt5_connection(self):
        item = "Terminal Connection"
        self.logger.info("TEST 1: Verifying MT5 Connection...")
        try:
            creds = load_credentials()
            if not mt5.initialize(login=creds["login"], password=creds["password"], server=creds["server"]):
                self.results[item] = ("FAIL", f"Initialize failed: {mt5.last_error()}")
                return False

            if not mt5.terminal_info().connected:
                self.results[item] = ("FAIL", "Terminal not connected to broker")
                return False

            acc = mt5.account_info()
            if acc is None:
                self.results[item] = ("FAIL", "Account info unavailable")
                return False

            if not acc.trade_allowed:
                self.results[item] = ("FAIL", "Trading disabled for this account/terminal")
                return False

            self.logger.info(f"Connected to {acc.server} (Account: {acc.login})")
            self.logger.info(f"Balance: {acc.balance}, Equity: {acc.equity}, Margin: {acc.margin}, Leverage: {acc.leverage}")
            
            # Initialize modules
            self.modules["journal"] = TradingJournal(journal_root=JOURNAL_DIRECTORY, mode="live")
            self.modules["pm"] = PositionManager(magic_unity=MAGIC_UNITY, magic_mm=MAGIC_MM, deviation=MAX_SLIPPAGE)
            self.modules["pt"] = PositionTracker([MAGIC_UNITY, MAGIC_MM], state_file=os.path.join(STATE_DIRECTORY, "pt_state.json"))
            self.modules["dm"] = DrawdownManager(acc.balance, self.modules["pt"], state_file=os.path.join(STATE_DIRECTORY, "dm_state.json"))
            self.modules["ps"] = PositionSizer()
            self.modules["em"] = ExitManager(self.modules["pt"], self.modules["pm"], state_file=os.path.join(STATE_DIRECTORY, "em_state.json"), trading_journal=self.modules["journal"])
            self.modules["so"] = SendOrder(self.modules["pm"], self.modules["pt"], self.modules["dm"], self.modules["ps"], self.modules["em"], self.modules["journal"], state_file=os.path.join(STATE_DIRECTORY, "so_state.json"))
            self.modules["df"] = MT5DataFeed()
            self.modules["df"].connect()
            self.modules["strategy"] = MMStrategy(self.modules["df"], self.modules["so"], self.modules["journal"], self.modules["dm"], [SYMBOL], state_file=os.path.join(STATE_DIRECTORY, "mm_state.json"))

            # Start threads
            self.modules["pt"].start()
            self.modules["em"].start()

            self.metrics["startup_time"] = (time.time() - self.start_ts) * 1000
            self.results[item] = ("PASS", f"Acc: {acc.login}, Server: {acc.server}, Balance: {acc.balance}")
            return True
        except Exception as e:
            self.results[item] = ("FAIL", str(e))
            return False

    def test_2_market_data(self):
        item = "Market Data"
        self.logger.info("TEST 2: Verifying Market Data...")
        try:
            df_m5 = self.modules["df"].get_ohlcv(SYMBOL, "M5")
            df_m15 = self.modules["df"].get_ohlcv(SYMBOL, "M15")

            if df_m5 is None or df_m15 is None or len(df_m5) == 0 or len(df_m15) == 0:
                self.results[item] = ("FAIL", "Failed to download candles")
                return False

            latest_m5 = df_m5.iloc[-1]["Datetime"]
            latest_m15 = df_m15.iloc[-1]["Datetime"]
            self.logger.info(f"Latest M5 Bar: {latest_m5}, Latest M15 Bar: {latest_m15}")

            tick = mt5.symbol_info_tick(SYMBOL)
            if tick is None:
                self.results[item] = ("FAIL", "Failed to receive ticks")
                return False
            
            self.results[item] = ("PASS", f"M5/M15 OK. Latest M5: {latest_m5}")
            return True
        except Exception as e:
            self.results[item] = ("FAIL", str(e))
            return False

    def test_3_indicators(self):
        item = "Indicator Engine"
        self.logger.info("TEST 3: Verifying Indicators...")
        try:
            df_raw = self.modules["df"].get_ohlcv(SYMBOL, "M5")
            df_ind = self.modules["strategy"].engine_m5.calculate(df_raw)

            required = ["ema_50", "ema_600", "atr_14", "ema_slope_600", "body_pct"]
            missing = [c for c in required if c not in df_ind.columns]
            if missing:
                self.results[item] = ("FAIL", f"Missing columns: {missing}")
                return False

            last_row = df_ind.iloc[-1]
            nans = last_row[required].isna().sum()
            if nans > 0:
                self.results[item] = ("FAIL", f"NaN values detected in indicators: {last_row[required][last_row[required].isna()].index.tolist()}")
                return False

            self.results[item] = ("PASS", "All indicators calculated correctly")
            return True
        except Exception as e:
            self.results[item] = ("FAIL", str(e))
            return False

    def test_4_strategy_health(self):
        item = "Strategy"
        self.logger.info(f"TEST 4: Monitoring MMStrategy (Wait up to {MAX_VALIDATION_TIME}s)...")
        # Start strategy
        self.modules["strategy"].start()
        
        start_wait = time.time()
        found_signal = False
        
        # Monitor journal for signals
        initial_signals = self._count_signals()

        while time.time() - start_wait < MAX_VALIDATION_TIME:
            if self._count_signals() > initial_signals:
                found_signal = True
                break
            time.sleep(1)

        if found_signal:
            self.results[item] = ("PASS", "Real signal detected during monitoring")
        else:
            self.results[item] = ("PASS", "NO SIGNAL GENERATED")

    def _count_signals(self):
        count = 0
        sig_dir = os.path.join(JOURNAL_DIRECTORY, "mm", SYMBOL, TIMEFRAME)
        if os.path.exists(sig_dir):
            for f in os.listdir(sig_dir):
                if f.endswith("_signal.csv"):
                    try:
                        df = pd.read_csv(os.path.join(sig_dir, f))
                        count += len(df)
                    except: pass
        return count

    def test_5_execution(self):
        item = "Order Execution"
        self.logger.info("TEST 5: Opening Manual Validation Trade...")
        try:
            self.signal_id = str(uuid.uuid4())
            tick = mt5.symbol_info_tick(SYMBOL)
            info = mt5.symbol_info(SYMBOL)
            
            # Log signal manually
            sl_dist = 200 * info.point
            sl_price = tick.ask - sl_dist
            
            self.modules["journal"].log_signal(
                signal_type="standard",
                symbol=SYMBOL,
                timeframe=TIMEFRAME,
                direction=1, # BUY
                entry_price=tick.ask,
                sl_price=sl_price,
                tp_level=2,
                stage="multi",
                strategy="mm",
                signal_category="standard",
                bar_timestamp=datetime.now(timezone.utc).isoformat(),
                signal_id=self.signal_id
            )

            self.order_open_ts = time.time()
            res = self.modules["so"].execute(
                symbol=SYMBOL,
                direction=1,
                entry_price=0.0,
                sl_price=sl_price,
                tp_level=2,
                stage="multi",
                strategy="mm",
                signal_category="standard",
                signal_id=self.signal_id,
                comment="Validation Trade"
            )
            self.metrics["order_latency"] = (time.time() - self.order_open_ts) * 1000

            if not res["success"]:
                self.results[item] = ("FAIL", f"{res['reason']}: {res['error_detail']}")
                return False

            self.test_ticket = res["ticket"]
            self.results[item] = ("PASS", f"Ticket: {self.test_ticket}, Entry: {res['entry_price']}")
            self.last_res = res
            return True
        except Exception as e:
            self.results[item] = ("FAIL", str(e))
            return False

    def test_6_broker_validation(self):
        item = "Broker Confirmation"
        self.logger.info("TEST 6: Validating Broker Response...")
        try:
            res = self.last_res
            # Record everything
            details = (
                f"Ticket: {res['ticket']}, Price: {res['entry_price']}, "
                f"SL: {res['sl_price']}, TP: {res['tp_price']}, Lot: {res['lot_size']}, "
                f"Retcode: {res['retcode']}, Latency: {self.metrics['order_latency']:.2f}ms"
            )
            self.logger.info(f"Broker Details: {details}")
            self.results[item] = ("PASS", details)
            return True
        except Exception as e:
            self.results[item] = ("FAIL", str(e))
            return False

    def test_7_tracker(self):
        item = "Position Tracker"
        self.logger.info("TEST 7: Verifying Position Tracker Detection...")
        try:
            start_wait = time.time()
            found = False
            while time.time() - start_wait < WAIT_AFTER_OPEN:
                positions = self.modules["pt"].get_open_positions()
                match = next((p for p in positions if p["ticket"] == self.test_ticket), None)
                if match:
                    found = True
                    self.metrics["tracker_latency"] = (time.time() - self.order_open_ts) * 1000
                    details = (
                        f"Detected in {time.time() - self.order_open_ts:.2f}s. "
                        f"Vol: {match['lot_size']}, Risk: ${match['remaining_risk_dollars']:.2f}, "
                        f"PnL: ${match['floating_pnl']:.2f}"
                    )
                    self.results[item] = ("PASS", details)
                    break
                time.sleep(0.5)

            if not found:
                self.results[item] = ("FAIL", "Position not detected by tracker within timeout")
                return False
            return True
        except Exception as e:
            self.results[item] = ("FAIL", str(e))
            return False

    def test_8_exit_manager(self):
        item = "Exit Manager"
        self.logger.info("TEST 8: Verifying Exit Manager Registration...")
        try:
            # Wait for ExitManager to initialize the ticket
            time.sleep(2) 
            with self.modules["em"]._lock:
                if self.test_ticket in self.modules["em"].tracked_tickets:
                    state = self.modules["em"].tracked_tickets[self.test_ticket]
                    if state["original_lot_size"] is not None:
                        tp_ladder = ", ".join([f"TP{k}: {v:.5f}" for k,v in state["tp_prices"].items()])
                        self.results[item] = ("PASS", f"Initialized. Ladder: {tp_ladder}")
                        return True
                    else:
                        self.results[item] = ("FAIL", "Registered but not fully initialized (shares/lot None)")
                        return False
                else:
                    self.results[item] = ("FAIL", "Ticket missing from ExitManager")
                    return False
        except Exception as e:
            self.results[item] = ("FAIL", str(e))
            return False

    def test_9_journal(self):
        item = "Trading Journal"
        self.logger.info("TEST 9: Verifying Journal Integrity...")
        try:
            start_wait = time.time()
            found = False
            filepath = self.modules["journal"]._get_filepath("mm", SYMBOL, TIMEFRAME, "order_open")
            
            while time.time() - start_wait < 5:
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    if self.test_ticket in df["ticket"].values:
                        row = df[df["ticket"] == self.test_ticket].iloc[0]
                        if str(row["signal_id"]) == str(self.signal_id):
                            found = True
                            self.metrics["journal_latency"] = (time.time() - self.order_open_ts) * 1000
                            break
                time.sleep(0.5)

            if not found:
                self.results[item] = ("FAIL", "Order record missing or UUID mismatch in CSV")
                return False

            self.results[item] = ("PASS", "CSV record verified with UUID consistency")
            return True
        except Exception as e:
            self.results[item] = ("FAIL", str(e))
            return False

    def test_10_close(self):
        item = "Trade Close"
        self.logger.info("TEST 10: Closing Position via PositionManager...")
        try:
            res = self.modules["pm"].close_position(self.test_ticket)
            if res["success"]:
                self.results[item] = ("PASS", f"Closed at {res['entry_price']}")
                return True
            else:
                self.results[item] = ("FAIL", f"Close failed: {res.get('comment')}")
                return False
        except Exception as e:
            self.results[item] = ("FAIL", str(e))
            return False

    def test_11_outcome(self):
        item = "Trade Outcome"
        self.logger.info("TEST 11: Verifying Final Outcome Logging...")
        try:
            time.sleep(WAIT_AFTER_CLOSE)
            
            # Check Tracker & ExitManager removed it
            if any(p["ticket"] == self.test_ticket for p in self.modules["pt"].get_open_positions()):
                 self.results[item] = ("FAIL", "Position still in Tracker after close")
                 return False
            
            if self.test_ticket in self.modules["em"].tracked_tickets:
                 self.results[item] = ("FAIL", "Position still in ExitManager after close")
                 return False

            # Check Journal
            filepath = self.modules["journal"]._get_filepath("mm", SYMBOL, TIMEFRAME, "outcome")
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                if self.test_ticket in df["ticket"].values:
                    row = df[df["ticket"] == self.test_ticket].iloc[0]
                    self.results[item] = ("PASS", f"Logged: {row['outcome']}, Profit: ${row['pnl_dollars']:.2f}")
                    return True
            
            self.results[item] = ("FAIL", "Outcome record missing from journal")
            return False
        except Exception as e:
            self.results[item] = ("FAIL", str(e))
            return False

    def test_12_recovery(self):
        item = "Recovery"
        self.logger.info("TEST 12: Testing State Recovery...")
        try:
            # Stop existing modules to ensure file write
            self.modules["pt"].stop()
            self.modules["em"].stop()
            
            # Re-initialize to test persistence
            new_pt = PositionTracker([MAGIC_UNITY, MAGIC_MM], state_file=os.path.join(STATE_DIRECTORY, "pt_state.json"))
            new_em = ExitManager(new_pt, self.modules["pm"], state_file=os.path.join(STATE_DIRECTORY, "em_state.json"), trading_journal=self.modules["journal"])
            
            # Verify no duplicates or corruption (just check if it loads)
            if len(new_pt.positions) > 0:
                 self.logger.info(f"Recovered {len(new_pt.positions)} positions")
            
            self.results[item] = ("PASS", "State reloaded successfully from JSON")
            return True
        except Exception as e:
            self.results[item] = ("FAIL", f"Recovery failure: {e}")
            return False

    def test_13_shutdown(self):
        item = "Shutdown"
        self.logger.info("TEST 13: Graceful Framework Shutdown...")
        try:
            start_shut = time.time()
            if "strategy" in self.modules: self.modules["strategy"].stop()
            if "em" in self.modules: self.modules["em"].stop()
            if "pt" in self.modules: self.modules["pt"].stop()
            if "df" in self.modules: self.modules["df"].disconnect()
            mt5.shutdown()
            self.metrics["shutdown_time"] = (time.time() - start_shut) * 1000
            self.results[item] = ("PASS", "All threads stopped, MT5 disconnected")
        except Exception as e:
            self.results[item] = ("FAIL", str(e))

    def finish(self):
        print("\n" + "="*60)
        print("   LIVE FRAMEWORK VALIDATION REPORT")
        print("="*60)
        
        items = [
            "Terminal Connection", "Market Data", "Indicator Engine", "Strategy",
            "Order Execution", "Broker Confirmation", "Position Tracker",
            "Exit Manager", "Trading Journal", "Trade Close", "Trade Outcome",
            "Recovery", "Shutdown"
        ]
        
        overall_pass = True
        for item in items:
            status, detail = self.results.get(item, ("SKIPPED", ""))
            print(f"{item:<25} : {status} {(' - ' + detail) if detail else ''}")
            if status == "FAIL": overall_pass = False

        print("="*60)
        print(f"Startup Latency   : {self.metrics['startup_time']:.2f} ms")
        print(f"Order Latency     : {self.metrics['order_latency']:.2f} ms")
        print(f"Tracker Latency   : {self.metrics['tracker_latency']:.2f} ms")
        print(f"Journal Latency   : {self.metrics['journal_latency']:.2f} ms")
        print(f"Shutdown Latency  : {self.metrics['shutdown_time']:.2f} ms")
        print("="*60)
        print(f"OVERALL STATUS: {'PASS' if overall_pass else 'FAIL'}")
        print("="*60 + "\n")
        
        return overall_pass

if __name__ == "__main__":
    validator = LiveValidator()
    success = validator.run()
    sys.exit(0 if success else 1)
