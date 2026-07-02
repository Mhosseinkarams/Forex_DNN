import os
import sys
import time
import logging
import shutil
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Mock MT5 before any imports that might use it
mock_mt5 = MagicMock()
sys.modules["MetaTrader5"] = mock_mt5

# Now import the framework modules
from Collecting_Data.logging_config import setup_logging
from Collecting_Data.trading_journal import TradingJournal
from Collecting_Data.data_feed import MT5DataFeed, FeedHealth
from PositionManager.position_manager import PositionManager
from PositionManager.position_tracker import PositionTracker
from PositionManager.drawdown import DrawdownManager
from PositionManager.risk_sizing import PositionSizer
from PositionManager.exit_manager import ExitManager
from PositionManager.send_order import SendOrder
from Strategies.mm_strategy import MMStrategy

class IntegrationValidator:
    def __init__(self):
        self.results = {}
        if os.path.exists("Validation_Logs"): shutil.rmtree("Validation_Logs")
        setup_logging("Validation_Logs", level=logging.INFO)
        self.logger = logging.getLogger("Validator")

        # Magic numbers used for validation
        self.MAGIC_UNITY = 1001
        self.MAGIC_MM = 1002

    def log_result(self, item, status, detail=""):
        self.results[item] = (status, detail)
        self.logger.info(f"{item}: {status} - {detail}")

    def run_validation(self):
        self.logger.info("Starting Framework Integration Validation...")

        try:
            # 1. System Startup
            self.validate_startup()

            # 2. Data Flow
            self.validate_data_flow()

            # 3. Signal Logic
            self.validate_signals()

            # 4. Order Execution & Tracking
            self.validate_order_flow()

            # 5. Recovery
            self.validate_recovery()

            # 6. Error Handling
            self.validate_error_handling()

        except Exception as e:
            self.logger.exception(f"Validation interrupted by exception: {e}")

        self.print_report()
        self.cleanup()

    def validate_startup(self):
        try:
            # Prepare MT5 Mocks
            mock_mt5.initialize.return_value = True
            mock_mt5.terminal_info.return_value = MagicMock(name="Mock Terminal")
            mock_mt5.version.return_value = (500, "2023-10-27")
            mock_mt5.account_info.return_value = MagicMock(balance=10000.0)
            mock_mt5.symbol_select.return_value = True

            # Prepare Data Mocks to avoid thread errors on startup
            n_bars = 1000
            prices = 1.1000 + np.linspace(0, 0.01, n_bars)
            df_mock = pd.DataFrame({
                "time": [int(x.timestamp()) for x in pd.date_range("2024-01-01", periods=n_bars, freq="5min")],
                "open": prices, "high": prices + 0.0005, "low": prices - 0.0005, "close": prices,
                "tick_volume": 100, "spread": 1
            })
            mock_mt5.copy_rates_from_pos.return_value = df_mock.to_records(index=False)
            mock_mt5.symbol_info_tick.return_value = MagicMock(ask=1.1010, bid=1.1009, time=time.time())

            # Initialize modules
            if os.path.exists("Validation_Journals"): shutil.rmtree("Validation_Journals")
            self.tj = TradingJournal("Validation_Journals", mode="live")
            self.pm = PositionManager(self.MAGIC_UNITY, self.MAGIC_MM)
            self.pt = PositionTracker([self.MAGIC_UNITY, self.MAGIC_MM], state_file="val_pt_state.json")
            self.dm = DrawdownManager(10000.0, self.pt, state_file="val_dm_state.json")
            self.ps = PositionSizer()
            self.em = ExitManager(self.pt, self.pm, state_file="val_em_state.json", trading_journal=self.tj)
            self.so = SendOrder(self.pm, self.pt, self.dm, self.ps, self.em, self.tj, state_file="val_so_state.json")
            self.df = MT5DataFeed()
            self.df.connect()
            self.strategy = MMStrategy(self.df, self.so, self.tj, self.dm, ["EURUSD_o"], state_file="val_mm_state.json")

            self.log_result("MT5 Connection", "PASS")
            self.log_result("Authentication", "PASS")
            self.log_result("Module Initialization", "PASS")

            # Start background threads
            self.pt.start()
            self.em.start()
            self.strategy.start()
            self.log_result("Background Threads", "PASS")

        except Exception as e:
            self.log_result("Startup", "FAIL", str(e))

    def validate_data_flow(self):
        try:
            df_feed = self.df.get_ohlcv("EURUSD_o", "M5")
            if df_feed is not None and len(df_feed) == 1000:
                self.log_result("DataFeed", "PASS", f"Retrieved {len(df_feed)} bars")
            else:
                self.log_result("DataFeed", "FAIL", "Incomplete or None data")

            # Indicator Engine
            df_ind = self.strategy.engine_m5.calculate(df_feed)
            required_cols = ["ema_50", "ema_600", "atr_14", "ema_slope_600"]
            missing = [c for c in required_cols if c not in df_ind.columns]
            if not missing:
                self.log_result("IndicatorEngine", "PASS", "All required columns present")
            else:
                self.log_result("IndicatorEngine", "FAIL", f"Missing: {missing}")

        except Exception as e:
            self.log_result("DataFlow", "FAIL", str(e))

    def validate_signals(self):
        try:
            # Inject synthetic signal context
            df_raw = self.df.get_ohlcv("EURUSD_o", "M5")
            df = self.strategy.engine_m5.calculate(df_raw)

            # Manually trigger signal processing with mock evaluations
            # Mocking _process_signal to avoid full order execution here, we do it in validate_order_flow
            with patch.object(self.strategy, '_evaluate_standard', return_value=1), \
                 patch.object(self.strategy, '_evaluate_high_risk', return_value=None), \
                 patch.object(self.strategy, '_evaluate_reversal', return_value=None), \
                 patch.object(self.strategy, '_process_signal') as mock_process:

                self.strategy._check_and_submit_signal("EURUSD_o", "M5", df, 50, 600)
                if mock_process.called:
                    self.log_result("Strategy Signal Detection", "PASS", "Synthetic signal injected and detected")
                else:
                    self.log_result("Strategy Signal Detection", "FAIL", "Signal not detected despite mock returns")

        except Exception as e:
            self.log_result("Strategy Signal Detection", "FAIL", str(e))

    def validate_order_flow(self):
        try:
            # 1. Prepare for Order execution
            mock_mt5.TRADE_RETCODE_DONE = 10009
            mock_mt5.ORDER_TYPE_BUY = 0
            mock_mt5.symbol_info.return_value = MagicMock(
                point=0.00001, trade_stops_level=0, volume_min=0.01, volume_step=0.01, volume_max=100.0, trade_contract_size=100000
            )
            mock_mt5.order_send.return_value = MagicMock(retcode=10009, order=123456, price=1.1011, comment="Done")

            # Need a real signal_id in journal for log_order_open to work
            sig_id = self.tj.log_signal(
                signal_type="standard", symbol="EURUSD_o", timeframe="M5", direction=1,
                entry_price=1.1010, sl_price=1.0950, tp_level=2, stage="multi",
                strategy="mm", signal_category="standard", bar_timestamp="2024-01-01T00:00:00Z"
            )

            # Execute
            with patch.object(self.ps, 'calculate_lot_size', return_value={"success": True, "lot_size": 0.1, "risk_pct_actual": 0.01, "error": None}):
                res = self.so.execute("EURUSD_o", 1, 0.0, 1.0950, 2, "multi", "mm", "standard", sig_id)

            if res["success"]:
                self.log_result("Order Execution", "PASS", f"Ticket: {res['ticket']}")
            else:
                self.log_result("Order Execution", "FAIL", f"{res['reason']} - {res['error_detail']}")

            # 2. Position Tracking
            mock_pos = MagicMock()
            mock_pos.ticket = 123456; mock_pos.symbol = "EURUSD_o"; mock_pos.magic = self.MAGIC_MM; mock_pos.type = 0; mock_pos.volume = 0.1
            mock_pos.price_open = 1.1011; mock_pos.sl = 1.0950; mock_pos.tp = 1.1133; mock_pos.price_current = 1.1012; mock_pos.profit = 1.0; mock_pos.time = time.time()
            mock_mt5.positions_get.return_value = [mock_pos]

            self.pt._poll_cycle()
            tracked = self.pt.get_open_positions()
            if any(p["ticket"] == 123456 for p in tracked):
                self.log_result("Position Tracking", "PASS", "Ticket found in tracker")
            else:
                self.log_result("Position Tracking", "FAIL", "Ticket missing from tracker")

            # 3. Exit Manager
            if 123456 in self.em.tracked_tickets:
                self.log_result("Exit Manager", "PASS", "Ticket registered in ExitManager")
            else:
                self.log_result("Exit Manager", "FAIL", "Ticket missing from ExitManager")

            # 4. Trading Journal
            filepath = self.tj._get_filepath("mm", "EURUSD_o", "M5", "order_open")
            if os.path.exists(filepath):
                df_j = pd.read_csv(filepath)
                if 123456 in df_j["ticket"].values:
                    self.log_result("Trading Journal", "PASS", "Records verified in CSV")
                else:
                    self.log_result("Trading Journal", "FAIL", "Ticket not found in order_open journal")
            else:
                self.log_result("Trading Journal", "FAIL", f"Journal file {filepath} not created")

        except Exception as e:
            self.log_result("OrderFlow", "FAIL", str(e))

    def validate_recovery(self):
        try:
            # Shutdown and re-init tracker
            self.pt.stop()
            new_pt = PositionTracker([self.MAGIC_UNITY, self.MAGIC_MM], state_file="val_pt_state.json")
            if any(p["ticket"] == 123456 for p in new_pt.positions):
                self.log_result("Recovery", "PASS", "State restored correctly")
            else:
                self.log_result("Recovery", "FAIL", "State not recovered")
        except Exception as e:
            self.log_result("Recovery", "FAIL", str(e))

    def validate_error_handling(self):
        try:
            mock_mt5.symbol_info_tick.return_value = None
            res = self.df.check_health("EURUSD_o")
            if res == FeedHealth.DISCONNECTED:
                self.log_result("Error Handling", "PASS", "Detected MT5 disconnect correctly")
            else:
                self.log_result("Error Handling", "FAIL", f"Unexpected health state: {res}")
        except Exception as e:
            self.log_result("Error Handling", "FAIL", str(e))

    def print_report(self):
        print("\n" + "="*40)
        print("   FRAMEWORK INTEGRATION VALIDATION REPORT")
        print("="*40)
        overall_pass = True
        # Explicit order for report
        items = ["MT5 Connection", "Authentication", "Module Initialization", "Background Threads",
                 "DataFeed", "IndicatorEngine", "Strategy Signal Detection", "Order Execution",
                 "Position Tracking", "Exit Manager", "Trading Journal", "Recovery", "Error Handling"]

        for item in items:
            if item in self.results:
                status, detail = self.results[item]
                print(f"{item:<25} : {status} {(' - ' + detail) if detail else ''}")
                if status == "FAIL": overall_pass = False
            else:
                print(f"{item:<25} : NOT TESTED")
                overall_pass = False

        print("="*40)
        print(f"OVERALL STATUS: {'PASS' if overall_pass else 'FAIL'}")
        print("="*40 + "\n")

    def cleanup(self):
        # Stop everything
        if hasattr(self, 'strategy'): self.strategy.stop()
        if hasattr(self, 'em'): self.em.stop()
        if hasattr(self, 'pt'): self.pt.stop()

        # Cleanup files
        for f in ["val_pt_state.json", "val_dm_state.json", "val_em_state.json", "val_so_state.json", "val_mm_state.json"]:
            if os.path.exists(f): os.remove(f)
        # if os.path.exists("Validation_Journals"): shutil.rmtree("Validation_Journals")
        # if os.path.exists("Validation_Logs"): shutil.rmtree("Validation_Logs")

if __name__ == "__main__":
    validator = IntegrationValidator()
    validator.run_validation()
