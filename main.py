import time
import logging
import signal
import sys
import os
from datetime import datetime, timezone

# Module imports
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

import MetaTrader5 as mt5

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SYMBOLS = ["EURUSD_o", "GBPUSD_o", "XAUUSD_o"]
TIMEFRAMES = ["M5", "M15"]
MAGIC_UNITY = 100001
MAGIC_MM = 100002
INITIAL_BALANCE = 10000.0  # Should be updated from account info on start
DAILY_LIMIT_PCT = 0.03
TOTAL_LIMIT_PCT = 0.10
POLL_INTERVAL_TRACKER = 5
POLL_INTERVAL_EXIT = 1
POLL_INTERVAL_STRATEGY = 5
JOURNAL_ROOT = "Journals"
LOG_DIR = "Logs"
STATE_DIR = "State"

# Ensure directories exist
for d in [JOURNAL_ROOT, LOG_DIR, STATE_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

logger = logging.getLogger("Main")

class TradingApplication:
    def __init__(self):
        self.running = True
        self.modules = {}

    def initialize(self):
        # 1. Initialize Logging
        setup_logging(LOG_DIR, level=logging.INFO)
        logger.info("========================================")
        logger.info("Initializing Forex Trading Framework...")
        logger.info("========================================")

        # 2. Initialize MT5
        creds = load_credentials(path="credentials.json")
        if not mt5.initialize(login=creds["login"], password=creds["password"], server=creds["server"]):
            logger.error(f"MT5 Initialization Failed: {mt5.last_error()}")
            sys.exit(1)

        logger.info(f"MT5 Connected: {mt5.terminal_info().name} (Account: {creds['login']})")

        # Get actual balance
        acc_info = mt5.account_info()
        if acc_info is None:
            logger.error("Failed to retrieve account info.")
            sys.exit(1)

        current_balance = acc_info.balance
        logger.info(f"Account Balance: ${current_balance:.2f}")

        # 3. Create Objects in dependency order
        try:
            # Journal
            tj = TradingJournal(journal_root=JOURNAL_ROOT, mode="live")
            self.modules["journal"] = tj

            # Position Manager
            pm = PositionManager(magic_unity=MAGIC_UNITY, magic_mm=MAGIC_MM)
            self.modules["position_manager"] = pm

            # Position Tracker
            pt = PositionTracker(
                magic_numbers=[MAGIC_UNITY, MAGIC_MM],
                poll_interval_seconds=POLL_INTERVAL_TRACKER,
                state_file=os.path.join(STATE_DIR, "position_tracker_state.json")
            )
            self.modules["position_tracker"] = pt

            # Drawdown Manager
            dm = DrawdownManager(
                initial_balance=current_balance, # Using current as initial for live start if not persisted
                position_tracker=pt,
                daily_limit_pct=DAILY_LIMIT_PCT,
                total_limit_pct=TOTAL_LIMIT_PCT,
                state_file=os.path.join(STATE_DIR, "drawdown_state.json")
            )
            self.modules["drawdown_manager"] = dm

            # Position Sizer
            ps = PositionSizer()
            self.modules["position_sizer"] = ps

            # Exit Manager
            em = ExitManager(
                position_tracker=pt,
                position_manager=pm,
                poll_interval_seconds=POLL_INTERVAL_EXIT,
                state_file=os.path.join(STATE_DIR, "exit_manager_state.json"),
                trading_journal=tj
            )
            self.modules["exit_manager"] = em

            # Send Order
            so = SendOrder(
                position_manager=pm,
                position_tracker=pt,
                drawdown_manager=dm,
                position_sizer=ps,
                exit_manager=em,
                trading_journal=tj,
                state_file=os.path.join(STATE_DIR, "send_order_state.json")
            )
            self.modules["send_order"] = so

            # Data Feed
            df = MT5DataFeed()
            if not df.connect():
                logger.error("DataFeed failed to connect.")
                sys.exit(1)
            self.modules["data_feed"] = df

            # MM Strategy
            strategy = MMStrategy(
                data_feed=df,
                send_order=so,
                trading_journal=tj,
                drawdown_manager=dm,
                symbols=SYMBOLS,
                poll_interval_seconds=POLL_INTERVAL_STRATEGY,
                state_file=os.path.join(STATE_DIR, "mm_strategy_state.json")
            )
            self.modules["strategy"] = strategy

            logger.info("Modules Initialized Successfully.")

        except Exception as e:
            logger.exception(f"Module Initialization Error: {e}")
            sys.exit(1)

    def start(self):
        logger.info("Starting Background Services...")

        self.modules["position_tracker"].start()
        logger.info("PositionTracker Started.")

        self.modules["exit_manager"].start()
        logger.info("ExitManager Started.")

        self.modules["strategy"].start()
        logger.info("Strategy Started.")

    def run(self):
        logger.info("Trading System is Active. Press Ctrl+C to shutdown.")
        try:
            while self.running:
                # Periodic high-level health checks or drawdown updates
                self.modules["drawdown_manager"].check()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown Signal Received.")
            self.shutdown()

    def shutdown(self):
        logger.info("Commencing Graceful Shutdown...")
        self.running = False

        if "strategy" in self.modules:
            logger.info("Stopping Strategy...")
            self.modules["strategy"].stop()

        if "exit_manager" in self.modules:
            logger.info("Stopping ExitManager...")
            self.modules["exit_manager"].stop()

        if "position_tracker" in self.modules:
            logger.info("Stopping PositionTracker...")
            self.modules["position_tracker"].stop()

        if "data_feed" in self.modules:
            logger.info("Closing DataFeed...")
            self.modules["data_feed"].disconnect()

        logger.info("Closing MT5 Connection...")
        mt5.shutdown()

        logger.info("Shutdown Complete.")
        sys.exit(0)

if __name__ == "__main__":
    app = TradingApplication()

    # Handle OS signals for termination
    def signal_handler(sig, frame):
        app.shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    app.initialize()
    app.start()
    app.run()
