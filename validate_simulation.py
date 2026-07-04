import os
import sys
import logging
import unittest.mock as mock

# Mock MetaTrader5 before other imports
mock_mt5 = mock.MagicMock()
mock_mt5.TIMEFRAME_M1 = 1
mock_mt5.TIMEFRAME_M5 = 5
mock_mt5.TIMEFRAME_M15 = 15
mock_mt5.ORDER_TYPE_BUY = 0
mock_mt5.ORDER_TYPE_SELL = 1
sys.modules["MetaTrader5"] = mock_mt5

import pandas as pd
from datetime import datetime, timezone
from simulation.simulation_clock import SimulationClock
from simulation.historical_data_feed import HistoricalDataFeed
from simulation.simulation_account import SimulationAccount
from simulation.simulation_broker import SimulationBroker
from simulation.simulation_environment import env
from simulation.simulation_order_engine import SimulationOrderEngine
from simulation.simulation_runner import SimulationRunner
from simulation.statistics_engine import StatisticsEngine
from simulation.backtest_report import BacktestReport

from PositionManager.position_tracker import PositionTracker
from PositionManager.drawdown import DrawdownManager
from PositionManager.risk_sizing import PositionSizer
from PositionManager.exit_manager import ExitManager
from Strategies.mm_strategy import MMStrategy
from Collecting_Data.trading_journal import TradingJournal
from Collecting_Data.logging_config import setup_logging

# Setup Logging
LOG_DIR = "Logs_Backtest"
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
setup_logging(LOG_DIR, level=logging.INFO)
logger = logging.getLogger("ValidateSimulation")

def run_validation():
    logger.info("Starting End-to-End Simulation Validation...")

    # 1. Setup Simulation Infrastructure
    clock = SimulationClock(start_time=datetime(2024, 1, 1, tzinfo=timezone.utc))
    account = SimulationAccount(initial_balance=10000.0)
    broker = SimulationBroker(account)
    data_feed = HistoricalDataFeed(data_dir="Data")

    # Load some data
    # Assuming Data/GBPUSD_M5.csv exists from previous exploration
    if not data_feed.load_symbol_data("GBPUSD_o", "M5", "GBPUSD_M5.csv"):
        logger.error("Failed to load GBPUSD_M5.csv")
        return

    # Set environment to backtest
    env.set_backtest_mode(clock, account, broker, data_feed)

    # Provide symbol info to broker
    broker.set_symbol_info("GBPUSD_o", {
        "digits": 5,
        "point": 0.00001,
        "volume_min": 0.01,
        "volume_step": 0.01,
        "volume_max": 100.0,
        "trade_contract_size": 100000
    })

    # 2. Initialize Trading Framework Modules
    journal = TradingJournal(journal_root="Journals_Backtest", mode="backtest")

    tracker = PositionTracker(
        magic_numbers=[100001, 100002],
        poll_interval_seconds=0, # Not used in manual poll
        state_file="State/sim_tracker_state.json"
    )

    dm = DrawdownManager(
        initial_balance=10000.0,
        position_tracker=tracker,
        state_file="State/sim_drawdown_state.json"
    )

    sizer = PositionSizer()

    # Mock PositionManager for backtest if it still tries to call MT5
    # Actually, we should probably have a SimulationPositionManager
    # For now, let's just use a simple one that routes to broker
    class SimPositionManager:
        def __init__(self, broker, clock):
            self.broker = broker
            self.clock = clock
        def close_position(self, ticket, volume=None):
            tick = env.symbol_info_tick("GBPUSD_o") # Symbol should be dynamic
            return {"success": self.broker.close_position(ticket, volume, tick.bid, self.clock.current_time().timestamp())}
        def modify_position(self, ticket, sl_price=None, tp_price=None):
            return {"success": self.broker.modify_position(ticket, sl_price, tp_price)}

    pm = SimPositionManager(broker, clock)

    em = ExitManager(
        position_tracker=tracker,
        position_manager=pm,
        trading_journal=journal,
        state_file="State/sim_exit_state.json"
    )

    order_engine = SimulationOrderEngine(
        simulation_broker=broker,
        position_tracker=tracker,
        drawdown_manager=dm,
        position_sizer=sizer,
        exit_manager=em,
        trading_journal=journal
    )

    strategy = MMStrategy(
        data_feed=data_feed,
        send_order=order_engine,
        trading_journal=journal,
        drawdown_manager=dm,
        symbols=["GBPUSD_o"],
        state_file="State/sim_strategy_state.json"
    )
    # Manual init for strategy internal states
    strategy.signal_history = {"GBPUSD_o": {"M5": [], "M15": []}}
    strategy._bar_counters = {"GBPUSD_o": {"M5": 0, "M15": 0}}

    # 3. Run Simulation
    runner = SimulationRunner(env, strategy, tracker, em, journal)
    stats = runner.run("GBPUSD_o", "M5", step_minutes=5)

    # 4. Generate Report
    if stats:
        report = BacktestReport.generate(
            stats, "MMStrategy", "GBPUSD_o", "M5",
            "Start", "End"
        )
        print(report)
        logger.info("Validation Report Generated.")
    else:
        logger.warning("No stats generated. Check if any signals were triggered.")

if __name__ == "__main__":
    # Create necessary directories
    for d in ["Journals_Backtest", "Logs_Backtest", "State"]:
        if not os.path.exists(d):
            os.makedirs(d)
    run_validation()
