import os
import sys
import logging
import yaml
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

# Framework Imports
from Configs.path_manager import PathManager
from Collecting_Data.trading_journal import TradingJournal
from Trade_Execution.position_manager import PositionManager
from Trade_Execution.position_tracker import PositionTracker
from Trade_Execution.drawdown import DrawdownManager
from Trade_Execution.risk_sizing import PositionSizer
from Trade_Execution.exit_manager import ExitManager
from Trade_Execution.send_order import SendOrder
from Visualization.chart_annotator import ChartAnnotationEngine

# Simulation Imports
from Simulation.simulation_environment import env as mt5
from Simulation.simulation_clock import SimulationClock
from Simulation.simulation_account import SimulationAccount
from Simulation.simulation_broker import SimulationBroker
from Simulation.historical_data_feed import HistoricalDataFeed

# Strategy Imports
from Strategies.mm_strategy import MMStrategy
from Strategies.sm_strategy import SMStrategy
from Strategies.unit_strategy import UniTStrategy

"""_summary_

Returns:
    _type_: _description_
"""
logger = logging.getLogger("TradingPipeline")


class TradingPipeline:
    """
    Central orchestrator for the live/backtest trading runtimes.
    Initializes and wires all modules in exact dependency order.
    Supports live, demo, paper, validation, backtest, and simulation run modes.
    """
    def __init__(self, config_path: str = "Configs/trading_config.yaml"):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.modules: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self):
        """Loads trading pipeline settings from YAML with safe defaults."""
        defaults = {
            "trading_mode": "backtest",
            "shadow_mode": True,
            "symbols": ["EURUSD"],
            "timeframe": "M5",
            "initial_balance": 10000.0,
            "leverage": 100,
            "magic_mm": 100002,
            "magic_sm": 100003,
            "magic_unit": 100004,
            "daily_limit_pct": 0.03,
            "total_limit_pct": 0.10,
            "components": {
                "data_feed": True,
                "spike_detection": True,
                "market_state_classifier": True,
                "level_break_classifier": True,
                "ml_decision_engine": True,
                "visualization": True,
                "journal": True,
                "signal_recorder": True
            },
            "strategies": {
                "mm_strategy": {"enabled": True},
                "sm_strategy": {"enabled": False},
                "unit_strategy": {"enabled": False}
            }
        }
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    cfg = yaml.safe_load(f)
                    if isinstance(cfg, dict):
                        defaults.update(cfg)
                        logger.info(f"Loaded trading config from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to read trading config: {e}. Using defaults.")
        self.config = defaults

    def bootstrap(self) -> bool:
        """
        Sequentially initializes all engines and sets up the active environment
        (live or simulated) in dependency order.
        """
        mode = self.config.get("trading_mode", "backtest").lower()
        symbols = self.config.get("symbols", ["EURUSD"])
        tf = self.config.get("timeframe", "M5")
        initial_balance = self.config.get("initial_balance", 10000.0)
        leverage = self.config.get("leverage", 100)

        logger.info(f"Bootstrapping Trading Framework in [{mode.upper()}] mode...")

        # 1. Setup Environment
        if mode in ["backtest", "simulation", "validation"]:
            # Set up Simulated Environment
            self.clock = SimulationClock(datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc))
            self.account = SimulationAccount(initial_balance, leverage)
            self.broker = SimulationBroker(self.account, self.clock)

            # Define standard properties for common symbols
            for sym in symbols:
                self.broker.set_symbol_info(sym, {
                    "digits": 5, "point": 0.00001, "volume_min": 0.01, "volume_step": 0.01, "volume_max": 100.0,
                    "trade_contract_size": 100000, "trade_stops_level": 0, "trade_tick_value": 1.0, "trade_tick_size": 0.00001
                })

            mt5.set_backtest_mode(self.broker, self.clock, self.account)

            # Setup Data Feed
            self.data_feed = HistoricalDataFeed()
            for sym in symbols:
                csv_path = PathManager.get_relative_path("historical_data", f"{sym}/{tf}.parquet")
                if not os.path.exists(csv_path):
                    csv_path = PathManager.get_relative_path("historical_data", f"{sym}/{tf}.csv")
                if os.path.exists(csv_path):
                    self.data_feed.load_csv(sym, tf, csv_path)
                else:
                    logger.warning(f"No raw historical file found for {sym} {tf} at {csv_path}!")

            # Set starting time
            timeline = self.data_feed.get_global_timeline()
            if timeline:
                self.clock.set_time(timeline[0])
                self.data_feed.seek_to_time(timeline[0])
                for sym in symbols:
                    bar = self.data_feed.get_current_bar(sym, tf)
                    if bar is not None:
                        self.broker.update_market_price(sym, bar['Close'], bar['Close'])

            self.modules["data_feed"] = self.data_feed
            logger.info("Simulated data feed initialized successfully.")
        else:
            # Live/Demo Mode
            mt5.set_live_mode()
            from Collecting_Data.data_feed import MT5DataFeed
            self.data_feed = MT5DataFeed()
            if not self.data_feed.connect():
                logger.error("Failed to connect live data feed to broker terminal!")
                return False
            self.modules["data_feed"] = self.data_feed
            logger.info("Live MT5 data feed connected successfully.")

        # 2. Build Sizers, Trackers, Journals, Exits in exact dependency order
        journal_root = "Validation_Journals" if mode == "validation" else "Journals"
        self.tj = TradingJournal(journal_root=journal_root, mode=mode)
        self.modules["journal"] = self.tj

        # Position Sizer and Tracker
        magic_numbers = [
            self.config.get("magic_mm", 100002),
            self.config.get("magic_sm", 100003),
            self.config.get("magic_unit", 100004)
        ]
        self.pm = PositionManager(magic_unity=magic_numbers[2], magic_mm=magic_numbers[0])
        self.pt = PositionTracker(
            magic_numbers=magic_numbers,
            poll_interval_seconds=0 if mode in ["backtest", "simulation"] else 5,
            state_file=os.path.join(journal_root, "position_tracker_state.json")
        )
        self.modules["position_tracker"] = self.pt

        # Drawdown Manager (acting as the RiskAdjuster)
        self.dm = DrawdownManager(
            initial_balance=initial_balance,
            position_tracker=self.pt,
            daily_limit_pct=self.config.get("daily_limit_pct", 0.03),
            total_limit_pct=self.config.get("total_limit_pct", 0.10),
            state_file=os.path.join(journal_root, "drawdown_state.json"),
            symbols=symbols
        )
        self.modules["drawdown_manager"] = self.dm

        self.ps = PositionSizer()
        self.modules["position_sizer"] = self.ps

        self.em = ExitManager(
            position_tracker=self.pt,
            position_manager=self.pm,
            trading_journal=self.tj,
            state_file=os.path.join(journal_root, "exit_manager_state.json")
        )
        self.modules["exit_manager"] = self.em

        # SendOrder
        self.so = SendOrder(
            position_manager=self.pm,
            position_tracker=self.pt,
            drawdown_manager=self.dm,
            position_sizer=self.ps,
            exit_manager=self.em,
            trading_journal=self.tj,
            state_file=os.path.join(journal_root, "send_order_state.json")
        )
        self.modules["send_order"] = self.so

        # Chart Annotation Visualization
        if self.config.get("components", {}).get("visualization", True):
            self.annotator = ChartAnnotationEngine()
            self.modules["annotator"] = self.annotator
            logger.info("Visualization engine linked successfully.")

        # 3. Instantiate Selected Strategy
        self.strategy = None
        strats = self.config.get("strategies", {})

        if strats.get("mm_strategy", {}).get("enabled", True):
            self.strategy = MMStrategy(
                data_feed=self.data_feed,
                send_order=self.so,
                trading_journal=self.tj,
                drawdown_manager=self.dm,
                symbols=symbols,
                poll_interval_seconds=0 if mode in ["backtest", "simulation"] else 5,
                state_file=os.path.join(journal_root, "mm_strategy_state.json"),
                annotator=getattr(self, "annotator", None)
            )
            logger.info("Loaded and enabled MM Strategy.")
        elif strats.get("sm_strategy", {}).get("enabled", False):
            self.strategy = SMStrategy(
                data_feed=self.data_feed,
                send_order=self.so,
                trading_journal=self.tj,
                drawdown_manager=self.dm,
                symbols=symbols,
                poll_interval_seconds=0 if mode in ["backtest", "simulation"] else 5,
                state_file=os.path.join(journal_root, "sm_strategy_state.json"),
                annotator=getattr(self, "annotator", None)
            )
            logger.info("Loaded and enabled SM Strategy (Ranging mean-reversion).")
        elif strats.get("unit_strategy", {}).get("enabled", False):
            self.strategy = UniTStrategy(
                data_feed=self.data_feed,
                send_order=self.so,
                trading_journal=self.tj,
                drawdown_manager=self.dm,
                symbols=symbols,
                poll_interval_seconds=0 if mode in ["backtest", "simulation"] else 5,
                state_file=os.path.join(journal_root, "unit_strategy_state.json"),
                annotator=getattr(self, "annotator", None)
            )
            logger.info("Loaded and enabled UniT Strategy.")

        if self.strategy is None:
            logger.error("No trading strategy enabled in configuration settings!")
            return False

        return True

    def run(self):
        """Starts background loops for live trading or sequentially steps for backtests."""
        mode = self.config.get("trading_mode", "backtest").lower()

        if mode in ["backtest", "simulation", "validation"]:
            logger.info("Starting sequential historical simulation loop...")
            timeline = self.data_feed.get_global_timeline()
            if not timeline:
                logger.error("No historical candles loaded for simulation!")
                return

            for current_time in timeline:
                self.clock.set_time(current_time)
                self.data_feed.seek_to_time(current_time)

                # Update prices
                tf = self.config.get("timeframe", "M5")
                for sym in self.config.get("symbols", ["EURUSD"]):
                    bar = self.data_feed.get_current_bar(sym, tf)
                    if bar is not None:
                        self.broker.update_market_price(sym, bar['Close'], bar['Close'])

                # Step cycles
                self.pt._poll_cycle()
                self.em._poll_cycle()
                self.dm.check()
                self.strategy._poll_cycle()

            logger.info("Historical simulation loop completed successfully.")

            # Generate and print performance statistics
            from trade_auditor import TradeAuditor
            from Simulation.statistics_engine import StatisticsEngine
            auditor = TradeAuditor(journal_root=self.tj.journal_root, mode="backtest")
            lifecycles = auditor.reconstruct_all()
            stats = StatisticsEngine(lifecycles)
            metrics = stats.calculate_metrics()
            logger.info(f"Simulation Win Rate: {metrics.get('win_rate', 0.0):.2%}")

        else:
            logger.info("Launching Live background polling threads...")
            self.pt.start()
            self.em.start()
            self.strategy.start()

            logger.info("Live Trading Pipeline is ACTIVE. Press Ctrl+C to stop.")
            try:
                while True:
                    self.dm.check()
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutdown signal intercepted. COMMENCING GRACEFUL SHUTDOWN...")
                self.shutdown()

    def shutdown(self):
        """Gracefully shuts down all active background threads and connections."""
        logger.info("Stopping all background services...")
        if "strategy" in self.modules:
            self.strategy.stop()
        if "exit_manager" in self.modules:
            self.em.stop()
        if "position_tracker" in self.modules:
            self.pt.stop()
        if "data_feed" in self.modules:
            self.data_feed.disconnect()
        mt5.shutdown()
        logger.info("Shutdown finalized. System offline.")
