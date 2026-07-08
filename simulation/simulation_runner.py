import logging
import os
import pandas as pd
from datetime import datetime, timedelta

from simulation.simulation_clock import SimulationClock
from simulation.simulation_account import SimulationAccount
from simulation.simulation_broker import SimulationBroker
from simulation.simulation_environment import env

from Collecting_Data.trading_journal import TradingJournal
from PositionManager.position_manager import PositionManager
from PositionManager.position_tracker import PositionTracker
from PositionManager.drawdown import DrawdownManager
from PositionManager.risk_sizing import PositionSizer
from PositionManager.exit_manager import ExitManager
from Strategies.mm_strategy import MMStrategy
from simulation.historical_data_feed import HistoricalDataFeed
from simulation.simulation_order_engine import SimulationOrderEngine
from simulation.statistics_engine import StatisticsEngine
from simulation.backtest_report import BacktestReport

logger = logging.getLogger("SimulationRunner")

class SimulationRunner:
    def __init__(
        self,
        symbols: list[str] | str,
        timeframes: list[str],
        data_files: dict,
        initial_balance: float = 10000.0,
        leverage: int = 100,
        journal_root: str = "Backtest_Journals"
    ):
        if isinstance(symbols, str):
            self.symbols = [symbols]
        else:
            self.symbols = symbols

        self.timeframes = timeframes
        self.data_files = data_files
        self.initial_balance = initial_balance
        self.journal_root = journal_root

        self.data_feed = HistoricalDataFeed()
        for (s, tf), path in data_files.items():
            self.data_feed.load_csv(s, tf, path)

        # Initialize clock to the earliest bar available in any dataset
        timeline = self.data_feed.get_global_timeline()
        if not timeline:
            raise ValueError("No data loaded in HistoricalDataFeed")

        start_time = timeline[0]
        self.clock = SimulationClock(start_time)
        self.account = SimulationAccount(initial_balance, leverage)
        self.broker = SimulationBroker(self.account, self.clock)

        for s in self.symbols:
            self.broker.set_symbol_info(s, self._get_default_symbol_info(s))

        env.set_backtest_mode(self.broker, self.clock, self.account)

        # Initial price update
        self.data_feed.seek_to_time(start_time)
        for s in self.symbols:
            # Try all timeframes to get an initial price
            for tf in self.timeframes:
                bar = self.data_feed.get_current_bar(s, tf)
                if bar is not None:
                    self.broker.update_market_price(s, bar['Close'], bar['Close'])
                    break

        state_dir = os.path.join(journal_root, "State")

        # Clean stale state for a fresh backtest run
        if os.path.exists(state_dir):
            import shutil
            try:
                shutil.rmtree(state_dir)
                logger.info(f"Cleared stale state directory: {state_dir}")
            except Exception as e:
                logger.warning(f"Failed to clear state directory: {e}")

        os.makedirs(state_dir, exist_ok=True)

        self.tj = TradingJournal(journal_root=journal_root, mode="backtest")
        self.pm = PositionManager(magic_unity=100001, magic_mm=100002)
        self.pt = PositionTracker(magic_numbers=[100001, 100002], poll_interval_seconds=0, state_file=os.path.join(state_dir, "position_tracker_state.json"))
        self.dm = DrawdownManager(initial_balance=initial_balance, position_tracker=self.pt, state_file=os.path.join(state_dir, "drawdown_manager_state.json"))
        self.ps = PositionSizer()
        self.em = ExitManager(position_tracker=self.pt, position_manager=self.pm, trading_journal=self.tj, state_file=os.path.join(state_dir, "exit_manager_state.json"))

        from PositionManager.send_order import SendOrder
        self.so = SendOrder(self.pm, self.pt, self.dm, self.ps, self.em, self.tj, state_file=os.path.join(state_dir, "send_order_state.json"))

        self.strategy = MMStrategy(
            data_feed=self.data_feed,
            send_order=self.so,
            trading_journal=self.tj,
            drawdown_manager=self.dm,
            symbols=self.symbols,
            poll_interval_seconds=0,
            state_file=os.path.join(state_dir, "mm_strategy_state.json")
        )

    def run(self):
        logger.info(f"Starting multi-symbol simulation for {self.symbols}...")

        timeline = self.data_feed.get_global_timeline()

        for current_time in timeline:
            self.clock.set_time(current_time)
            self.data_feed.seek_to_time(current_time)

            # Update market prices for all symbols at this timestamp
            for s in self.symbols:
                # Try all available timeframes to ensure we have the most recent price
                for tf in self.timeframes:
                    bar = self.data_feed.get_current_bar(s, tf)
                    if bar is not None:
                        self.broker.update_market_price(s, bar['Close'], bar['Close'])
                        break

            # Poll components
            self.pt._poll_cycle()
            self.em._poll_cycle()
            self.dm.check()
            self.strategy._poll_cycle()

        logger.info("Simulation finished.")
        self.generate_report()

    def _get_default_symbol_info(self, symbol: str) -> dict:
        """Returns standard broker properties for common symbols to improve simulation accuracy."""
        info = {
            "digits": 5,
            "point": 0.00001,
            "volume_min": 0.01,
            "volume_step": 0.01,
            "volume_max": 100.0,
            "trade_contract_size": 100000,
            "trade_stops_level": 0
        }

        s_up = symbol.upper()
        if "JPY" in s_up:
            info["digits"] = 3
            info["point"] = 0.001
        elif "XAU" in s_up or "GOLD" in s_up:
            info["digits"] = 2
            info["point"] = 0.01
            info["trade_contract_size"] = 100
        elif "YM" in s_up or "DJI" in s_up:
            info["digits"] = 2
            info["point"] = 1.0
            info["trade_contract_size"] = 1
        elif "DAX" in s_up or "DE30" in s_up or "FDAX" in s_up:
            info["digits"] = 2
            info["point"] = 1.0
            info["trade_contract_size"] = 1

        return info

    def generate_report(self):
        from trade_auditor import TradeAuditor
        # Re-ensure backtest mode in environment for reconstruction
        env.set_backtest_mode(self.broker, self.clock, self.account)
        auditor = TradeAuditor(journal_root=self.journal_root, mode="backtest")
        lifecycles = auditor.reconstruct_all()

        stats = StatisticsEngine(lifecycles)
        metrics = stats.calculate_metrics()

        symbol_label = ", ".join(self.symbols) if len(self.symbols) < 4 else f"{len(self.symbols)} symbols"
        report = BacktestReport.generate_summary("MMStrategy", symbol_label, self.timeframes[0], metrics)
        print(report)

        with open(os.path.join(self.journal_root, "backtest_report.txt"), "w") as f:
            f.write(report)
