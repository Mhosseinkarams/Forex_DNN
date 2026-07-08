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
        symbol: str,
        timeframes: list[str],
        data_files: dict,
        initial_balance: float = 10000.0,
        leverage: int = 100,
        journal_root: str = "Backtest_Journals"
    ):
        self.symbol = symbol
        self.timeframes = timeframes
        self.data_files = data_files
        self.initial_balance = initial_balance
        self.journal_root = journal_root

        self.data_feed = HistoricalDataFeed()
        for (s, tf), path in data_files.items():
            self.data_feed.load_csv(s, tf, path)

        start_time = self.data_feed.get_current_bar(symbol, timeframes[0])['Datetime']
        self.clock = SimulationClock(start_time)
        self.account = SimulationAccount(initial_balance, leverage)
        self.broker = SimulationBroker(self.account, self.clock)

        self.broker.set_symbol_info(symbol, {
            "digits": 5,
            "point": 0.00001,
            "volume_min": 0.01,
            "volume_step": 0.01,
            "volume_max": 100.0,
            "trade_contract_size": 100000,
            "trade_stops_level": 0
        })

        env.set_backtest_mode(self.broker, self.clock, self.account)

        bar = self.data_feed.get_current_bar(symbol, timeframes[0])
        self.broker.update_market_price(symbol, bar['Close'], bar['Close'])

        state_dir = os.path.join(journal_root, "State")
        os.makedirs(state_dir, exist_ok=True)

        self.tj = TradingJournal(journal_root=journal_root, mode="backtest")
        self.pm = PositionManager(magic_unity=100001, magic_mm=100002)
        self.pt = PositionTracker(magic_numbers=[100001, 100002], poll_interval_seconds=0, state_file=os.path.join(state_dir, "position_state.json"))
        self.dm = DrawdownManager(initial_balance=initial_balance, position_tracker=self.pt, state_file=os.path.join(state_dir, "drawdown_state.json"))
        self.ps = PositionSizer()
        self.em = ExitManager(position_tracker=self.pt, position_manager=self.pm, trading_journal=self.tj, state_file=os.path.join(state_dir, "exit_manager_state.json"))

        from PositionManager.send_order import SendOrder
        self.so = SendOrder(self.pm, self.pt, self.dm, self.ps, self.em, self.tj, state_file=os.path.join(state_dir, "send_order_state.json"))

        self.strategy = MMStrategy(
            data_feed=self.data_feed,
            send_order=self.so,
            trading_journal=self.tj,
            drawdown_manager=self.dm,
            symbols=[symbol],
            poll_interval_seconds=0
        )

    def run(self):
        logger.info(f"Starting simulation for {self.symbol}...")

        while not self.data_feed.is_finished():
            bar = self.data_feed.get_current_bar(self.symbol, self.timeframes[0])
            self.clock.set_time(bar['Datetime'])
            self.broker.update_market_price(self.symbol, bar['Close'], bar['Close'])

            self.pt._poll_cycle()
            self.em._poll_cycle()
            self.dm.check()
            self.strategy._poll_cycle()

            self.data_feed.advance()

        logger.info("Simulation finished.")
        self.generate_report()

    def generate_report(self):
        from trade_auditor import TradeAuditor
        # Re-ensure backtest mode in environment for reconstruction
        env.set_backtest_mode(self.broker, self.clock, self.account)
        auditor = TradeAuditor(journal_root=self.journal_root, mode="backtest")
        lifecycles = auditor.reconstruct_all()

        stats = StatisticsEngine(lifecycles)
        metrics = stats.calculate_metrics()

        report = BacktestReport.generate_summary("MMStrategy", self.symbol, self.timeframes[0], metrics)
        print(report)

        with open(os.path.join(self.journal_root, "backtest_report.txt"), "w") as f:
            f.write(report)
