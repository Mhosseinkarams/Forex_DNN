import logging
import os
import pandas as pd
from datetime import timedelta
from simulation.simulation_environment import env
from simulation.statistics_engine import StatisticsEngine
from simulation.backtest_report import BacktestReport
from Collecting_Data.position_lifecycle import PositionLifecycleBuilder
from trade_auditor import TradeAuditor

logger = logging.getLogger("SimulationRunner")

class SimulationRunner:
    def __init__(self, environment, strategy, tracker, exit_manager, journal):
        self.env = environment
        self.strategy = strategy
        self.tracker = tracker
        self.em = exit_manager
        self.tj = journal

    def run(self, symbol: str, timeframe: str, step_minutes: int = 5):
        logger.info(f"Starting Simulation for {symbol} {timeframe}")

        # We need to manually drive the poll cycles since we don't have background threads in backtest
        # strategy._poll_cycle()
        # tracker._poll_cycle()
        # em._poll_cycle()

        while not self.env.data_feed.is_finished(symbol, timeframe):
            # 1. Advance Clock
            current_time = self.env.get_now()
            self.env.clock.advance(timedelta(minutes=step_minutes))

            # 2. Advance Data Feed
            self.env.data_feed.advance(symbol, timeframe)

            # 3. Drive Broker Price Updates (simulated tick at candle end)
            bar = self.env.data_feed.get_current_bar(symbol, timeframe)
            if bar is not None:
                # Use Close as Bid/Ask for Phase 1
                self.env.broker.update_market_price(
                    symbol=symbol,
                    bid=bar["Close"],
                    ask=bar["Close"],
                    time_ts=bar["Datetime"].timestamp()
                )

            # 4. Update Drawdown Manager
            self.strategy.drawdown_manager.check()

            # 5. Run Strategy Poll
            self.strategy._poll_cycle()

            # 6. Run Tracker Poll
            self.tracker._poll_cycle()

            # 7. Run Exit Manager Poll
            self.em._poll_cycle()

        logger.info("Simulation Finished.")

        # Finalize
        return self.generate_results(symbol, timeframe)

    def generate_results(self, symbol, timeframe):
        # Load all lifecycles from journal
        auditor = TradeAuditor(journal_root=self.tj.journal_root, mode=self.tj.mode)
        journal_df = auditor.load_journal_data()

        # We need to find all signal_ids
        if journal_df.empty:
            logger.warning("No trades recorded during simulation.")
            return None

        signal_ids = journal_df['signal_id'].unique()
        lifecycles = []

        for sid in signal_ids:
            # We don't have real-time access to state files here easily if they weren't saved,
            # but in backtest, we can pass them or just rely on journal + broker deals.
            broker_data = {"deals": self.env.broker.get_history_deals(sid)} # Wait, deals are by position ticket
            # This needs a bit of fixup in how we retrieve lifecycles
            pass

        # For Phase 1, let's just use the TradeAuditor's ability to reconstruct
        # Actually, let's use the ones that ExitManager should have already logged.

        # Read from the lifecycle CSV
        filepath = self.tj._get_filepath(self.strategy.strategy_name if hasattr(self.strategy, 'strategy_name') else 'mm', symbol, timeframe, "lifecycle")
        if os.path.exists(filepath):
            df_lc = pd.read_csv(filepath)
            # Reconstruct dummy lifecycles for StatisticsEngine or just make StatisticsEngine accept DF
            # Let's make a simple one
            stats = StatisticsEngine.calculate_metrics_from_df(df_lc, self.env.account.initial_balance)
            return stats

        return None

# Adding a helper to StatisticsEngine
