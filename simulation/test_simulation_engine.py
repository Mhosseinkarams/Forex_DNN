import unittest
import os
import shutil
import pandas as pd
from datetime import datetime, timezone
from simulation.simulation_runner import SimulationRunner
from trade_auditor import TradeAuditor

class TestSimulationEngine(unittest.TestCase):
    def setUp(self):
        self.symbol = "TEST_o"
        self.journal_root = "Test_Backtest_Journals"
        if os.path.exists(self.journal_root):
            shutil.rmtree(self.journal_root)

        # Create dummy data: ascending trend, no pullbacks
        data = {
            "Datetime": pd.date_range("2024-01-01", periods=200, freq="5min"),
            "Open":  [1.1000 + i*0.0001 for i in range(200)],
            "High":  [1.1005 + i*0.0001 for i in range(200)],
            "Low":   [1.0998 + i*0.0001 for i in range(200)], # Higher Low to avoid SL
            "Close": [1.1000 + i*0.0001 for i in range(200)],
            "TickVolume": [100]*200,
            "Spread": [1]*200
        }
        self.csv_path = "test_data.csv"
        pd.DataFrame(data).to_csv(self.csv_path, index=False)

    def tearDown(self):
        pass

    def test_simulation_run(self):
        runner = SimulationRunner(
            symbol=self.symbol,
            timeframes=["M5"],
            data_files={(self.symbol, "M5"): self.csv_path},
            journal_root=self.journal_root
        )

        # Manually trigger a trade
        runner.clock.set_time(datetime(2024, 1, 1, 0, 30, tzinfo=timezone.utc))
        runner.broker.update_market_price(self.symbol, 1.1006, 1.1006)

        sid = runner.tj.log_signal(
            signal_type="standard",
            symbol=self.symbol,
            timeframe="M5",
            direction=1,
            entry_price=1.1006,
            sl_price=1.1000,
            exit_profile="single",
            strategy="mm",
            signal_category="standard",
            bar_timestamp=str(runner.clock.current_time()),
            tp_level=1
        )

        res = runner.so.execute(
            symbol=self.symbol,
            direction=1,
            entry_price=0.0,
            sl_price=1.1000,
            exit_profile="single",
            strategy="mm",
            signal_category="standard",
            signal_id=sid
        )

        self.assertTrue(res["success"])
        ticket = res["ticket"]

        runner.run()

        auditor = TradeAuditor(journal_root=self.journal_root, mode="backtest")
        lifecycles = auditor.reconstruct_all()

        self.assertGreaterEqual(len(lifecycles), 1)
        lc = lifecycles[0]
        self.assertEqual(lc.execution.ticket, ticket)
        self.assertEqual(lc.outcome.status, "completed")
        self.assertIn("tp", lc.outcome.strategy_reason.lower())

if __name__ == "__main__":
    unittest.main()
