import unittest
import pandas as pd
import json
from datetime import datetime, timezone
from Collecting_Data.position_lifecycle import (
    PositionLifecycle,
    PositionLifecycleBuilder,
    SignalInfo,
    ExecutionInfo,
    ManagementInfo,
    OutcomeInfo
)

class TestPositionLifecycle(unittest.TestCase):
    def setUp(self):
        self.signal_id = "test-signal-123"
        self.journal_data = [
            {
                "signal_id": self.signal_id,
                "event_type": "signal",
                "system_timestamp": "2023-10-27T10:00:00Z",
                "bar_timestamp": "2023-10-27T10:00:00Z",
                "strategy": "MMStrategy",
                "symbol": "EURUSD_o",
                "timeframe": "M5",
                "direction": 1,
                "entry_price": 1.1000,
                "sl_price": 1.0950,
                "signal_category": "standard",
                "extra_fields": "{'rsi': 30}"
            },
            {
                "signal_id": self.signal_id,
                "event_type": "order_open",
                "system_timestamp": "2023-10-27T10:00:01Z",
                "ticket": 123456,
                "actual_entry": 1.1001,
                "actual_sl": 1.0950,
                "actual_tp": 1.1100,
                "lot_size": 0.1,
                "risk_pct": 1.0
            },
            {
                "signal_id": self.signal_id,
                "event_type": "outcome",
                "system_timestamp": "2023-10-27T11:00:00Z",
                "close_price": 1.1050,
                "pnl_dollars": 50.0,
                "duration_seconds": 3600,
                "outcome": "tp1"
            }
        ]
        self.df = pd.DataFrame(self.journal_data)

    def test_builder_basic(self):
        lifecycle = PositionLifecycleBuilder.build_from_data(self.signal_id, self.df)
        self.assertIsNotNone(lifecycle)
        self.assertEqual(lifecycle.signal.signal_id, self.signal_id)
        self.assertEqual(lifecycle.execution.ticket, 123456)
        self.assertEqual(lifecycle.outcome.realized_profit, 50.0)
        self.assertEqual(lifecycle.outcome.status, "completed")

    def test_builder_open_position(self):
        # Remove outcome event
        df_open = self.df[self.df['event_type'] != 'outcome']
        lifecycle = PositionLifecycleBuilder.build_from_data(self.signal_id, df_open)
        self.assertIsNotNone(lifecycle)
        self.assertEqual(lifecycle.outcome.status, "open")
        self.assertEqual(lifecycle.outcome.realized_profit, 0.0)

    def test_serialization(self):
        lifecycle = PositionLifecycleBuilder.build_from_data(self.signal_id, self.df)

        # Dictionary
        d = lifecycle.to_dict()
        self.assertIn('signal', d)
        self.assertEqual(d['signal']['signal_id'], self.signal_id)

        # JSON
        j = lifecycle.to_json()
        data = json.loads(j)
        self.assertEqual(data['execution']['ticket'], 123456)

        # CSV Row (flattened)
        csv_row = lifecycle.to_csv_row()
        # Builder to_csv_row only flattens dicts currently, but dataclasses should be dicts in asdict
        # Actually I need to check my implementation of to_csv_row

        # Markdown
        md = lifecycle.to_markdown()
        self.assertIn("# Position Lifecycle - Ticket 123456", md)
        self.assertIn("**Signal ID:** `test-signal-123`", md)

    def test_broker_enrichment(self):
        broker_data = {
            "deals": [
                {"ticket": 999, "entry": 0, "price": 1.1002, "magic": 777, "time": 1698393600},
                {"ticket": 1000, "entry": 1, "price": 1.1055, "profit": 53.0, "reason": 5, "time": 1698397200}
            ]
        }
        lifecycle = PositionLifecycleBuilder.build_from_data(self.signal_id, self.df, broker_data=broker_data)
        self.assertEqual(lifecycle.execution.actual_entry, 1.1002)
        self.assertEqual(lifecycle.execution.magic_number, 777)
        self.assertEqual(lifecycle.outcome.realized_profit, 53.0)
        self.assertEqual(lifecycle.outcome.broker_reason, "5")

if __name__ == '__main__':
    unittest.main()
