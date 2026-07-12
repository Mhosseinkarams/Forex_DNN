import unittest
import os
import shutil
import json
import tempfile
import pandas as pd
from datetime import datetime, timezone, timedelta
import sys
# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.getcwd(), '..')) if os.path.basename(os.getcwd()) == 'examples' else os.getcwd()
if project_root not in sys.path: sys.path.insert(0, project_root)

from Collecting_Data.historical_data_collector import (
    HistoricalDataCollector,
    MockDataProvider
)


class TestHistoricalDataCollector(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test storage
        self.test_dir = tempfile.mkdtemp()
        self.provider = MockDataProvider()
        self.collector = HistoricalDataCollector(
            provider=self.provider,
            output_dir=self.test_dir,
            format="both",
            chunk_size_days=10,
            max_workers=1
        )

    def tearDown(self):
        # Clean up the temporary directory after tests
        shutil.rmtree(self.test_dir)

    def test_merging_and_duplicate_removal(self):
        # Create mock chunks with overlap and unsorted timestamps
        base_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

        chunk1 = pd.DataFrame({
            "Datetime": [base_time + timedelta(minutes=5 * i) for i in range(5)],
            "Open": [1.1000 + 0.0001 * i for i in range(5)],
            "High": [1.1005 + 0.0001 * i for i in range(5)],
            "Low": [1.0995 + 0.0001 * i for i in range(5)],
            "Close": [1.1001 + 0.0001 * i for i in range(5)],
            "TickVolume": [100 + i for i in range(5)],
            "Spread": [2 for _ in range(5)]
        })

        # Overlapping chunk starting from index 3 of chunk1 (offset by 3)
        chunk2 = pd.DataFrame({
            "Datetime": [base_time + timedelta(minutes=5 * i) for i in range(3, 8)],
            "Open": [1.1000 + 0.0001 * i for i in range(3, 8)],
            "High": [1.1005 + 0.0001 * i for i in range(3, 8)],
            "Low": [1.0995 + 0.0001 * i for i in range(3, 8)],
            "Close": [1.1001 + 0.0001 * i for i in range(3, 8)],
            "TickVolume": [100 + i for i in range(3, 8)],
            "Spread": [2 for _ in range(3, 8)]
        })

        # Merge them
        merged = self.collector.merge_chunks([chunk2, chunk1])  # deliberately pass chunk2 first to check sorting

        # Assertions
        self.assertEqual(len(merged), 8)  # 0 to 7 (no duplicates)
        # Check chronological order
        self.assertTrue(merged["Datetime"].is_monotonic_increasing)
        # Check unique timestamps
        self.assertEqual(merged["Datetime"].nunique(), len(merged))

    def test_missing_candle_detection(self):
        # Continuous M5 series
        base_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        datetimes = [base_time + timedelta(minutes=5 * i) for i in range(10)]

        df = pd.DataFrame({
            "Datetime": datetimes,
            "Open": [1.1000] * 10,
            "High": [1.1005] * 10,
            "Low": [1.0995] * 10,
            "Close": [1.1001] * 10,
            "TickVolume": [100] * 10,
            "Spread": [2] * 10
        })

        # 1. Continuous should have 0 missing candles
        missing_count = self.collector.detect_missing_candles(df, "M5")
        self.assertEqual(missing_count, 0)

        # 2. Drop 3 candles from the middle
        # Indexes 4, 5, 6 dropped (indices remaining: 0, 1, 2, 3, 7, 8, 9)
        df_gap = df.drop([4, 5, 6]).reset_index(drop=True)
        missing_count_gap = self.collector.detect_missing_candles(df_gap, "M5")
        self.assertEqual(missing_count_gap, 3)

        # 3. Test weekend gap ignore: 52 hours gap should not report thousands of missing candles
        df_weekend = pd.DataFrame({
            "Datetime": [base_time, base_time + timedelta(hours=52)],
            "Open": [1.1000, 1.1000],
            "High": [1.1005, 1.1005],
            "Low": [1.0995, 1.0995],
            "Close": [1.1001, 1.1001],
            "TickVolume": [100, 100],
            "Spread": [2, 2]
        })
        missing_weekend = self.collector.detect_missing_candles(df_weekend, "M5")
        # With weekend subtraction heuristic, missing should be very small or 0
        self.assertLess(missing_weekend, 50)

    def test_chunk_downloading(self):
        # We query a 30-day range with a chunk_size_days of 10.
        # This should trigger exactly 3 chunk downloads.
        start_date = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
        end_date = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)

        # Count how many times fetch_chunk is called by wrapping/mocking
        original_fetch_chunk = self.provider.fetch_chunk
        calls = []

        def mock_fetch_chunk(symbol, timeframe, start_time, end_time):
            calls.append((start_time, end_time))
            return original_fetch_chunk(symbol, timeframe, start_time, end_time)

        self.provider.fetch_chunk = mock_fetch_chunk

        res = self.collector.download_symbol("EURUSD", "M5", start_date, end_date)

        self.assertEqual(len(calls), 3)  # 3 chunks
        self.assertEqual(res["status"], "Success")
        self.assertGreater(res["bars"], 0)

    def test_resume_functionality(self):
        # Pre-seed pre-existing data on disk to verify resume does not lose historic data
        symbol_dir = os.path.join(self.test_dir, "EURUSD")
        os.makedirs(symbol_dir, exist_ok=True)
        csv_path = os.path.join(symbol_dir, "M5.csv")
        parquet_path = os.path.join(symbol_dir, "M5.parquet")

        historic_time = datetime(2026, 6, 10, 0, 0, tzinfo=timezone.utc)
        historic_df = pd.DataFrame({
            "Datetime": [historic_time],
            "Open": [1.0900],
            "High": [1.0950],
            "Low": [1.0850],
            "Close": [1.0910],
            "TickVolume": [50],
            "Spread": [2]
        })
        historic_df.to_csv(csv_path, index=False)
        historic_df.to_parquet(parquet_path, index=False)

        # Pre-seed the state file indicating partial completion of a symbol
        state_key = "EURUSD_M5"
        resume_time = datetime(2026, 6, 15, 0, 0, tzinfo=timezone.utc)

        self.collector.state[state_key] = {
            "symbol": "EURUSD",
            "timeframe": "M5",
            "last_downloaded_datetime": resume_time.isoformat(),
            "chunk_index": 2,
            "status": "in_progress"
        }
        self.collector._save_state()

        # Download from 2026-06-01 to 2026-07-01
        start_date = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
        end_date = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)

        calls = []
        def mock_fetch_chunk(symbol, timeframe, start_time, end_time):
            calls.append((start_time, end_time))
            # Return an empty but valid DataFrame to speed up
            return pd.DataFrame({
                "Datetime": [start_time, end_time],
                "Open": [1.1000, 1.1000],
                "High": [1.1005, 1.1005],
                "Low": [1.0995, 1.0995],
                "Close": [1.1001, 1.1001],
                "TickVolume": [100, 100],
                "Spread": [2, 2]
            })

        self.provider.fetch_chunk = mock_fetch_chunk

        res = self.collector.download_symbol("EURUSD", "M5", start_date, end_date)

        # Should start from 2026-06-15 instead of 2026-06-01!
        # First call's start_time must be resume_time
        self.assertEqual(calls[0][0], resume_time)
        self.assertEqual(res["status"], "Success")

        # Verify that original pre-seeded data was retained and merged with new chunks
        saved_df = pd.read_parquet(parquet_path)
        datetimes_list = saved_df["Datetime"].tolist()
        self.assertIn(historic_time, datetimes_list)
        self.assertIn(resume_time, datetimes_list)
        self.assertGreater(len(saved_df), 1)

    def test_metadata_generation(self):
        start_date = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
        end_date = datetime(2026, 6, 5, 0, 0, tzinfo=timezone.utc)

        self.collector.download_symbol("EURUSD", "M5", start_date, end_date)

        # Check metadata.json file exists
        metadata_path = os.path.join(self.test_dir, "EURUSD", "metadata.json")
        self.assertTrue(os.path.exists(metadata_path))

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.assertEqual(metadata["symbol"], "EURUSD")
        self.assertEqual(metadata["timeframe"], "M5")
        self.assertEqual(metadata["broker"], "MockBroker")
        self.assertIn("number_of_bars", metadata)
        self.assertIn("download_date", metadata)


class TestConsoleProgressMonitor(unittest.TestCase):
    def test_worker_registration(self):
        from Collecting_Data.historical_data_collector import ConsoleProgressMonitor
        monitor = ConsoleProgressMonitor(num_workers=4, enabled=False)

        slot1 = monitor.register_worker("thread-a")
        slot2 = monitor.register_worker("thread-b")
        slot1_again = monitor.register_worker("thread-a")

        self.assertEqual(slot1, 1)
        self.assertEqual(slot2, 2)
        self.assertEqual(slot1_again, 1)

    def test_progress_updates(self):
        from Collecting_Data.historical_data_collector import ConsoleProgressMonitor
        monitor = ConsoleProgressMonitor(num_workers=2, enabled=False)

        # Test updating slot
        monitor.update(1, "EURUSD", "M5", 2, 5, "Downloading")
        self.assertEqual(monitor.slots[1]["symbol"], "EURUSD")
        self.assertEqual(monitor.slots[1]["chunk"], 2)
        self.assertEqual(monitor.slots[1]["total"], 5)
        self.assertEqual(monitor.slots[1]["status"], "Downloading")

    def test_drawing_and_clearing_safeguards(self):
        from Collecting_Data.historical_data_collector import ConsoleProgressMonitor
        monitor = ConsoleProgressMonitor(num_workers=2, enabled=True)
        # Force TTY to True to execute full print escape-paths under test
        monitor.enabled = True

        # Should execute draw cleanly without crash
        monitor.update(1, "GBPUSD", "M15", 1, 3, "Running")
        monitor.update(2, "EURUSD", "M5", 0, 3, "Starting")

        # Test clear/redraw cleanly
        monitor.clear()
        monitor.redraw()
        monitor.clear()

    def test_logging_handler_integration(self):
        import logging
        from Collecting_Data.historical_data_collector import ConsoleProgressMonitor, ProgressAwareStreamHandler

        monitor = ConsoleProgressMonitor(num_workers=1, enabled=True)
        monitor.enabled = True
        monitor.active_lines_printed = True

        import io
        stream = io.StringIO()
        handler = ProgressAwareStreamHandler(monitor=monitor, stream=stream)

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test_path.py",
            lineno=10,
            msg="Progress-aware log works!",
            args=(),
            exc_info=None
        )

        handler.emit(record)
        output = stream.getvalue()
        self.assertIn("Progress-aware log works!", output)


if __name__ == "__main__":
    unittest.main()
