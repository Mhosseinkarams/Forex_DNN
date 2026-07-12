"""
test_historical_data_collector.py
---------------------------------
Unit tests for historical data collection, validation, merging,
deduplication, resume functionality, and metadata generation.
"""

import os
import json
import shutil
import tempfile
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from Collecting_Data.historical_data_collector import (
    HistoricalDataCollector,
    MockDataProvider,
    TIMEFRAME_DURATIONS,
)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


def test_validate_chunk_integrity():
    """Test data validation engine inside HistoricalDataCollector."""
    # 1. Base clean data
    timestamps = pd.date_range(start="2025-01-01 00:00:00", periods=10, freq="5min", tz=timezone.utc)
    df = pd.DataFrame({
        "Datetime": timestamps,
        "Open": [1.1000] * 10,
        "High": [1.1010] * 10,
        "Low": [1.0990] * 10,
        "Close": [1.1000] * 10,
        "TickVolume": [100] * 10,
        "Spread": [1] * 10
    })

    # Validate clean data
    report = HistoricalDataCollector.validate_chunk(df, "M5")
    assert report["missing_timestamps"] == 0
    assert report["duplicates"] == 0
    assert report["invalid_ohlc"] == 0
    assert report["negative_volume"] == 0
    assert report["timezone_consistent"] is True
    assert len(report["errors"]) == 0

    # 2. Timezone-naive check
    df_naive = df.copy()
    df_naive["Datetime"] = df_naive["Datetime"].dt.tz_localize(None)
    report_naive = HistoricalDataCollector.validate_chunk(df_naive, "M5")
    assert report_naive["timezone_consistent"] is False
    assert any("timezone" in err.lower() for err in report_naive["errors"])

    # 3. Duplicate check
    df_dup = df.copy()
    df_dup.loc[1, "Datetime"] = df_dup.loc[0, "Datetime"] # create a duplicate
    report_dup = HistoricalDataCollector.validate_chunk(df_dup, "M5")
    assert report_dup["duplicates"] == 1
    assert any("duplicate" in err.lower() for err in report_dup["errors"])

    # 4. Invalid OHLC check
    df_invalid = df.copy()
    df_invalid.loc[2, "High"] = 1.0900 # High less than Low
    df_invalid.loc[3, "Open"] = -0.050 # Negative Open
    report_invalid = HistoricalDataCollector.validate_chunk(df_invalid, "M5")
    assert report_invalid["invalid_ohlc"] == 2
    assert any("invalid ohlc" in err.lower() for err in report_invalid["errors"])

    # 5. Negative Volume check
    df_neg_vol = df.copy()
    df_neg_vol.loc[4, "TickVolume"] = -10
    report_neg_vol = HistoricalDataCollector.validate_chunk(df_neg_vol, "M5")
    assert report_neg_vol["negative_volume"] == 1
    assert any("negative volume" in err.lower() for err in report_neg_vol["errors"])

    # 6. Missing timestamps check (Gap detection)
    df_gap = df.copy()
    # Remove index 5 to create a 10-minute gap instead of 5-minute
    df_gap = df_gap.drop(index=5).reset_index(drop=True)
    report_gap = HistoricalDataCollector.validate_chunk(df_gap, "M5")
    assert report_gap["missing_timestamps"] == 1
    assert any("missing candles" in err.lower() for err in report_gap["errors"])


def test_merging_and_duplicate_removal(temp_output_dir):
    """Test that downloaded chunks are correctly merged and duplicates are removed."""
    provider = MockDataProvider()
    collector = HistoricalDataCollector(provider=provider, output_dir=temp_output_dir, chunk_days=5)

    symbol = "EURUSD"
    timeframe = "M5"

    # Step 1: Download first chunk range (1/1 to 1/6)
    start_1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end_1 = datetime(2025, 1, 6, tzinfo=timezone.utc)
    res_1 = collector.download_symbol(symbol, timeframe, start_1, end_1)
    assert res_1["Status"] == "SUCCESS"

    final_pq_path = os.path.join(temp_output_dir, symbol, f"{timeframe}.parquet")
    assert os.path.exists(final_pq_path)
    df_1 = pd.read_parquet(final_pq_path)
    initial_len = len(df_1)

    # Step 2: Simulate another download with overlap by deleting progress state but keeping the file.
    # Start the second download from 1/3 (overlapping 1/3 to 1/6) up to 1/11.
    if f"{symbol}_{timeframe}" in collector.state:
        del collector.state[f"{symbol}_{timeframe}"]
        collector._save_state()

    start_2 = datetime(2025, 1, 3, tzinfo=timezone.utc)
    end_2 = datetime(2025, 1, 11, tzinfo=timezone.utc)

    res_2 = collector.download_symbol(symbol, timeframe, start_2, end_2)
    assert res_2["Status"] == "SUCCESS"

    # Step 3: Verify the merged final file
    df_merged = pd.read_parquet(final_pq_path)

    # Expected datetimes: from 1/1 to 1/11 (10 full days).
    # Since there are no duplicates, the timestamps should be strictly monotonic increasing.
    assert len(df_merged) > initial_len
    assert df_merged["Datetime"].is_monotonic_increasing
    assert not df_merged["Datetime"].duplicated().any()


def test_resume_functionality(temp_output_dir):
    """Test that Collector correctly pauses and resumes using state json."""
    provider = MockDataProvider()
    collector = HistoricalDataCollector(provider=provider, output_dir=temp_output_dir, chunk_days=5)

    symbol = "GBPUSD"
    timeframe = "M5"
    start_dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end_dt = datetime(2025, 1, 11, tzinfo=timezone.utc)

    # 1. Download first chunk range (1/1 to 1/6)
    # Let's save a state manually to simulate interrupt
    interrupted_dt = datetime(2025, 1, 6, tzinfo=timezone.utc)
    collector.update_state(symbol, timeframe, interrupted_dt, 1)

    # Verify resume info returns the interrupted date and chunk index
    resume_dt, chunk_idx = collector.get_resume_info(symbol, timeframe, start_dt)
    assert resume_dt == interrupted_dt
    assert chunk_idx == 1

    # Run downloader: it should start from 2025-01-06 up to 2025-01-11
    res = collector.download_symbol(symbol, timeframe, start_dt, end_dt)
    assert res["Status"] == "SUCCESS"

    # The resulting file should contain data starting from 2025-01-06 (or whatever is the start of the resumed chunk)
    final_pq_path = os.path.join(temp_output_dir, symbol, f"{timeframe}.parquet")
    df = pd.read_parquet(final_pq_path)
    assert df["Datetime"].min() == interrupted_dt


def test_metadata_generation(temp_output_dir):
    """Test metadata.json structure and contents."""
    provider = MockDataProvider()
    collector = HistoricalDataCollector(provider=provider, output_dir=temp_output_dir, chunk_days=10)

    symbol = "EURUSD"
    timeframe = "M5"
    start_dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end_dt = datetime(2025, 1, 11, tzinfo=timezone.utc)

    res = collector.download_symbol(symbol, timeframe, start_dt, end_dt)
    assert res["Status"] == "SUCCESS"

    metadata_path = os.path.join(temp_output_dir, symbol, "metadata.json")
    assert os.path.exists(metadata_path)

    with open(metadata_path, "r") as f:
        meta = json.load(f)

    assert meta["symbol"] == symbol
    assert meta["timeframe"] == timeframe
    assert "download_date" in meta
    assert "first_candle" in meta
    assert "last_candle" in meta
    assert meta["number_of_bars"] > 0
    assert meta["broker"] == "MockBroker"
    assert meta["timezone"] == "UTC"
