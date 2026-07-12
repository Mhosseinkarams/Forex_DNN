import argparse
import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# Add parent directory to path so imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML.label_engine import LabelEngine
from ML.dataset_validator import DatasetValidator
from Market_Data_Pipeline.structure_engine import MarketStructureEngine
from Market_Data_Pipeline.supply_demand_engine import SupplyDemandEngine

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("GenerateDatasetCLI")


def generate_synthetic_candles(num_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generates a deterministic synthetic DataFrame of OHLCV candles with basic EMA/ATR.
    Used for local testing, CI, and demos when live MT5 data is not present.
    """
    logger.info(f"Generating {num_bars} deterministic synthetic candles...")
    np.random.seed(seed)

    start_date = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    datetimes = [start_date + timedelta(minutes=15 * i) for i in range(num_bars)]

    # Simulate a trending/ranging random walk
    prices = np.zeros(num_bars)
    prices[0] = 1.1000
    for i in range(1, num_bars):
        # Every 100 bars, change regime
        regime = (i // 120) % 3
        if regime == 0:
            # Bull Trend
            change = np.random.normal(0.0003, 0.0002)
        elif regime == 1:
            # Range
            change = np.random.normal(0.0, 0.0003)
        else:
            # Bear Trend
            change = np.random.normal(-0.0003, 0.0002)
        prices[i] = prices[i-1] + change

    # Ensure no negative prices
    prices = np.clip(prices, 0.5000, 2.5000)

    # Build High, Low, Open, Close, Volume
    opens = np.zeros(num_bars)
    highs = np.zeros(num_bars)
    lows = np.zeros(num_bars)
    closes = np.zeros(num_bars)
    volumes = np.random.randint(100, 1500, size=num_bars).astype(float)
    spreads = np.random.randint(1, 5, size=num_bars).astype(float)

    for i in range(num_bars):
        if i == 0:
            opens[i] = prices[i]
        else:
            opens[i] = closes[i-1]
        closes[i] = prices[i]

        # Ensure standard candle relationships
        highs[i] = max(opens[i], closes[i]) + np.abs(np.random.normal(0.0005, 0.0002))
        lows[i] = min(opens[i], closes[i]) - np.abs(np.random.normal(0.0005, 0.0002))

    df = pd.DataFrame({
        "Datetime": datetimes,
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "TickVolume": volumes,
        "Spread": spreads
    })

    # Pre-calculate Indicators to ensure feature extraction is fully supported
    df["ema_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["ema_600"] = df["Close"].ewm(span=600, adjust=False).mean()

    # Basic ATR
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(window=14).mean().fillna(0.0001)

    return df


def run_pipeline(args):
    """
    Orchestrates the entire data preparation, labeling, validation, and serialization pipeline.
    """
    logger.info("Starting Dataset Generation Pipeline...")

    # Step 1: Load/Generate historical candles
    df_ohlcv = None
    if args.input_csv:
        if os.path.exists(args.input_csv):
            logger.info(f"Loading input CSV from {args.input_csv}")
            df_ohlcv = pd.read_csv(args.input_csv)
            # Parse datetime
            if "Datetime" in df_ohlcv.columns:
                df_ohlcv["Datetime"] = pd.to_datetime(df_ohlcv["Datetime"])
        else:
            logger.error(f"Input CSV not found at: {args.input_csv}")
            sys.exit(1)
    else:
        # Fallback to deterministic synthetic generation
        df_ohlcv = generate_synthetic_candles(num_bars=args.num_bars, seed=args.seed)

    # Step 2: Initialize Engines
    ms_engine = MarketStructureEngine(lookback=3)
    sd_engine = SupplyDemandEngine(atr_period=14, impulse_threshold=2.0)

    engine = LabelEngine(
        window_size=args.window_size,
        window_stride=args.window_stride
    )

    # Step 3: Run Sliding Window labeling and feature extraction
    df_labeled = engine.generate_dataset(
        data_inputs=df_ohlcv,
        symbol=args.symbol,
        timeframe=args.timeframe,
        ms_engine=ms_engine,
        sd_engine=sd_engine
    )

    # Step 4: Validate generated dataset
    logger.info("Performing dataset validation checks...")
    validator = DatasetValidator()
    report = validator.validate(df_labeled, expected_window_size=args.window_size)

    # Log validation summary
    logger.info(f"Validation Report - Is Valid: {report['is_valid']}")
    logger.info(f"Total rows: {report['metrics']['total_samples']}, Columns: {report['metrics']['columns_count']}")
    logger.info(f"Class distribution: {report['metrics']['class_distribution']}")

    for warning in report["warnings"]:
        logger.warning(f"Validation Warning: {warning}")
    for error in report["errors"]:
        logger.error(f"Validation Error: {error}")

    # Step 5: Save dataset & reproducibility manifest
    logger.info(f"Saving outputs...")
    extra_metadata = {
        "pipeline_run_datetime": datetime.now(timezone.utc).isoformat(),
        "input_dataset_source": args.input_csv if args.input_csv else "deterministic_synthetic",
        "validation_is_valid": report["is_valid"],
        "validation_warnings_count": len(report["warnings"]),
        "validation_errors_count": len(report["errors"]),
    }

    engine.save_dataset_and_manifest(
        df=df_labeled,
        output_path=args.output_csv,
        manifest_path=args.output_manifest,
        extra_metadata=extra_metadata
    )

    logger.info("Pipeline run complete!")
    if not report["is_valid"]:
        logger.error("WARNING: The generated dataset failed critical validation rules! Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic Sliding-Window Dataset Generator CLI")
    parser.add_argument("--input_csv", type=str, default="", help="Path to input CSV containing historical OHLCV data.")
    parser.add_argument("--output_csv", type=str, default="output/market_state_dataset.csv", help="Path to write the labeled output CSV.")
    parser.add_argument("--output_manifest", type=str, default="output/market_state_dataset_manifest.json", help="Path to write the JSON manifest.")
    parser.add_argument("--window_size", type=int, default=35, help="Size of the sliding window (default: 35).")
    parser.add_argument("--window_stride", type=int, default=1, help="Stride of the sliding window (default: 1).")
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Symbol identifier for the metadata columns.")
    parser.add_argument("--timeframe", type=str, default="M15", help="Timeframe identifier for the metadata columns.")
    parser.add_argument("--num_bars", type=int, default=500, help="Number of bars to generate if using synthetic data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic synthetic data generation.")

    parsed_args = parser.parse_args()
    run_pipeline(parsed_args)
