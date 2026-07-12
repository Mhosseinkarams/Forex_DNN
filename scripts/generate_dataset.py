import os
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from ML.label_engine import LabelEngine
from ML.market_state_labeler import MarketStateLabeler
from ML.dataset_validator import DatasetValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("GenerateDataset")


def generate_synthetic_ohlcv(n_bars: int = 1000, start_price: float = 1.1000) -> pd.DataFrame:
    """
    Generates realistic, deterministic synthetic OHLCV data with clear trends,
    ranging periods, and volatility structures to serve as a high-quality testbed.
    """
    logger.info(f"Generating {n_bars} bars of high-quality synthetic OHLCV data...")
    np.random.seed(42)  # Reproducibility

    times = [datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * i) for i in range(n_bars)]

    # We will simulate three distinct market regimes dynamically scaled to n_bars:
    # 1. First 40% of bars: Trending up
    # 2. Next 30% of bars: Ranging sideways (mean-reverting)
    # 3. Remaining bars: Trending down

    prices = np.zeros(n_bars)
    prices[0] = start_price

    part1 = int(n_bars * 0.4)
    part2 = int(n_bars * 0.7)

    # 1. Bull trend
    for i in range(1, part1):
        # Small upward drift with some noise
        prices[i] = prices[i-1] + 0.00015 + np.random.normal(0, 0.0001)

    # 2. Range
    range_mid = prices[part1 - 1] if part1 > 0 else start_price
    for i in range(part1, part2):
        # Mean reverting to range_mid
        prices[i] = range_mid + np.random.normal(0, 0.0005)
        # Smooth transitions
        if i > 0:
            prices[i] = 0.9 * prices[i-1] + 0.1 * prices[i]

    # 3. Bear trend
    for i in range(part2, n_bars):
        prices[i] = prices[i-1] - 0.0002 + np.random.normal(0, 0.0001)

    df = pd.DataFrame({
        "Datetime": times,
        "Open": prices,
        "High": prices + np.abs(np.random.normal(0.0003, 0.0001, n_bars)),
        "Low": prices - np.abs(np.random.normal(0.0003, 0.0001, n_bars)),
        "Close": prices,
        "TickVolume": np.random.randint(50, 500, n_bars).astype(float),
        "Spread": np.random.randint(1, 5, n_bars).astype(float)
    })

    # Fix High/Low boundaries so High >= Max(Open, Close) and Low <= Min(Open, Close)
    df["High"] = df[["Open", "Close", "High"]].max(axis=1)
    df["Low"] = df[["Open", "Close", "Low"]].min(axis=1)

    return df


def main():
    parser = argparse.ArgumentParser(description="Forex_DNN Labeled Dataset Generator")
    parser.add_argument("--input", "-i", type=str, default="", help="Path to input raw/enriched OHLCV CSV file (if empty, generates synthetic data)")
    parser.add_argument("--output", "-o", type=str, default="output/market_state_dataset.csv", help="Path to output labeled dataset CSV")
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Symbol name (e.g. EURUSD)")
    parser.add_argument("--timeframe", type=str, default="M5", help="Timeframe name (e.g. M5, M15)")
    parser.add_argument("--window_size", type=int, default=35, help="Sliding window size (candles)")
    parser.add_argument("--window_stride", type=int, default=1, help="Sliding window stride (candles)")
    parser.add_argument("--min_confidence", type=float, default=0.4, help="Minimum labeling confidence threshold")
    args = parser.parse_args()

    # Load or generate raw data
    if args.input and os.path.exists(args.input):
        logger.info(f"Loading raw data from {args.input}...")
        df = pd.read_csv(args.input)
    else:
        logger.warning(f"Input file '{args.input}' not specified or not found. Fallback to generating high-quality synthetic data.")
        df = generate_synthetic_ohlcv(n_bars=1200)

    # Initialize components
    labeler = MarketStateLabeler(
        label_version="1.0",
        engine_version="1.0",
        min_confidence=args.min_confidence
    )

    engine = LabelEngine(
        window_size=args.window_size,
        window_stride=args.window_stride
    )

    # Generate labeled dataset
    dataset_df = engine.generate(
        df=df,
        symbol=args.symbol,
        timeframe=args.timeframe,
        labeler=labeler,
        output_csv_path=args.output
    )

    # Validate output dataset
    validator = DatasetValidator(expected_window_size=args.window_size)
    validation_report = validator.validate(dataset_df)

    # Print Validation & Class Distribution Report
    print("\n==================================================")
    print("        DATASET GENERATION SUMMARY REPORT         ")
    print("==================================================")
    print(f"Dataset Rows Generated : {len(dataset_df)}")
    print(f"Validation Status      : {'PASSED' if validation_report['is_valid'] else 'FAILED'}")

    print("\nValidation Checks Status:")
    for check_name, status in validation_report["checks"].items():
        print(f" - {check_name:<25}: {status}")

    if validation_report["errors"]:
        print("\nValidation Errors:")
        for err in validation_report["errors"]:
            print(f" ❌ {err}")

    if validation_report["warnings"]:
        print("\nValidation Warnings:")
        for warn in validation_report["warnings"]:
            print(f" ⚠️ {warn}")

    print("\nClass Distribution:")
    if not dataset_df.empty and "label" in dataset_df.columns:
        counts = dataset_df["label"].value_counts()
        total = len(dataset_df)
        for lbl, cnt in counts.items():
            print(f" - {lbl:<15}: {cnt:<5} ({cnt/total:.2%})")
    else:
        print(" - Empty dataset: No classes to distribute.")
    print("==================================================\n")


if __name__ == "__main__":
    main()
