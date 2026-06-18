import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Collecting_Data.preproc_single_inout import SingleInOutPreprocessor


def chronological_split(df, train_fraction):
    split_idx = int(len(df) * train_fraction)
    if split_idx <= 1 or split_idx >= len(df):
        raise ValueError("train_fraction leaves too little data for train/test evaluation")
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def evaluate_previous_direction_baseline(test_df):
    actual = test_df["Binary_Label"].to_numpy(dtype=int)
    previous_direction = (test_df["Close"].diff() > 0).astype(int).shift(1)
    predictions = previous_direction.fillna(1).to_numpy(dtype=int)

    accuracy = float(np.mean(predictions == actual))
    positive_rate = float(np.mean(predictions))
    majority_accuracy = float(max(np.mean(actual == 0), np.mean(actual == 1)))

    return {
        "samples": int(len(test_df)),
        "accuracy": accuracy,
        "positive_prediction_rate": positive_rate,
        "majority_class_accuracy": majority_accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="Run a chronological no-ML baseline for Forex_DNN.")
    parser.add_argument("--data", default="GBPUSD_1h.csv", help="CSV filename inside Data/")
    parser.add_argument("--train-fraction", type=float, default=0.7, help="Chronological train split fraction")
    args = parser.parse_args()

    preprocessor = SingleInOutPreprocessor()
    df = preprocessor.preprocess(args.data)
    if df is None or df.empty:
        raise SystemExit(f"No usable rows found for Data/{args.data}")

    train_df, test_df = chronological_split(df, args.train_fraction)
    metrics = evaluate_previous_direction_baseline(test_df)

    print("Forex_DNN baseline")
    print(f"Data file: Data/{args.data}")
    print(f"Rows: {len(df)} total, {len(train_df)} train, {len(test_df)} test")
    print("Model: previous-bar direction")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
