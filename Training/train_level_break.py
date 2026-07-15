import os
import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd

from ML.models.level_break_probability import LevelBreakProbabilityModel
from ML.trainer import Trainer
from ML.evaluator import Evaluator


def run_training(dataset_path: str, model_save_path: str, random_seed: int = 42):
    print("==================================================")
    print("    TRAINING LEVEL BREAK PROBABILITY MODEL        ")
    print("==================================================")

    # Initialize Directory Layout Structure
    for d in [
        "raw_data", "processed_data", "cache", "datasets",
        "models", "models/MarketState", "models/LevelBreak",
        "experiments", "training_runs", "reports", "backtests"
    ]:
        os.makedirs(d, exist_ok=True)

    np.random.seed(random_seed)

    # Check if dataset exists, if not generate dummy
    if not os.path.exists(dataset_path):
        print(f"Dataset path '{dataset_path}' not found. Creating a synthetic dataset for demonstration...")
        n_samples = 1000
        from ML.feature_registry import FeatureRegistry
        reg = FeatureRegistry()
        enabled_features = [f.name for f in reg.list_enabled()]

        data = {feat: np.random.randn(n_samples) for feat in enabled_features}
        df = pd.DataFrame(data)
        # Class labels: 0 for REJECT, 1 for BREAK
        df["target"] = np.random.choice([0, 1], n_samples)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n_samples, freq="5min")
        os.makedirs(os.path.dirname(os.path.abspath(dataset_path)), exist_ok=True)
        df.to_csv(dataset_path, index=False)
        print(f"Synthetic dataset saved to {dataset_path}")

    if dataset_path.endswith(".parquet"):
        df = pd.read_parquet(dataset_path)
    else:
        df = pd.read_csv(dataset_path)
    print(f"Loaded dataset containing {len(df)} samples.")

    # Initialize child class with external YAML config
    config_path = "configs/level_break.yaml" if os.path.exists("configs/level_break.yaml") else None

    clf = LevelBreakProbabilityModel(
        model_type="lightgbm",
        config_path=config_path,
        random_state=random_seed
    )

    # Use Trainer to perform train/val split, training, and registration
    trainer = Trainer(random_seed=random_seed)

    # Extract feature columns automatically, avoiding metadata leakage
    metadata_cols = [
        "label", "target", "confidence", "timestamp", "Datetime", "symbol", "timeframe",
        "window_start", "window_end", "sample_id", "label_version", "engine_version",
        "meta_labeler_rule_fired", "Open", "High", "Low", "Close", "TickVolume", "Spread",
        "ema_50", "ema_600", "ema_800", "zone_type"
    ]
    feature_cols = [c for c in df.columns if c not in metadata_cols and not c.startswith("meta_labeler_")]

    train_results = trainer.train_model(
        model=clf,
        df=df,
        target_col="target",
        feature_cols=feature_cols,
        test_size=0.2,
        chronological=True,
        dataset_version="unknown",
        dataset_hash="unknown",
        model_save_path=model_save_path,
        is_production=True,
        version="1.0.0"
    )

    X_val = train_results["X_val"]
    y_val = train_results["y_val"]

    # Use Evaluator to create premium Markdown & HTML reports
    evaluator = Evaluator(output_dir="reports")
    classes = ["REJECT", "BREAK"]

    evaluator.evaluate_and_report(
        model=clf,
        X_val=X_val,
        y_val=y_val,
        classes=classes,
        report_name="level_break_evaluation_report"
    )

    print(f"\n--- Model Performance Summary ---")
    print(clf.get_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="output/level_break_dataset.csv")
    parser.add_argument("--model", type=str, default="output/level_break_probability.joblib")
    args = parser.parse_args()

    run_training(args.dataset, args.model)
