import os
import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd

from Configs.path_manager import PathManager
from ML.trainer import Trainer
from ML.evaluator import Evaluator
from ML.feature_registry import FeatureRegistry, TARGET_COLUMNS


def run_training(dataset_path: str, model_save_path: str, target_col: str = "strategy_outcome", random_seed: int = 42):
    print("==================================================")
    print("     TRAINING STRATEGY OUTCOME QUALITY MODEL      ")
    print("==================================================")

    from Collecting_Data.memory_monitor import MemoryMonitor
    mem_monitor = MemoryMonitor()
    mem_monitor.check("Training start")

    PathManager.ensure_all_dirs()
    np.random.seed(random_seed)

    if not os.path.exists(dataset_path):
        print(f"Dataset path '{dataset_path}' not found. Creating synthetic dataset...")
        n_samples = 1000
        reg = FeatureRegistry()
        enabled_features = [f.name for f in reg.list_enabled()]

        data = {feat: np.random.randn(n_samples) for feat in enabled_features}
        df = pd.DataFrame(data)
        df[target_col] = np.random.choice(["WIN", "LOSS", "TIMEOUT"], n_samples)
        df["datetime"] = pd.date_range("2024-01-01", periods=n_samples, freq="5min")
        os.makedirs(os.path.dirname(os.path.abspath(dataset_path)), exist_ok=True)
        df.to_parquet(dataset_path, index=False)
        print(f"Synthetic dataset saved to {dataset_path}")

    if dataset_path.endswith(".parquet"):
        df = pd.read_parquet(dataset_path)
    else:
        df = pd.read_csv(dataset_path)

    # Filter valid target rows
    if target_col in df.columns:
        df = df[df[target_col].notnull() & (df[target_col] != "AMBIGUOUS")].copy()

    print(f"Loaded dataset containing {len(df)} valid samples.")
    mem_monitor.check("Dataset loaded")

    from ML.models.trade_quality_model import TradeQualityModel
    clf = TradeQualityModel(
        model_type="lightgbm",
        random_state=random_seed
    )

    trainer = Trainer(random_seed=random_seed)

    # Derive feature columns by excluding target and metadata columns
    feature_cols = [c for c in df.columns if c not in TARGET_COLUMNS and not c.startswith("meta_")]

    train_results = trainer.train_model(
        model=clf,
        df=df,
        target_col=target_col,
        feature_cols=feature_cols,
        test_size=0.2,
        chronological=True,
        purge_window=55,
        dataset_version="2.0.0-causal",
        dataset_hash="causal_hash",
        model_save_path=model_save_path,
        is_production=True,
        version="1.0.0"
    )

    X_val = train_results["X_val"]
    y_val = train_results["y_val"]

    evaluator = Evaluator(output_dir=PathManager.get_relative_path("reports"))
    classes = ["WIN", "LOSS", "TIMEOUT"]

    evaluator.evaluate_and_report(
        model=clf,
        X_val=X_val,
        y_val=y_val,
        classes=classes,
        report_name="strategy_outcome_evaluation_report"
    )

    print(f"\n--- Model Performance Summary ---")
    print(clf.get_summary())

    mem_monitor.check("Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=PathManager.get_relative_path("datasets", "market_state_dataset.parquet"))
    parser.add_argument("--model", type=str, default=PathManager.get_relative_path("models", "TradeQuality/trade_quality_model.joblib"))
    parser.add_argument("--target", type=str, default="strategy_outcome")
    args = parser.parse_args()

    run_training(args.dataset, args.model, target_col=args.target)
