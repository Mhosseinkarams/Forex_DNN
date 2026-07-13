import os
import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from ML.models.market_state_classifier import MarketStateClassifier
from ML.trainer import Trainer
from ML.evaluator import Evaluator


def run_training(dataset_path: str, model_save_path: str, random_seed: int = 42):
    print("==================================================")
    print("      TRAINING MARKET STATE CLASSIFIER MODEL      ")
    print("==================================================")

    # Initialize Directory Layout Structure
    for d in [
        "raw_data", "processed_data", "cache", "datasets",
        "models", "models/MarketState", "models/LevelBreak",
        "experiments", "training_runs", "reports", "backtests"
    ]:
        os.makedirs(d, exist_ok=True)

    np.random.seed(random_seed)

    # Check if dataset exists, if not generate or use synthetic dummy data for demo
    if not os.path.exists(dataset_path):
        print(f"Dataset path '{dataset_path}' not found. Creating a synthetic dataset for demonstration purposes...")
        n_samples = 1000
        from ML.feature_registry import FeatureRegistry
        reg = FeatureRegistry()
        enabled_features = [f.name for f in reg.list_enabled()]

        data = {feat: np.random.randn(n_samples) for feat in enabled_features}
        df = pd.DataFrame(data)
        # Class labels: 0 for TREND, 1 for RANGE, 2 for TRANSITION
        df["label"] = np.random.choice(["TREND", "RANGE", "TRANSITION"], n_samples)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n_samples, freq="5min")
        os.makedirs(os.path.dirname(os.path.abspath(dataset_path)), exist_ok=True)
        df.to_csv(dataset_path, index=False)
        print(f"Synthetic dataset saved to {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset containing {len(df)} samples.")

    # Initialize child class with external YAML config
    config_path = "configs/market_state.yaml" if os.path.exists("configs/market_state.yaml") else None

    clf = MarketStateClassifier(
        model_type="lightgbm",
        config_path=config_path,
        random_state=random_seed
    )

    # Read dataset metadata if available
    dataset_dir = os.path.dirname(os.path.abspath(dataset_path))
    metadata_path = os.path.join(dataset_dir, "metadata.json")

    dataset_version = "unknown"
    dataset_hash = "unknown"

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                meta = json.load(f)
                dataset_version = meta.get("dataset_version", os.path.basename(dataset_dir))
                dataset_hash = meta.get("fingerprint", {}).get("dataset_hash", "unknown")
        except Exception as e:
            print(f"Warning: Failed to load dataset metadata: {e}")

    # Use Trainer to perform train/val split, training, and registration
    trainer = Trainer(random_seed=random_seed)

    # Extract feature columns automatically
    feature_cols = [c for c in df.columns if c not in ["label", "confidence", "timestamp"]]

    train_results = trainer.train_model(
        model=clf,
        df=df,
        target_col="label",
        feature_cols=feature_cols,
        test_size=0.2,
        chronological=True,
        dataset_version=dataset_version,
        dataset_hash=dataset_hash,
        model_save_path=model_save_path,
        is_production=True,
        version="1.0.0"
    )

    X_val = train_results["X_val"]
    y_val = train_results["y_val"]

    # Use Evaluator to create premium Markdown & HTML reports
    evaluator = Evaluator(output_dir="reports")
    classes = ["TREND", "RANGE", "TRANSITION"]

    evaluator.evaluate_and_report(
        model=clf,
        X_val=X_val,
        y_val=y_val,
        classes=classes,
        report_name="market_state_evaluation_report"
    )

    print(f"\n--- Model Performance Summary ---")
    print(clf.get_summary())

    # Compatibility check: save model reproducibility companion json file (reproducibility.json)
    git_commit = clf.metadata.get("git_commit", "unknown")
    reproducibility = {
        "trained_from_dataset": dataset_version,
        "dataset_hash": dataset_hash,
        "git_commit": git_commit,
        "training_script_version": "2.0-production-ml",
        "training_date": datetime.now().isoformat()
    }

    model_dir = os.path.dirname(os.path.abspath(model_save_path))
    repro_path = os.path.join(model_dir, "reproducibility.json")
    with open(repro_path, "w") as f:
        json.dump(reproducibility, f, indent=4)
    print(f"Saved model reproducibility registry to {repro_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="output/market_state_dataset.csv")
    parser.add_argument("--model", type=str, default="output/market_state_classifier.joblib")
    args = parser.parse_args()

    run_training(args.dataset, args.model)
