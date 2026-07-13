import os
import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from ML.models.market_state_classifier import MarketStateClassifier

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
        # Make a dummy dataset
        n_samples = 1000
        # Generate random features (e.g., 49 features matching registry count)
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

    # Separate features and labels
    feature_cols = [c for c in df.columns if c not in ["label", "confidence", "timestamp"]]
    X = df[feature_cols]
    y = df["label"]

    # Encode labels to integer indices
    classes = ["TREND", "RANGE", "TRANSITION"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = y.map(class_to_idx).fillna(2).astype(int).to_numpy()

    # Chronological Split (No random shuffling)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]

    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # Train MarketStateClassifier wrapper
    clf = MarketStateClassifier(model_type="lightgbm", random_state=random_seed)
    clf.fit(X_train, y_train, feature_names=feature_cols)

    # Predict and Evaluate
    y_pred = clf.model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n--- Model Performance Evaluation ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score (Weighted): {f1:.4f}")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=classes))

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))

    # Print Feature Importance
    print("\n--- Top 10 Feature Importances ---")
    importances = clf.get_feature_importance()
    sorted_importance = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for name, imp in sorted_importance[:10]:
        print(f"{name:<35}: {imp:.4f}")

    # Check SHAP availability
    try:
        import shap
        print("\nCalculating SHAP values...")
        explainer = shap.TreeExplainer(clf.model)
        shap_values = explainer.shap_values(X_test)
        print("SHAP calculation: SUCCESS")
    except ImportError:
        print("\nSHAP library not installed. Skipping SHAP explanation calculation.")

    # Save the trained model
    clf.save(model_save_path)
    print(f"Model saved successfully to {model_save_path}")

    # Enforce Model Reproducibility & Registry
    dataset_dir = os.path.dirname(os.path.abspath(dataset_path))
    metadata_path = os.path.join(dataset_dir, "metadata.json")

    dataset_version = "unknown"
    dataset_hash = "unknown"
    git_commit = "unknown"

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                meta = json.load(f)
                dataset_version = meta.get("dataset_version", os.path.basename(dataset_dir))
                dataset_hash = meta.get("fingerprint", {}).get("dataset_hash", "unknown")
                git_commit = meta.get("fingerprint", {}).get("git_commit", "unknown")
        except Exception as e:
            print(f"Warning: Failed to load dataset metadata: {e}")

    reproducibility = {
        "trained_from_dataset": dataset_version,
        "dataset_hash": dataset_hash,
        "git_commit": git_commit,
        "training_script_version": "1.1",
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
