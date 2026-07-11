import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from ML.models.level_break_probability import LevelBreakProbabilityModel

def run_training(dataset_path: str, model_save_path: str, random_seed: int = 42):
    print("==================================================")
    print("    TRAINING LEVEL BREAK PROBABILITY MODEL        ")
    print("==================================================")

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

    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset containing {len(df)} samples.")

    # Separate features and targets
    feature_cols = [c for c in df.columns if c not in ["target", "zone_type", "timestamp"]]
    X = df[feature_cols]
    y = df["target"]

    # Chronological Split (No random shuffling)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx].to_numpy(), y.iloc[split_idx:].to_numpy()

    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # Train LevelBreakProbabilityModel wrapper
    clf = LevelBreakProbabilityModel(model_type="lightgbm", random_state=random_seed)
    clf.fit(X_train, y_train, feature_names=feature_cols)

    # Predict and Evaluate
    y_pred = clf.model.predict(X_test)
    y_proba = clf.model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("\n--- Model Performance Evaluation ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["REJECT", "BREAK"]))

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="output/level_break_dataset.csv")
    parser.add_argument("--model", type=str, default="output/level_break_probability.joblib")
    args = parser.parse_args()

    run_training(args.dataset, args.model)
