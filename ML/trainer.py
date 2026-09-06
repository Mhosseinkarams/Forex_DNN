import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from Configs.path_manager import PathManager
from ML.base_model import BaseTradingModel
from ML.model_registry import ModelRegistry

logger = logging.getLogger("Trainer")


class Trainer:
    """
    Standardized Training Engine for any BaseTradingModel subclass.
    Handles train/test split, chronological alignments, calibration, metric calculations,
    and automated registration of model checkpoints with metadata preservation.
    """
    def __init__(
        self,
        random_seed: int = 42,
        model_registry_path: str = None
    ):
        if model_registry_path is None:
            model_registry_path = PathManager.get_relative_path("models", "model_registry.json")
        self.random_seed = random_seed
        self.registry = ModelRegistry(registry_path=model_registry_path)

    def train_model(
        self,
        model: BaseTradingModel,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        chronological: bool = True,
        calibration_method: Optional[str] = None,
        dataset_version: str = "unknown",
        dataset_hash: str = "unknown",
        model_save_path: Optional[str] = None,
        is_production: bool = False,
        version: str = "1.0.0",
        purge_window: int = 55,
        embargo_window: int = 20
    ) -> Dict[str, Any]:
        """
        Runs the full training workflow:
        1. Select features and targets.
        2. Splitting (Chronological or Shuffled).
        3. Fit.
        4. Optional calibration.
        5. Evaluate on Validation set.
        6. Optional persistence and automated registration in Model Registry.
        """
        np.random.seed(self.random_seed)

        # 1. Resolve feature columns
        if feature_cols is None:
            # Dynamically ask model for required features
            feature_cols = model.get_features()
            # Intersect with columns actually available in df
            feature_cols = [c for c in feature_cols if c in df.columns]

        logger.info(f"Training features list size: {len(feature_cols)}")

        # Ensure features exist
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Feature columns {missing} not found in input DataFrame.")

        X = df[feature_cols]
        y = df[target_col]

        # Convert categorical targets to stable, domain-defined class IDs.  The
        # prediction schemas use these IDs, so alphabetical encoding would
        # silently map RANGE/TRANSITION probabilities to the wrong regimes.
        if y.dtype == object or isinstance(y.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(y):
            observed_classes = set(y.dropna().unique())
            if model.__class__.__name__ == "MarketStateClassifier":
                class_to_idx = {"TREND": 0, "RANGE": 1, "TRANSITION": 2}
            elif model.__class__.__name__ == "LevelBreakProbabilityModel":
                class_to_idx = {"REJECT": 0, "BREAK": 1}
            else:
                class_to_idx = {
                    value: index for index, value in enumerate(sorted(observed_classes))
                }

            unknown_classes = observed_classes - set(class_to_idx)
            if unknown_classes:
                raise ValueError(f"Unsupported target classes: {sorted(unknown_classes)}")

            y = y.map(class_to_idx)
            unique_classes = sorted(y.dropna().unique())
            logger.info(f"Encoded class target mapping: {class_to_idx}")
        else:
            unique_classes = sorted(list(pd.Series(y).dropna().unique()))

        # 2. Split Data with Purge and Embargo
        n_samples = len(df)
        val_size = int(n_samples * test_size)

        if chronological:
            train_end = max(1, n_samples - val_size - purge_window)
            val_start = train_end + purge_window

            X_train, X_val = X.iloc[:train_end], X.iloc[val_start:]
            y_train, y_val = y.iloc[:train_end].to_numpy(), y.iloc[val_start:].to_numpy()
            logger.info(f"Purged Chronological Split: Train [0..{train_end}] | PURGE GAP [{train_end}..{val_start}] ({purge_window} bars) | Val [{val_start}..{n_samples}]")
        else:
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            train_indices = indices[:n_samples - val_size]
            val_indices = indices[n_samples - val_size:]
            X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
            y_train, y_val = y.iloc[train_indices].to_numpy(), y.iloc[val_indices].to_numpy()

        logger.info(f"Training split: {len(X_train)} samples | Validation split: {len(X_val)} samples")

        # 3. Model Fit
        model.fit(
            X_train,
            y_train,
            feature_names=feature_cols,
            dataset_version=dataset_version,
            dataset_hash=dataset_hash
        )

        # 4. Calibration
        if calibration_method:
            model.calibrate(X_val, y_val, method=calibration_method)

        # 5. Evaluate
        metrics = self._evaluate_model(model, X_val, y_val, unique_classes)
        model.metadata["metrics"] = metrics
        model.metadata["validation_samples"] = len(X_val)

        logger.info(f"Evaluation results: {metrics}")

        # 6. Save & Register
        if model_save_path:
            model.save(model_save_path)
            self.registry.register_model(
                model_name=model.__class__.__name__,
                version=version,
                model_path=model_save_path,
                metrics=metrics,
                dataset_version=dataset_version,
                dataset_hash=dataset_hash,
                feature_registry_version=model.metadata["feature_registry_version"],
                model_type=model.model_type,
                is_production=is_production
            )

        return {
            "model": model,
            "metrics": metrics,
            "feature_cols": feature_cols,
            "X_train": X_train,
            "X_val": X_val,
            "y_train": y_train,
            "y_val": y_val
        }

    def _evaluate_model(self, model: BaseTradingModel, X_val: pd.DataFrame, y_val: np.ndarray, unique_classes: List[Any]) -> Dict[str, Any]:
        """
        Evaluate classification metrics on the validation set.
        """
        # Batch inference
        # Since our model's predict returns structured objects, let's predict row-by-row or extract raw prediction directly
        # Let's bypass prediction objects for bulk metric calculations to make it performant
        inference_engine = model.calibrated_model if model.calibrated_model is not None else model.model
        y_pred = inference_engine.predict(X_val)
        y_proba = inference_engine.predict_proba(X_val)

        acc = accuracy_score(y_val, y_pred)

        metrics = {
            "accuracy": float(acc)
        }

        # Handle multiclass vs binary metrics
        is_binary = len(unique_classes) <= 2

        if is_binary:
            metrics["precision"] = float(precision_score(y_val, y_pred, average="binary", zero_division=0))
            metrics["recall"] = float(recall_score(y_val, y_pred, average="binary", zero_division=0))
            metrics["f1"] = float(f1_score(y_val, y_pred, average="binary", zero_division=0))
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_val, y_proba[:, 1]))
        else:
            metrics["precision"] = float(precision_score(y_val, y_pred, average="weighted", zero_division=0))
            metrics["recall"] = float(recall_score(y_val, y_pred, average="weighted", zero_division=0))
            metrics["f1"] = float(f1_score(y_val, y_pred, average="weighted", zero_division=0))

        # Include confusion matrix as a nested list for serialization
        cm = confusion_matrix(y_val, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        return metrics
