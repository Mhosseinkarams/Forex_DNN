import os
import json
import logging
import hashlib
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
import joblib
import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

from sklearn.calibration import CalibratedClassifierCV
from ML.feature_registry import FeatureRegistry

logger = logging.getLogger("BaseTradingModel")


@dataclass
class MarketStatePrediction:
    """
    Structured, typed prediction result for Market State Classifier.
    """
    regime: str
    trend_probability: float
    range_probability: float
    transition_probability: float
    confidence: float
    trend_strength: float = 0.0
    expected_volatility: float = 0.0
    expected_persistence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LevelBreakPrediction:
    """
    Structured, typed prediction result for Level Break Probability.
    """
    break_probability: float
    reject_probability: float
    confidence: float
    expected_move: float = 0.0
    expected_time_to_break: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TradeQualityPrediction:
    """
    Structured, typed prediction result for Trade Quality Model.
    """
    quality_score: float
    expected_win_rate: float
    confidence: float
    expected_risk_reward: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseTradingModel(ABC):
    """
    Abstract Parent Class of every ML model in the Forex_DNN framework.
    Handles save/load, metadata tracking, feature registry integration,
    model/dataset/feature versioning, prediction/probability interface, and calibration.
    """
    def __init__(
        self,
        model_type: str = "lightgbm",
        config_path: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.config_path = config_path
        self.registry = FeatureRegistry(load_defaults=True)

        # Load hyperparameters (YAML config file -> manual dict -> defaults)
        self.hyperparameters = self._resolve_hyperparameters(config_path, hyperparameters)

        # Core model reference
        self.model = None
        self.calibrated_model = None  # To hold calibrated prefitted model if fitted
        self._feature_names: Optional[List[str]] = None

        # Metadata fields
        self.metadata: Dict[str, Any] = {
            "model_name": self.__class__.__name__,
            "model_type": self.model_type,
            "version": "1.0.0",
            "training_date": None,
            "dataset_version": "unknown",
            "dataset_hash": "unknown",
            "feature_registry_version": self.registry.compute_hash(),
            "feature_count": 0,
            "training_samples": 0,
            "validation_samples": 0,
            "hyperparameters": self.hyperparameters,
            "metrics": {},
            "git_commit": self._get_git_commit(),
            "random_seed": self.random_state,
            "calibration_method": None
        }

    def _resolve_hyperparameters(self, config_path: Optional[str], manual_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge default, YAML configuration, and manual dictionary hyperparameters.
        """
        params = self.default_hyperparameters().copy()

        # Load from YAML if provided
        if config_path and os.path.exists(config_path):
            if yaml is None:
                logger.warning("PyYAML not installed. Cannot load yaml config. Using defaults.")
            else:
                try:
                    with open(config_path, "r") as f:
                        yaml_params = yaml.safe_load(f)
                        if isinstance(yaml_params, dict):
                            # Handle potential nested 'hyperparameters' key
                            model_params = yaml_params.get("hyperparameters", yaml_params)
                            params.update(model_params)
                            logger.info(f"Loaded hyperparameters from {config_path}")
                except Exception as e:
                    logger.error(f"Failed to load yaml config from {config_path}: {e}")

        # Apply manual parameters override
        if manual_params:
            params.update(manual_params)

        return params

    def _get_git_commit(self) -> str:
        """
        Safely retrieve the latest git commit hash.
        """
        import subprocess
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            if res.returncode == 0:
                return res.stdout.strip()
        except Exception:
            pass
        return "unknown"

    @abstractmethod
    def build_model(self):
        """
        Instantiate the underlying scikit-learn or LightGBM model using self.hyperparameters.
        """
        pass

    @abstractmethod
    def prediction_schema(self, probas: np.ndarray, raw_pred: np.ndarray) -> Union[MarketStatePrediction, LevelBreakPrediction]:
        """
        Convert prediction arrays/probabilities to strongly-typed Prediction objects.
        """
        pass

    @abstractmethod
    def required_feature_groups(self) -> List[str]:
        """
        Returns list of feature groups or category names required by this model.
        """
        pass

    @abstractmethod
    def evaluation_metrics(self) -> List[str]:
        """
        List of validation metrics tracked for this model type.
        """
        pass

    @abstractmethod
    def default_hyperparameters(self) -> Dict[str, Any]:
        """
        Provide default hyperparameter dictionary.
        """
        pass

    def get_features(self) -> List[str]:
        """
        Query enabled features based on the required feature groups and FeatureRegistry.
        """
        groups = self.required_feature_groups()
        features = []
        for g in groups:
            selected = self.registry.select_group(g)
            for f in selected:
                if f.enabled and f.name not in features:
                    features.append(f.name)
        return features

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None,
        dataset_version: str = "unknown",
        dataset_hash: str = "unknown"
    ) -> 'BaseTradingModel':
        """
        Fits the underlying classifier model and updates metadata.
        """
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
        elif feature_names is not None:
            self._feature_names = list(feature_names)
        else:
            # Generate dummy names
            self._feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Ensure the model is initialized
        if self.model is None:
            self.build_model()

        # Update metadata before fit
        self.metadata["dataset_version"] = dataset_version
        self.metadata["dataset_hash"] = dataset_hash
        self.metadata["feature_count"] = X.shape[1]
        self.metadata["training_samples"] = len(X)
        self.metadata["training_date"] = datetime.now().isoformat()

        # Handle fit based on model backend
        if self.model_type == "lightgbm" and hasattr(self.model, "fit"):
            if isinstance(X, pd.DataFrame):
                self.model.fit(X, y)
            else:
                self.model.fit(X, y, feature_name=self._feature_names)
        else:
            self.model.fit(X, y)

        logger.info(f"Fitted model of type '{self.model_type}' successfully.")
        return self

    def calibrate(self, X_val: pd.DataFrame, y_val: np.ndarray, method: str = "sigmoid") -> 'BaseTradingModel':
        """
        Calibrate probabilities using Platt Scaling ('sigmoid') or Isotonic Regression ('isotonic').
        Wraps the fitted base model using scikit-learn CalibratedClassifierCV.
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before calibration.")

        logger.info(f"Calibrating model probabilities using method='{method}'...")
        try:
            from sklearn.calibration import FrozenEstimator
            calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(self.model), method=method)
        except ImportError:
            calibrated = CalibratedClassifierCV(estimator=self.model, method=method, cv="prefit")

        calibrated.fit(X_val, y_val)

        self.calibrated_model = calibrated
        self.metadata["calibration_method"] = method
        logger.info("Probability calibration complete.")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> Union[MarketStatePrediction, LevelBreakPrediction]:
        """
        Perform inference for a snapshot or feature row and return a structured prediction object.
        Supports pd.DataFrame, 2D np.ndarray, 1D np.ndarray, or snapshots as dictionaries.
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized or fitted yet.")

        # Resolve dict snapshots into ordered dataframe/numpy arrays
        if isinstance(X, dict):
            # Parse single row snapshot
            X = pd.DataFrame([X])

        # If pd.DataFrame, align features with the trained feature names if available
        if isinstance(X, pd.DataFrame) and self._feature_names is not None:
            missing_cols = [c for c in self._feature_names if c not in X.columns]
            if missing_cols:
                for c in missing_cols:
                    # Fallback to feature registry default
                    if self.registry.exists(c):
                        X[c] = self.registry.get(c).default_value
                    else:
                        X[c] = 0.0
            # Ensure correct column ordering
            X = X[self._feature_names]

        # Use calibrated model if available, otherwise fallback to base model
        inference_engine = self.calibrated_model if self.calibrated_model is not None else self.model

        probas = inference_engine.predict_proba(X)
        raw_pred = inference_engine.predict(X)

        return self.prediction_schema(probas, raw_pred)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> Dict[str, float]:
        """
        Returns raw dictionary of class probabilities for backward compatibility.
        """
        pred = self.predict(X)
        res = {}
        if isinstance(pred, MarketStatePrediction):
            res["TREND"] = pred.trend_probability
            res["RANGE"] = pred.range_probability
            res["TRANSITION"] = pred.transition_probability
            res["confidence"] = pred.confidence
        elif isinstance(pred, LevelBreakPrediction):
            res["BREAK"] = pred.break_probability
            res["REJECT"] = pred.reject_probability
            res["confidence"] = pred.confidence
        elif isinstance(pred, TradeQualityPrediction):
            res["QUALITY_SCORE"] = pred.quality_score
            res["EXPECTED_WIN_RATE"] = pred.expected_win_rate
            res["confidence"] = pred.confidence
        return res

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Extract feature importance from the underlying model and map to feature names.
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted to retrieve feature importances.")

        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            return {}

        feature_names = None
        if hasattr(self.model, "feature_name_"):
            feature_names = self.model.feature_name_
        elif self._feature_names is not None:
            feature_names = self._feature_names

        if feature_names is not None and len(feature_names) == len(importances):
            return {name: float(imp) for name, imp in zip(feature_names, importances)}
        else:
            return {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}

    def save(self, path: str):
        """
        Saves the complete model wrapper using joblib, including hyperparameters and metadata.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # Store metadata inside the joblib structure
        payload = {
            "model_class": self.__class__.__name__,
            "model_type": self.model_type,
            "random_state": self.random_state,
            "hyperparameters": self.hyperparameters,
            "_feature_names": self._feature_names,
            "metadata": self.metadata,
            "model": self.model,
            "calibrated_model": self.calibrated_model
        }
        joblib.dump(payload, path)

        # Save a companion JSON file for direct inspection/registry consumption
        meta_path = path.replace(".joblib", "_metadata.json")
        try:
            with open(meta_path, "w") as f:
                json.dump(self.metadata, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save metadata companion json to {meta_path}: {e}")

        logger.info(f"Model saved successfully to {path}")

    @classmethod
    def load(cls, path: str) -> 'BaseTradingModel':
        """
        Loads the saved model wrapper, restoring state, calibrated models, features, and metadata.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model found at {path}")

        payload = joblib.load(path)

        # Re-instantiate the correct child class
        # Note: we pass dummy hyperparameters first, then overwrite
        instance = cls(
            model_type=payload["model_type"],
            random_state=payload["random_state"],
            hyperparameters=payload["hyperparameters"]
        )
        instance._feature_names = payload["_feature_names"]
        instance.metadata = payload["metadata"]
        instance.model = payload["model"]
        instance.calibrated_model = payload["calibrated_model"]

        logger.info(f"Model loaded successfully from {path}")
        return instance

    def get_summary(self) -> str:
        """
        Return a beautiful markdown-friendly summary of the model state.
        """
        summary = [
            f"# Model Summary: {self.__class__.__name__}",
            f"- **Backend Engine**: {self.model_type.upper()}",
            f"- **Target Strategy**: {self.required_feature_groups()}",
            f"- **Training Date**: {self.metadata['training_date']}",
            f"- **Dataset Version**: {self.metadata['dataset_version']}",
            f"- **Feature Count**: {self.metadata['feature_count']}",
            f"- **Training Samples**: {self.metadata['training_samples']}",
            f"- **Calibration Status**: {'Calibrated (' + self.metadata['calibration_method'] + ')' if self.calibrated_model else 'Raw probabilities'}",
            f"- **Registry Hash**: `{self.metadata['feature_registry_version'][:12]}`",
            "",
            "## Active Hyperparameters",
        ]
        for k, v in self.hyperparameters.items():
            summary.append(f"- **{k}**: {v}")

        return "\n".join(summary)
