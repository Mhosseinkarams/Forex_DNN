import os
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from ML.feature_registry import FeatureRegistry

class MarketStateClassifier:
    """
    Wrapper class for the Market State Classifier model.
    Supports LightGBM and RandomForest backends, and handles feature registry integration.
    """
    def __init__(self, model_type: str = "lightgbm", random_state: int = 42):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.registry = FeatureRegistry(load_defaults=True)
        self._feature_names = None

        if self.model_type == "lightgbm":
            self.model = LGBMClassifier(
                random_state=self.random_state,
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                verbosity=-1
            )
        elif self.model_type == "randomforest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choose 'lightgbm' or 'randomforest'.")

    def fit(self, X, y, feature_names=None):
        """
        Fits the underlying classifier model.
        """
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
        elif feature_names is not None:
            self._feature_names = list(feature_names)

        if self.model_type == "lightgbm" and self._feature_names is not None:
            # Avoid passing feature_name if data format is numpy array and mismatch is possible
            if isinstance(X, pd.DataFrame):
                self.model.fit(X, y)
            else:
                self.model.fit(X, y, feature_name=self._feature_names)
        else:
            self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predicts classes.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Performs inference for a single row or features and returns a dictionary of
        probabilities matching the classes ["TREND", "RANGE", "TRANSITION"] plus confidence.
        """
        # Obtain probability array from standard model
        probas = self.model.predict_proba(X)

        # If input has multiple rows, we return list of dicts or just the first row dict
        # Since MMStrategy does it row by row, let's extract the first row
        if probas.ndim == 2:
            row_probas = probas[0]
        else:
            row_probas = probas

        # Classes are encoded as: TREND=0, RANGE=1, TRANSITION=2
        class_names = ["TREND", "RANGE", "TRANSITION"]
        prob_dict = {}
        for i, name in enumerate(class_names):
            if i < len(row_probas):
                prob_dict[name] = float(row_probas[i])
            else:
                prob_dict[name] = 0.0

        prob_dict["confidence"] = float(max(row_probas))
        return prob_dict

    def get_feature_importance(self):
        """
        Returns a dictionary mapping feature names to their importance values.
        """
        importances = self.model.feature_importances_
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
        Saves the complete model wrapper using joblib.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        """
        Loads the complete model wrapper.
        """
        return joblib.load(path)
