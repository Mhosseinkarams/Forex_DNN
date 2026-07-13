import numpy as np
from typing import Dict, List, Any, Union
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from ML.base_model import BaseTradingModel, MarketStatePrediction


class MarketStateClassifier(BaseTradingModel):
    """
    Refactored Market State Classifier model wrapped inside the Production ML Framework.
    Saves and loads correctly, integrates with YAML configs and the Feature Registry,
    and returns rich MarketStatePrediction objects.
    """
    def build_model(self):
        """
        Instantiate the underlying LightGBM or RandomForest classifier.
        """
        if self.model_type == "lightgbm":
            self.model = LGBMClassifier(
                random_state=self.random_state,
                n_estimators=self.hyperparameters.get("n_estimators", 100),
                learning_rate=self.hyperparameters.get("learning_rate", 0.05),
                max_depth=self.hyperparameters.get("max_depth", 6),
                num_leaves=self.hyperparameters.get("num_leaves", 31),
                verbosity=self.hyperparameters.get("verbosity", -1)
            )
        elif self.model_type == "randomforest":
            self.model = RandomForestClassifier(
                n_estimators=self.hyperparameters.get("n_estimators", 100),
                max_depth=self.hyperparameters.get("max_depth", 8),
                random_state=self.random_state,
                n_jobs=self.hyperparameters.get("n_jobs", -1)
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Choose 'lightgbm' or 'randomforest'.")

    def prediction_schema(self, probas: np.ndarray, raw_pred: np.ndarray) -> MarketStatePrediction:
        """
        Convert LightGBM/RandomForest output arrays to structured MarketStatePrediction.
        Classes are: TREND=0, RANGE=1, TRANSITION=2
        """
        # Support batch prediction or single prediction
        if probas.ndim == 2:
            row_probas = probas[0]
            pred_val = raw_pred[0]
        else:
            row_probas = probas
            pred_val = raw_pred

        class_names = ["TREND", "RANGE", "TRANSITION"]
        prob_dict = {}
        for i, name in enumerate(class_names):
            if i < len(row_probas):
                prob_dict[name] = float(row_probas[i])
            else:
                prob_dict[name] = 0.0

        # Mapping index back to label
        regime = class_names[int(pred_val)] if int(pred_val) < len(class_names) else "TRANSITION"

        # Simple dynamic metrics from input data/predictions
        # Since we're dealing with raw probabilities, confidence is the max of class probabilities
        confidence = float(max(row_probas))

        return MarketStatePrediction(
            regime=regime,
            trend_probability=prob_dict["TREND"],
            range_probability=prob_dict["RANGE"],
            transition_probability=prob_dict["TRANSITION"],
            confidence=confidence,
            trend_strength=prob_dict["TREND"],  # For backward-compatibility or dynamic analysis
            expected_volatility=0.0,
            expected_persistence=confidence
        )

    def required_feature_groups(self) -> List[str]:
        """
        Market state classification uses indicator and trend/range/compression groups.
        """
        return ["Indicator", "Trend", "Volume", "SMC_Structural"]

    def evaluation_metrics(self) -> List[str]:
        return ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "confusion_matrix"]

    def default_hyperparameters(self) -> Dict[str, Any]:
        if self.model_type == "lightgbm":
            return {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 6,
                "num_leaves": 31,
                "verbosity": -1
            }
        else:
            return {
                "n_estimators": 100,
                "max_depth": 8,
                "n_jobs": -1
            }
