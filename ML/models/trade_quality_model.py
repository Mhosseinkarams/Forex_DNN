import numpy as np
from typing import Dict, List, Any, Union
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ML.base_model import BaseTradingModel, TradeQualityPrediction


class TradeQualityModel(BaseTradingModel):
    """
    Trade Quality Model wrapped inside the Production ML Framework.
    Estimates the potential trade quality and success probabilities using various features.
    """
    def build_model(self):
        """
        Instantiate the underlying LightGBM or RandomForest model (classifier or regressor).
        For simplicity, let's treat it as a classifier or regressor depending on configuration.
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

    def prediction_schema(self, probas: np.ndarray, raw_pred: np.ndarray) -> TradeQualityPrediction:
        """
        Convert probabilities to structured TradeQualityPrediction.
        Classes are: LOW_QUALITY=0, HIGH_QUALITY=1
        """
        if probas.ndim == 2:
            row_probas = probas[0]
        else:
            row_probas = probas

        # If binary classification (LOW_QUALITY, HIGH_QUALITY)
        class_names = ["LOW_QUALITY", "HIGH_QUALITY"]
        inference_engine = self.calibrated_model or self.model
        class_ids = getattr(inference_engine, "classes_", range(len(row_probas)))
        prob_dict = {name: 0.0 for name in class_names}
        for class_id, probability in zip(class_ids, row_probas):
            if 0 <= int(class_id) < len(class_names):
                prob_dict[class_names[int(class_id)]] = float(probability)

        confidence = float(max(row_probas))
        # Quality score can be mapped directly to the HIGH_QUALITY probability
        quality_score = prob_dict["HIGH_QUALITY"]

        return TradeQualityPrediction(
            quality_score=quality_score,
            expected_win_rate=quality_score,  # Simple proxy for example
            confidence=confidence,
            expected_risk_reward=1.5  # Standard default estimate
        )

    def required_feature_groups(self) -> List[str]:
        """
        Trade quality estimation uses SMC structure, Supply/Demand, Indicator, and Volatility groups.
        """
        return ["SMC_Structural", "Supply_Demand", "Indicator", "Volatility"]

    def evaluation_metrics(self) -> List[str]:
        return ["accuracy", "precision_binary", "recall_binary", "f1_binary", "roc_auc", "confusion_matrix"]

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
