import numpy as np
from typing import Dict, List, Any, Union
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from ML.base_model import BaseTradingModel, LevelBreakPrediction


class LevelBreakProbabilityModel(BaseTradingModel):
    """
    Refactored Level Break Probability Model wrapped inside the Production ML Framework.
    Saves and loads correctly, integrates with YAML configs and the Feature Registry,
    and returns rich LevelBreakPrediction objects.
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

    def prediction_schema(self, probas: np.ndarray, raw_pred: np.ndarray) -> LevelBreakPrediction:
        """
        Convert LightGBM/RandomForest output arrays to structured LevelBreakPrediction.
        Classes are: REJECT=0, BREAK=1
        """
        if probas.ndim == 2:
            row_probas = probas[0]
        else:
            row_probas = probas

        class_names = ["REJECT", "BREAK"]
        inference_engine = self.calibrated_model or self.model
        class_ids = getattr(inference_engine, "classes_", range(len(row_probas)))
        prob_dict = {name: 0.0 for name in class_names}
        for class_id, probability in zip(class_ids, row_probas):
            if 0 <= int(class_id) < len(class_names):
                prob_dict[class_names[int(class_id)]] = float(probability)

        confidence = float(max(row_probas))

        return LevelBreakPrediction(
            break_probability=prob_dict["BREAK"],
            reject_probability=prob_dict["REJECT"],
            confidence=confidence,
            expected_move=0.0,
            expected_time_to_break=0.0
        )

    def required_feature_groups(self) -> List[str]:
        """
        Level break classification uses SMC structure and zone/liquidity features.
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
