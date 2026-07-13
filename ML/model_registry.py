import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Type
from ML.base_model import BaseTradingModel
from ML.models.market_state_classifier import MarketStateClassifier
from ML.models.level_break_probability import LevelBreakProbabilityModel

logger = logging.getLogger("ModelRegistry")


class ModelRegistry:
    """
    Centralized model registry to register, version, track, and load ML models.
    Saves metadata in 'models/model_registry.json' and allows easy retrieval
    of the latest production models.
    """
    def __init__(self, registry_path: str = "models/model_registry.json"):
        self.registry_path = registry_path
        self.registry_data: Dict[str, Any] = {"models": []}
        self._load_registry()

    def _load_registry(self):
        """
        Loads registered model metadata from disk.
        """
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "models" in data:
                        self.registry_data = data
                    else:
                        logger.warning("Registry file format is invalid, initializing empty registry.")
            except Exception as e:
                logger.error(f"Failed to load model registry from {self.registry_path}: {e}")

    def _save_registry(self):
        """
        Saves updated registry metadata to disk.
        """
        os.makedirs(os.path.dirname(os.path.abspath(self.registry_path)), exist_ok=True)
        try:
            with open(self.registry_path, "w") as f:
                json.dump(self.registry_data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")

    def register_model(
        self,
        model_name: str,
        version: str,
        model_path: str,
        metrics: Dict[str, float],
        dataset_version: str,
        dataset_hash: str,
        feature_registry_version: str,
        model_type: str = "lightgbm",
        is_production: bool = False
    ):
        """
        Register a new trained model run in the registry.
        """
        # If setting this run to production, demote previous production runs of the same model_name
        if is_production:
            for m in self.registry_data["models"]:
                if m["model_name"] == model_name:
                    m["is_production"] = False

        record = {
            "model_name": model_name,
            "version": version,
            "model_path": model_path,
            "metrics": metrics,
            "dataset_version": dataset_version,
            "dataset_hash": dataset_hash,
            "feature_registry_version": feature_registry_version,
            "model_type": model_type,
            "is_production": is_production,
            "registered_at": datetime.now().isoformat()
        }
        self.registry_data["models"].append(record)
        self._save_registry()
        logger.info(f"Successfully registered model '{model_name}' version '{version}' at '{model_path}'")

    def get_latest_version(self, model_name: str) -> Optional[str]:
        """
        Helper to find the latest version string registered for a model.
        """
        versions = [
            m["version"] for m in self.registry_data["models"] if m["model_name"] == model_name
        ]
        if not versions:
            return None
        # Sort version strings or use registration order (latest registered is last)
        return versions[-1]

    def load_latest_production(self, model_name: str) -> Optional[BaseTradingModel]:
        """
        Load the latest marked 'production' model of a given type.
        If no production model is explicitly flagged, fallback to the latest registered model.
        """
        candidates = [
            m for m in self.registry_data["models"] if m["model_name"] == model_name and m.get("is_production", False)
        ]

        if not candidates:
            # Fallback to any model under that name
            candidates = [m for m in self.registry_data["models"] if m["model_name"] == model_name]

        if not candidates:
            logger.warning(f"No registered models found for model_name: '{model_name}'")
            return None

        # Pick the latest candidate by registration time / order
        selected_record = candidates[-1]
        model_path = selected_record["model_path"]

        # Infer class by name
        if "MarketStateClassifier" in model_name:
            cls: Type[BaseTradingModel] = MarketStateClassifier
        elif "LevelBreakProbability" in model_name:
            cls = LevelBreakProbabilityModel
        else:
            raise ValueError(f"Unknown model class name: '{model_name}'. Cannot load from registry.")

        logger.info(f"Loading '{model_name}' from path: {model_path}")
        return cls.load(model_path)

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models.
        """
        return self.registry_data["models"]
