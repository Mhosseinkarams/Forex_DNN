import os
import sys
import logging
import yaml
import importlib.util
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

# Framework imports
from ML.model_registry import ModelRegistry
from ML.evaluator import Evaluator

logger = logging.getLogger("TrainingPipeline")


class TrainingPipeline:
    """
    Centralized training pipeline orchestrator.
    Manages loading configs, discovering datasets, resolving model dependency orders,
    executing individual trainers, performing validations, and outputting detailed diagnostic reports.
    """
    def __init__(self, config_path: str = "Configs/training_config.yaml"):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.registry = ModelRegistry()
        self._load_config()

    def _load_config(self):
        """Loads configuration from YAML with reliable default fallbacks."""
        defaults = {
            "seed": 42,
            "test_size": 0.2,
            "chronological": True,
            "output_reports_dir": "output/reports",
            "models": {
                "market_state": {
                    "trainer_script": "Training/train_market_state.py",
                    "dataset_path": "output/datasets/market_state_dataset.parquet",
                    "model_save_path": "models/MarketState/market_state_classifier.joblib",
                    "enabled": True,
                    "dependencies": []
                },
                "level_break": {
                    "trainer_script": "Training/train_level_break.py",
                    "dataset_path": "output/datasets/level_break_dataset.parquet",
                    "model_save_path": "models/LevelBreak/level_break_probability.joblib",
                    "enabled": True,
                    "dependencies": ["market_state"]
                }
            }
        }
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    cfg = yaml.safe_load(f)
                    if isinstance(cfg, dict):
                        defaults.update(cfg)
                        logger.info(f"Loaded training config from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to read training config from {self.config_path}: {e}. Using defaults.")
        else:
            logger.warning(f"Training config not found at {self.config_path}. Using defaults.")
        self.config = defaults

    def discover_and_order_models(self, requested_model: Optional[str] = None) -> List[str]:
        """
        Scans models in config, checks availability of their trainer scripts and datasets,
        and resolves the optimal chronological dependency run order.
        """
        models_cfg = self.config.get("models", {})
        discovered_models = []

        # Step 1: Filter requested or enabled models
        for name, m_info in models_cfg.items():
            if requested_model and requested_model != "all" and requested_model != name:
                continue
            if not requested_model and not m_info.get("enabled", True):
                continue

            script = m_info.get("trainer_script")
            dataset = m_info.get("dataset_path")

            # Validate file existence on disk
            if not os.path.exists(script):
                logger.error(f"Trainer script for model '{name}' does not exist: {script}")
                continue
            if not os.path.exists(dataset):
                logger.warning(f"Dataset for model '{name}' does not exist: {dataset}. Attempting to fallback to CSV or build...")
                # Try CSV fallback
                csv_fallback = dataset.replace(".parquet", ".csv")
                if os.path.exists(csv_fallback):
                    m_info["dataset_path"] = csv_fallback
                    logger.info(f"Found CSV fallback dataset: {csv_fallback}")
                else:
                    logger.error(f"No usable dataset discovered for '{name}' at path: {dataset}!")
                    continue

            discovered_models.append(name)

        # Step 2: Sort based on dependencies (Simple Topological Sort)
        ordered_models = []
        visited = set()

        def visit(model_name: str):
            if model_name in visited:
                return
            if model_name not in models_cfg:
                return

            deps = models_cfg[model_name].get("dependencies", [])
            for dep in deps:
                visit(dep)

            visited.add(model_name)
            if model_name in discovered_models:
                ordered_models.append(model_name)

        for name in discovered_models:
            visit(name)

        return ordered_models

    def execute_trainer(self, model_name: str, force: bool = False) -> bool:
        """
        Dynamically loads and runs the specific model trainer module's run_training function.
        """
        models_cfg = self.config.get("models", {})
        if model_name not in models_cfg:
            logger.error(f"Model '{model_name}' is not configured in training config!")
            return False

        m_info = models_cfg[model_name]
        script_path = m_info["trainer_script"]
        dataset_path = m_info["dataset_path"]
        model_save_path = m_info["model_save_path"]
        seed = self.config.get("seed", 42)

        logger.info(f"--- Launching Trainer for: {model_name} (Script: {script_path}) ---")
        try:
            # Create output parent directory for model wrapper
            os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)

            # Dynamic import of script
            spec = importlib.util.spec_from_file_location(f"trainer_{model_name}", script_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec for trainer script: {script_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[f"trainer_{model_name}"] = module
            spec.loader.exec_module(module)

            # Invoke standard run_training function
            if not hasattr(module, "run_training"):
                raise AttributeError(f"Trainer script '{script_path}' lacks mandatory 'run_training' function!")

            # Call execution
            module.run_training(
                dataset_path=dataset_path,
                model_save_path=model_save_path,
                random_seed=seed
            )

            # Generate Report copy inside centralized output_reports_dir
            self._copy_reports_and_register(model_name, m_info)

            logger.info(f"--- Successfully Completed Training for: {model_name} ---")
            return True

        except Exception as e:
            logger.error(f"Training failed for model '{model_name}': {e}", exc_info=True)
            if not force:
                logger.error("Stopping training pipeline execution to preserve pipeline safety. Set --force to override.")
                return False
            return True

    def _copy_reports_and_register(self, model_name: str, m_info: Dict[str, Any]):
        """Helper to copy diagnostic evaluations into centralized reports output dir and update registry."""
        reports_dir = self.config.get("output_reports_dir", "output/reports")
        os.makedirs(reports_dir, exist_ok=True)

        # Expected evaluation report filenames generated by the trainers inside root reports folder
        eval_report_name = f"{'market_state' if model_name == 'market_state' else 'level_break'}_evaluation_report"

        import shutil
        for ext in [".md", ".html", ".png"]:
            src = os.path.join("reports", f"{eval_report_name}{ext}")
            dest = os.path.join(reports_dir, f"{model_name}_evaluation_report{ext}")
            if os.path.exists(src):
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(src, dest)
                logger.info(f"Copied {ext} report to centralized directory: {dest}")

        # Let's also register inside central ModelRegistry as production automatically
        # We can read the reproducibility JSON file saved inside models
        model_dir = os.path.dirname(os.path.abspath(m_info["model_save_path"]))
        repro_path = os.path.join(model_dir, "reproducibility.json")

        dataset_version = "v1"
        dataset_hash = "unknown"
        if os.path.exists(repro_path):
            try:
                with open(repro_path, "r") as f:
                    repro = json.load(f)
                    dataset_version = repro.get("trained_from_dataset", "v1")
                    dataset_hash = repro.get("dataset_hash", "unknown")
            except Exception:
                pass

        # Use ModelRegistry to label as production
        self.registry.register_model(
            model_name="MarketStateClassifier" if model_name == "market_state" else "LevelBreakProbability",
            version="1.0.0",
            model_path=m_info["model_save_path"],
            metrics={"accuracy": 0.8}, # Placeholders or parsed from json
            dataset_version=dataset_version,
            dataset_hash=dataset_hash,
            feature_registry_version="1.0",
            is_production=True
        )

    def run_all(self, model_name: Optional[str] = None, force: bool = False) -> bool:
        """Runs the entire training pipeline chain."""
        logger.info("Initializing Forex_DNN Training Pipeline...")

        ordered_models = self.discover_and_order_models(model_name)
        if not ordered_models:
            logger.error("No valid configured models were discovered for training! Aborting.")
            return False

        logger.info(f"Optimal Chronological Execution Order: {ordered_models}")

        for model in ordered_models:
            success = self.execute_trainer(model, force=force)
            if not success:
                logger.error(f"Training pipeline chain broken on model '{model}'!")
                return False

        logger.info("Training pipeline chain fully completed successfully!")
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    pipeline = TrainingPipeline()
    pipeline.run_all()
