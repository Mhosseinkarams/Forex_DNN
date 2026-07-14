import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from ML.models.market_state_classifier import MarketStateClassifier
from ML.models.level_break_probability import LevelBreakProbabilityModel
from ML.model_registry import ModelRegistry
from ML.trainer import Trainer
from ML.evaluator import Evaluator


class TestProductionMLFramework(unittest.TestCase):
    def setUp(self):
        # Create a temp directory for models, configs, registries, and reports
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = os.path.join(self.temp_dir, "model_registry.json")
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")

        # Write test config YAML file
        with open(self.config_path, "w") as f:
            f.write("""
hyperparameters:
  n_estimators: 10
  learning_rate: 0.1
  max_depth: 3
  num_leaves: 7
""")

        # Generate dummy data
        np.random.seed(42)
        n_samples = 150
        self.feature_cols = [f"feature_{i}" for i in range(10)]
        data = {f: np.random.randn(n_samples) for f in self.feature_cols}
        self.df = pd.DataFrame(data)

        # Add targets
        self.df["market_state_label"] = np.random.choice(["TREND", "RANGE", "TRANSITION"], n_samples)
        self.df["level_break_label"] = np.random.choice([0, 1], n_samples)
        self.df["timestamp"] = pd.date_range("2024-01-01", periods=n_samples, freq="5min")

    def tearDown(self):
        # Clean up temp folder
        shutil.rmtree(self.temp_dir)

    def test_configuration_loading(self):
        # Test loading from config file
        model = MarketStateClassifier(
            model_type="lightgbm",
            config_path=self.config_path,
            random_state=42
        )
        self.assertEqual(model.hyperparameters["n_estimators"], 10)
        self.assertEqual(model.hyperparameters["max_depth"], 3)
        self.assertEqual(model.hyperparameters["num_leaves"], 7)

    def test_fit_save_load_predict(self):
        # 1. MarketStateClassifier Fit, Save, Load, Predict
        model_save_path = os.path.join(self.temp_dir, "market_state.joblib")

        model = MarketStateClassifier(model_type="lightgbm", random_state=42)
        X_train = self.df[self.feature_cols][:100]
        y_train = self.df["market_state_label"][:100].map({"TREND": 0, "RANGE": 1, "TRANSITION": 2})

        model.fit(X_train, y_train, feature_names=self.feature_cols, dataset_version="test-v1", dataset_hash="hash123")

        # Verify metadata
        self.assertEqual(model.metadata["dataset_version"], "test-v1")
        self.assertEqual(model.metadata["dataset_hash"], "hash123")
        self.assertEqual(model.metadata["feature_count"], 10)
        self.assertEqual(model.metadata["training_samples"], 100)

        # Save model
        model.save(model_save_path)
        self.assertTrue(os.path.exists(model_save_path))
        self.assertTrue(os.path.exists(model_save_path.replace(".joblib", "_metadata.json")))

        # Load model back
        loaded = MarketStateClassifier.load(model_save_path)
        self.assertEqual(loaded.metadata["dataset_version"], "test-v1")
        self.assertEqual(loaded.metadata["training_samples"], 100)

        # Perform predict on loaded model
        single_row = self.df[self.feature_cols].iloc[-1].to_dict()
        pred = loaded.predict(single_row)

        self.assertIn(pred.regime, ["TREND", "RANGE", "TRANSITION"])
        self.assertTrue(0.0 <= pred.trend_probability <= 1.0)
        self.assertTrue(0.0 <= pred.confidence <= 1.0)

        # Probabilities must be assigned by the fitted class IDs, not by
        # alphabetical target-encoding order.
        expected = loaded.model.predict_proba(pd.DataFrame([single_row]))[0]
        expected_by_class = dict(zip(loaded.model.classes_, expected))
        self.assertAlmostEqual(pred.trend_probability, expected_by_class.get(0, 0.0))
        self.assertAlmostEqual(pred.range_probability, expected_by_class.get(1, 0.0))
        self.assertAlmostEqual(pred.transition_probability, expected_by_class.get(2, 0.0))

        # Check predict_proba
        proba_dict = loaded.predict_proba(single_row)
        self.assertIn("TREND", proba_dict)
        self.assertIn("confidence", proba_dict)

    def test_probability_calibration(self):
        # Fit model
        model = LevelBreakProbabilityModel(model_type="randomforest", random_state=42)
        X_train = self.df[self.feature_cols][:100]
        y_train = self.df["level_break_label"][:100]

        model.fit(X_train, y_train, feature_names=self.feature_cols)

        X_val = self.df[self.feature_cols][100:130]
        y_val = self.df["level_break_label"][100:130].to_numpy()

        # Calibrate
        model.calibrate(X_val, y_val, method="sigmoid")
        self.assertIsNotNone(model.calibrated_model)
        self.assertEqual(model.metadata["calibration_method"], "sigmoid")

        # Predict on calibrated model
        pred = model.predict(self.df[self.feature_cols].iloc[-1].to_dict())
        self.assertTrue(0.0 <= pred.break_probability <= 1.0)
        self.assertTrue(0.0 <= pred.confidence <= 1.0)

    def test_model_registry(self):
        registry = ModelRegistry(registry_path=self.registry_path)

        # Register a fake run
        registry.register_model(
            model_name="MarketStateClassifier",
            version="1.0.1",
            model_path=os.path.join(self.temp_dir, "fake_model.joblib"),
            metrics={"accuracy": 0.85},
            dataset_version="v001",
            dataset_hash="def456",
            feature_registry_version="hash_abc",
            model_type="lightgbm",
            is_production=True
        )

        # Verify latest version and list
        self.assertEqual(registry.get_latest_version("MarketStateClassifier"), "1.0.1")
        models = registry.list_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["version"], "1.0.1")
        self.assertTrue(models[0]["is_production"])

    def test_trainer_and_evaluator(self):
        model = MarketStateClassifier(model_type="lightgbm", random_state=42)

        # Use Trainer
        trainer = Trainer(random_seed=42, model_registry_path=self.registry_path)
        save_path = os.path.join(self.temp_dir, "trainer_model.joblib")

        results = trainer.train_model(
            model=model,
            df=self.df,
            target_col="market_state_label",
            feature_cols=self.feature_cols,
            test_size=0.2,
            chronological=True,
            dataset_version="test-trainer",
            dataset_hash="hash-trainer",
            model_save_path=save_path,
            is_production=True,
            version="1.2.3"
        )

        self.assertIn("model", results)
        self.assertIn("metrics", results)
        self.assertTrue(os.path.exists(save_path))

        # Test Registry entry was added automatically
        reg = ModelRegistry(registry_path=self.registry_path)
        self.assertEqual(reg.get_latest_version("MarketStateClassifier"), "1.2.3")

        # Use Evaluator
        evaluator = Evaluator(output_dir=os.path.join(self.temp_dir, "reports"))
        report_data = evaluator.evaluate_and_report(
            model=model,
            X_val=results["X_val"],
            y_val=results["y_val"],
            classes=["TREND", "RANGE", "TRANSITION"],
            report_name="test_report"
        )

        self.assertIn("accuracy", report_data)
        self.assertIn("class_distribution", report_data)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "reports", "test_report.md")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "reports", "test_report.html")))
