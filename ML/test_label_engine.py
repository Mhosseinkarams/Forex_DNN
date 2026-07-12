import unittest
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from ML.label_engine import LabelEngine, BaseLabeler
from ML.market_state_labeler import MarketStateLabeler
from ML.dataset_validator import DatasetValidator
from scripts.generate_dataset import generate_synthetic_ohlcv


class DummyLabeler(BaseLabeler):
    """Simple labeler that returns TREND or None for testing purposes."""
    def __init__(self, filter_even: bool = False):
        self.filter_even = filter_even
        self.label_version = "test_1.0"
        self.engine_version = "test_1.0"

    def label_window(self, df, msg, start, end):
        if self.filter_even and end % 2 == 0:
            return None, 0.0, "filter_even_index"
        return "TREND", 0.9, "always_trend"


class TestLabelEngine(unittest.TestCase):
    def setUp(self):
        # Generate short 150 bars dataset
        self.df = generate_synthetic_ohlcv(n_bars=150)
        self.temp_csv_path = "output/test_label_engine_output.csv"
        self.temp_manifest_path = self.temp_csv_path + ".manifest.json"

    def tearDown(self):
        for path in [self.temp_csv_path, self.temp_manifest_path]:
            if os.path.exists(path):
                os.remove(path)

    def test_sliding_window_indices_and_stride(self):
        # window_size=35, window_stride=5
        engine = LabelEngine(window_size=35, window_stride=5)
        labeler = DummyLabeler(filter_even=False)

        res_df = engine.generate(self.df, "EURUSD", "M5", labeler)

        # Total rows should be (len(df) - window_size) / stride + 1 = (150 - 35)/5 + 1 = 115/5 + 1 = 24 rows
        # But wait, we have a warmup check in raw IndicatorEngine which might drop some rows or the dataset is long enough
        # Let's check that the generated windows are consecutive and start indices have stride of 5
        self.assertFalse(res_df.empty)
        self.assertEqual(len(res_df), 24)

        start_indices = res_df["window_start"].tolist()
        for i in range(1, len(start_indices)):
            self.assertEqual(start_indices[i] - start_indices[i-1], 5)

    def test_handling_unlabeled_samples(self):
        # DummyLabeler with filter_even=True should remove half the samples
        engine = LabelEngine(window_size=35, window_stride=1)
        labeler = DummyLabeler(filter_even=True)

        res_df = engine.generate(self.df, "EURUSD", "M5", labeler)

        # Verify that all retained rows have odd window_end index
        end_indices = res_df["window_end"].tolist()
        for idx in end_indices:
            self.assertTrue(idx % 2 != 0)

    def test_market_state_labeling_rules(self):
        # Let's verify MarketStateLabeler rules directly on pre-formed df and msg
        # Since we want to test TREND, let's inject a very strong trend
        # By setting high separation and a confirmed BOS
        df_trend = generate_synthetic_ohlcv(n_bars=1200)
        labeler = MarketStateLabeler(min_confidence=0.4)
        engine = LabelEngine(window_size=35, window_stride=1)

        res_df = engine.generate(df_trend, "EURUSD", "M5", labeler)

        # We expect TREND to be detected on the upward trending part of the synthetic dataset (index > 600)
        # Verify that we generated some classes
        self.assertFalse(res_df.empty)
        unique_labels = res_df["label"].unique()
        self.assertTrue(len(unique_labels) > 0)
        self.assertTrue(all(lbl in ["TREND", "RANGE", "TRANSITION"] for lbl in unique_labels))

    def test_dataset_validator(self):
        validator = DatasetValidator(expected_window_size=35)

        # 1. Valid DataFrame
        engine = LabelEngine(window_size=35, window_stride=1)
        labeler = DummyLabeler()
        df_valid = engine.generate(self.df, "EURUSD", "M5", labeler)

        report = validator.validate(df_valid)
        self.assertTrue(report["is_valid"])
        self.assertEqual(report["checks"]["missing_values"], "PASSED")
        self.assertEqual(report["checks"]["duplicate_samples"], "PASSED")
        self.assertEqual(report["checks"]["window_consistency"], "PASSED")

        # 2. Duplicate Check
        df_dup = pd.concat([df_valid, df_valid.iloc[[0]]]).reset_index(drop=True)
        report_dup = validator.validate(df_dup)
        self.assertFalse(report_dup["is_valid"])
        self.assertEqual(report_dup["checks"]["duplicate_samples"], "FAILED")

        # 3. Missing values Check
        df_missing = df_valid.copy()
        df_missing.loc[0, "label"] = None
        report_missing = validator.validate(df_missing)
        self.assertFalse(report_missing["is_valid"])
        self.assertEqual(report_missing["checks"]["missing_values"], "FAILED")

        # 4. Window size mismatch Check
        validator_wrong_size = DatasetValidator(expected_window_size=50)
        report_wrong_size = validator_wrong_size.validate(df_valid)
        self.assertFalse(report_wrong_size["is_valid"])
        self.assertEqual(report_wrong_size["checks"]["window_consistency"], "FAILED")

    def test_dataset_manifest_and_output(self):
        engine = LabelEngine(window_size=35, window_stride=1)
        labeler = DummyLabeler()

        res_df = engine.generate(
            df=self.df,
            symbol="EURUSD",
            timeframe="M5",
            labeler=labeler,
            output_csv_path=self.temp_csv_path
        )

        # Verify files were created
        self.assertTrue(os.path.exists(self.temp_csv_path))
        self.assertTrue(os.path.exists(self.temp_manifest_path))

        # Verify manifest JSON fields
        with open(self.temp_manifest_path, "r") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["window_size"], 35)
        self.assertEqual(manifest["window_stride"], 1)
        self.assertEqual(manifest["label_version"], "test_1.0")
        self.assertEqual(manifest["symbols"], ["EURUSD"])
        self.assertEqual(manifest["timeframes"], ["M5"])
        self.assertTrue("feature_registry_version" in manifest)
        self.assertTrue("final_class_distribution" in manifest)


if __name__ == "__main__":
    unittest.main()
