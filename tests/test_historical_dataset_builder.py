import os
import shutil
import unittest
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta, timezone
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.getcwd(), '..')) if os.path.basename(os.getcwd()) == 'examples' else os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ML.feature_registry import FeatureRegistry
from ML.dataset_validator import ValidationReport
from Market_Data_Pipeline.historical_dataset_builder import HistoricalDatasetBuilder, DatasetVersionManager

class TestHistoricalDatasetBuilder(unittest.TestCase):
    def setUp(self):
        # Create temp folders
        self.test_input_dir = "test_historical_inputs"
        self.test_output_dir = "test_historical_outputs"
        os.makedirs(self.test_input_dir, exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)

        # Clean cache directory if exists
        if os.path.exists("cache"):
            shutil.rmtree("cache")

        # Generate some synthetic data for a couple of symbols
        self.symbols = ["XAUUSD", "YM"]
        self.timeframe = "M5"
        self.num_bars = 200

        # Create synthetic CSV/Parquet files
        for sym in self.symbols:
            # We can use nested structure
            sym_dir = os.path.join(self.test_input_dir, sym)
            os.makedirs(sym_dir, exist_ok=True)

            df = self._generate_synthetic_candles(self.num_bars)
            parquet_path = os.path.join(sym_dir, f"{self.timeframe}.parquet")
            df.to_parquet(parquet_path, index=False)

    def tearDown(self):
        # Clean up temp folders
        if os.path.exists(self.test_input_dir):
            shutil.rmtree(self.test_input_dir)
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)
        if os.path.exists("cache"):
            shutil.rmtree("cache")
        if os.path.exists("datasets"):
            shutil.rmtree("datasets")

    def _generate_synthetic_candles(self, num_bars: int) -> pd.DataFrame:
        np.random.seed(42)
        start_date = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        datetimes = [start_date + timedelta(minutes=5 * i) for i in range(num_bars)]
        prices = 1.1000 + np.cumsum(np.random.randn(num_bars) * 0.0002)

        opens = np.zeros(num_bars)
        highs = np.zeros(num_bars)
        lows = np.zeros(num_bars)
        closes = np.zeros(num_bars)
        volumes = np.random.randint(100, 1500, size=num_bars).astype(float)
        spreads = np.random.randint(1, 5, size=num_bars).astype(float)

        for i in range(num_bars):
            if i == 0:
                opens[i] = prices[i]
            else:
                opens[i] = closes[i-1]
            closes[i] = prices[i]
            highs[i] = max(opens[i], closes[i]) + abs(np.random.normal(0.0005, 0.0002))
            lows[i] = min(opens[i], closes[i]) - abs(np.random.normal(0.0005, 0.0002))

        df = pd.DataFrame({
            "Datetime": datetimes,
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "TickVolume": volumes,
            "Spread": spreads
        })

        # Pre-calculate Indicators to satisfy label engines
        df["ema_50"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["ema_600"] = df["Close"].ewm(span=600, adjust=False).mean()
        df["ema_800"] = df["Close"].ewm(span=800, adjust=False).mean()

        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift(1)).abs()
        low_close = (df["Low"] - df["Close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(window=14).mean().fillna(0.0001)

        # Add indicators needed by standard features
        df["ema_slope_50"] = (df["ema_50"] - df["ema_50"].shift(32)).abs() / df["atr_14"]
        df["ema_slope_600"] = (df["ema_600"] - df["ema_600"].shift(32)).abs() / df["atr_14"]
        df["ema_slope_800"] = (df["ema_800"] - df["ema_800"].shift(32)).abs() / df["atr_14"]
        df["dist_ema_50"] = (df["Close"] - df["ema_50"]) / df["atr_14"]
        df["dist_ema_600"] = (df["Close"] - df["ema_600"]) / df["atr_14"]
        df["dist_ema_800"] = (df["Close"] - df["ema_800"]) / df["atr_14"]
        df["body_size"] = (df["Close"] - df["Open"]).abs()
        df["avg_body_size"] = df["body_size"].rolling(window=20, min_periods=1).mean()
        df["body_vs_avg"] = np.where(df["avg_body_size"] != 0, df["body_size"] / df["avg_body_size"], 0.0)
        df["candle_size"] = abs(df["High"] - df["Low"])
        df["body_pct"] = np.where(df["candle_size"] != 0, df["body_size"] / df["candle_size"], 0.0)
        df["upper_shadow"] = df["High"] - df[["Open", "Close"]].max(axis=1)
        df["lower_shadow"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
        df["total_shadow"] = df["upper_shadow"] + df["lower_shadow"]
        df["body_shadow_ratio"] = df["body_size"] / (df["total_shadow"] + 1e-9)
        df["rolling_body_shadow_ratio"] = df["body_shadow_ratio"].rolling(window=5, min_periods=1).mean()
        df["candle_direction"] = np.where(df["Close"] > df["Open"], 1, np.where(df["Close"] < df["Open"], -1, 0))

        return df

    def test_loads_symbols(self):
        builder = HistoricalDatasetBuilder(
            input_dir=self.test_input_dir,
            output_dir=self.test_output_dir,
            timeframe=self.timeframe
        )
        files = builder.find_files()
        self.assertIn("XAUUSD", files)
        self.assertIn("YM", files)
        self.assertEqual(len(files), 2)

    def test_generates_rolling_windows(self):
        builder = HistoricalDatasetBuilder(
            input_dir=self.test_input_dir,
            output_dir=self.test_output_dir,
            window_size=35,
            timeframe=self.timeframe
        )
        files = builder.find_files()
        df_labeled = builder.process_symbol("XAUUSD", files["XAUUSD"])

        # Check standard columns
        self.assertFalse(df_labeled.empty)
        self.assertIn("symbol", df_labeled.columns)
        self.assertIn("timeframe", df_labeled.columns)
        self.assertIn("target", df_labeled.columns)
        self.assertIn("confidence", df_labeled.columns)

    def test_feature_registry_integration(self):
        builder = HistoricalDatasetBuilder(
            input_dir=self.test_input_dir,
            output_dir=self.test_output_dir,
            timeframe=self.timeframe
        )
        enabled_feats = [f.name for f in builder.registry.list_enabled()]
        self.assertIn("ema50_slope", enabled_feats)
        self.assertIn("risk_reward_estimate", enabled_feats)

    def test_engines_integration_and_metadata_generation(self):
        builder = HistoricalDatasetBuilder(
            input_dir=self.test_input_dir,
            output_dir=self.test_output_dir,
            window_size=35,
            timeframe=self.timeframe,
            version="v001"
        )
        df_final, metadata = builder.build_dataset(max_workers=2)

        self.assertFalse(df_final.empty)
        self.assertEqual(metadata["version"], "v001")
        self.assertEqual(metadata["window_size"], 35)
        self.assertEqual(metadata["timeframe"], self.timeframe)
        self.assertEqual(metadata["sample_count"], len(df_final))
        self.assertIn("XAUUSD", metadata["symbols"])
        self.assertIn("YM", metadata["symbols"])

        # Check metadata json saved
        meta_path = os.path.join(self.test_output_dir, "dataset_v001_metadata.json")
        self.assertTrue(os.path.exists(meta_path))
        with open(meta_path, "r") as f:
            saved_metadata = json.load(f)
        self.assertEqual(saved_metadata["version"], "v001")

    def test_parquet_and_csv_saving(self):
        builder = HistoricalDatasetBuilder(
            input_dir=self.test_input_dir,
            output_dir=self.test_output_dir,
            window_size=35,
            timeframe=self.timeframe,
            version="v002"
        )
        df_final, metadata = builder.build_dataset(max_workers=1)

        # Compatibility files
        parquet_path = os.path.join(self.test_output_dir, "dataset_v002.parquet")
        csv_path = os.path.join(self.test_output_dir, "dataset_v002.csv")

        self.assertTrue(os.path.exists(parquet_path))
        self.assertTrue(os.path.exists(csv_path))

        # Check they load properly
        df_parquet = pd.read_parquet(parquet_path)
        df_csv = pd.read_csv(csv_path)

        self.assertEqual(len(df_parquet), len(df_final))
        self.assertEqual(len(df_csv), len(df_final))

    def test_validation_report(self):
        builder = HistoricalDatasetBuilder(
            input_dir=self.test_input_dir,
            output_dir=self.test_output_dir,
            window_size=35,
            timeframe=self.timeframe,
            version="v003"
        )
        df_final, metadata = builder.build_dataset(max_workers=1)

        self.assertIn("validation", metadata)
        self.assertIn("is_valid", metadata["validation"])
        self.assertIn("nan_count", metadata["validation"])
        self.assertIn("duplicate_rows", metadata["validation"])
        self.assertIn("memory_usage_mb", metadata["validation"])

    def test_cache_and_resume(self):
        builder = HistoricalDatasetBuilder(
            input_dir=self.test_input_dir,
            output_dir=self.test_output_dir,
            window_size=35,
            timeframe=self.timeframe,
            version="v004"
        )
        # Execute 1: Computes and caches
        df1, meta1 = builder.build_dataset(max_workers=1)
        cache_dir = builder._get_cache_dir()
        self.assertTrue(os.path.exists(cache_dir))
        self.assertTrue(os.path.exists(os.path.join(cache_dir, "XAUUSD_processed.parquet")))

        # Execute 2: Should hit cache
        df2, meta2 = builder.build_dataset(max_workers=1)
        self.assertEqual(len(df1), len(df2))

    def test_dataset_versioning_manager(self):
        manager = DatasetVersionManager()
        next_v = manager.resolve_next_version()
        self.assertRegex(next_v, r"^v\d{3}$")

        version_dir = manager.get_version_dir(next_v)
        os.makedirs(version_dir, exist_ok=True)
        # Resolved next version should increment now
        next_v2 = manager.resolve_next_version()
        self.assertNotEqual(next_v, next_v2)

    def test_deterministic_sample_ids(self):
        builder = HistoricalDatasetBuilder(
            input_dir=self.test_input_dir,
            output_dir=self.test_output_dir,
            window_size=35,
            timeframe=self.timeframe,
            version="v005"
        )
        df, meta = builder.build_dataset(max_workers=1)
        self.assertIn("sample_id", df.columns)
        first_id = df["sample_id"].iloc[0]
        # Should look like: XAUUSD_M5_2026-01-01T00-00_2026-01-01T02-50
        self.assertRegex(first_id, r"^[A-Z0-9]+_M5_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}$")

    def test_engine_registration_plugin(self):
        class MockCustomEngine:
            called = False
            def process(self, df: pd.DataFrame) -> pd.DataFrame:
                df = df.copy()
                df["custom_plugin_column"] = 9.99
                MockCustomEngine.called = True
                return df

        builder = HistoricalDatasetBuilder(
            input_dir=self.test_input_dir,
            output_dir=self.test_output_dir,
            window_size=35,
            timeframe=self.timeframe,
            version="v006"
        )
        engine = MockCustomEngine()
        builder.register_engine(engine)

        # Verify it was inserted correctly into the pipeline steps
        self.assertIn(engine, builder.pipeline.steps)
        df, meta = builder.build_dataset(max_workers=1)
        self.assertTrue(MockCustomEngine.called)
        self.assertIn("custom_plugin_column", df.columns)

    def test_feature_and_label_snapshots(self):
        builder = HistoricalDatasetBuilder(
            input_dir=self.test_input_dir,
            output_dir=self.test_output_dir,
            window_size=35,
            timeframe=self.timeframe,
            version="v007"
        )
        df, meta = builder.build_dataset(max_workers=1)

        v_dir = os.path.join("datasets", "v007")
        self.assertTrue(os.path.exists(os.path.join(v_dir, "feature_registry.json")))
        self.assertTrue(os.path.exists(os.path.join(v_dir, "label_config.json")))
        self.assertTrue(os.path.exists(os.path.join(v_dir, "statistics.json")))
        self.assertTrue(os.path.exists(os.path.join(v_dir, "manifest.json")))
        self.assertTrue(os.path.exists(os.path.join(v_dir, "engine_versions.json")))

if __name__ == "__main__":
    unittest.main()
