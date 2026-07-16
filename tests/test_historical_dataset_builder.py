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
from Market_Data_Pipeline.historical_dataset_builder import HistoricalDatasetBuilder

class TestHistoricalDatasetBuilder(unittest.TestCase):
    def setUp(self):
        # Create temp folders
        self.test_input_dir = "test_historical_inputs"
        self.test_output_dir = "test_historical_outputs"
        self.test_datasets_dir = "test_historical_datasets"
        self.test_cache_dir = "test_historical_cache"

        os.makedirs(self.test_input_dir, exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)
        os.makedirs(self.test_datasets_dir, exist_ok=True)
        os.makedirs(self.test_cache_dir, exist_ok=True)

        # Generate some synthetic data for a couple of symbols
        self.symbols = ["XAUUSD", "YM"]
        self.timeframe = "M5"
        self.num_bars = 200

        # Create synthetic CSV/Parquet files
        for sym in self.symbols:
            sym_dir = os.path.join(self.test_input_dir, sym)
            os.makedirs(sym_dir, exist_ok=True)

            df = self._generate_synthetic_candles(self.num_bars)
            parquet_path = os.path.join(sym_dir, f"{self.timeframe}.parquet")
            df.to_parquet(parquet_path, index=False)

    def tearDown(self):
        # Clean up temp folders
        for folder in [self.test_input_dir, self.test_output_dir, self.test_datasets_dir, self.test_cache_dir]:
            if os.path.exists(folder):
                shutil.rmtree(folder)

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
            timeframe=self.timeframe,
            cache_dir=self.test_cache_dir,
            datasets_dir=self.test_datasets_dir
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
            timeframe=self.timeframe,
            cache_dir=self.test_cache_dir,
            datasets_dir=self.test_datasets_dir
        )
        files = builder.find_files()
        df_labeled = builder.process_symbol("XAUUSD", files["XAUUSD"])

        self.assertFalse(df_labeled.empty)
        self.assertIn("symbol", df_labeled.columns)
        self.assertIn("timeframe", df_labeled.columns)
        self.assertIn("target", df_labeled.columns)
        self.assertIn("confidence", df_labeled.columns)

    def test_feature_registry_integration(self):
        builder = HistoricalDatasetBuilder(
            input_dir=self.test_input_dir,
            output_dir=self.test_output_dir,
            timeframe=self.timeframe,
            cache_dir=self.test_cache_dir,
            datasets_dir=self.test_datasets_dir
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
            version="v001",
            cache_dir=self.test_cache_dir,
            datasets_dir=self.test_datasets_dir
        )
        df_final, metadata = builder.build_dataset(max_workers=2)

        self.assertFalse(df_final.empty)
        self.assertEqual(metadata["version"], "v001")
        self.assertEqual(metadata["window_size"], 35)
        self.assertEqual(metadata["timeframe"], self.timeframe)
        self.assertEqual(metadata["sample_count"], len(df_final))
        self.assertIn("XAUUSD", metadata["symbols"])
        self.assertIn("YM", metadata["symbols"])

        meta_path = os.path.join(self.test_datasets_dir, "v001", "metadata.json")
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
            version="v002",
            cache_dir=self.test_cache_dir,
            datasets_dir=self.test_datasets_dir
        )
        df_final, metadata = builder.build_dataset(max_workers=1)

        parquet_path = os.path.join(self.test_datasets_dir, "v002", "dataset.parquet")
        csv_path = os.path.join(self.test_datasets_dir, "v002", "dataset.csv")

        self.assertTrue(os.path.exists(parquet_path))
        self.assertTrue(os.path.exists(csv_path))

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
            version="v003",
            cache_dir=self.test_cache_dir,
            datasets_dir=self.test_datasets_dir
        )
        df_final, metadata = builder.build_dataset(max_workers=1)

        self.assertIn("validation", metadata)
        self.assertIn("is_valid", metadata["validation"])
        self.assertIn("nan_count", metadata["validation"])
        self.assertIn("duplicate_rows", metadata["validation"])
        self.assertIn("memory_usage_mb", metadata["validation"])

    def test_caching_and_resume(self):
        # 1. First run processes and caches
        builder1 = HistoricalDatasetBuilder(
            input_dir=self.test_input_dir,
            output_dir=self.test_output_dir,
            window_size=35,
            timeframe=self.timeframe,
            version="v004",
            cache_dir=self.test_cache_dir,
            datasets_dir=self.test_datasets_dir
        )
        df1, _ = builder1.build_dataset(max_workers=1)
        self.assertFalse(df1.empty)

        # Check cache file exists
        cache_file = os.path.join(self.test_cache_dir, f"XAUUSD_{self.timeframe}_v004_cache.parquet")
        self.assertTrue(os.path.exists(cache_file))

        # 2. Second run loads from cache (extremely fast!)
        builder2 = HistoricalDatasetBuilder(
            input_dir=self.test_input_dir,
            output_dir=self.test_output_dir,
            window_size=35,
            timeframe=self.timeframe,
            version="v004",
            cache_dir=self.test_cache_dir,
            datasets_dir=self.test_datasets_dir
        )
        df2, _ = builder2.build_dataset(max_workers=1)
        self.assertEqual(len(df1), len(df2))

    def test_version_increment(self):
        builder1 = HistoricalDatasetBuilder(
            input_dir=self.test_input_dir,
            output_dir=self.test_output_dir,
            window_size=35,
            timeframe=self.timeframe,
            cache_dir=self.test_cache_dir,
            datasets_dir=self.test_datasets_dir
        )
        df1, meta1 = builder1.build_dataset(max_workers=1)
        self.assertEqual(meta1["version"], "v001")

        builder2 = HistoricalDatasetBuilder(
            input_dir=self.test_input_dir,
            output_dir=self.test_output_dir,
            window_size=35,
            timeframe=self.timeframe,
            cache_dir=self.test_cache_dir,
            datasets_dir=self.test_datasets_dir
        )
        df2, meta2 = builder2.build_dataset(max_workers=1)
        self.assertEqual(meta2["version"], "v002")

    def test_engine_plugin_registration(self):
        class MockCustomEngine:
            def __init__(self):
                self.executed = False
            def process(self, df):
                self.executed = True
                df["custom_plugin_col"] = 42.0
                return df

        builder = HistoricalDatasetBuilder(
            input_dir=self.test_input_dir,
            output_dir=self.test_output_dir,
            timeframe=self.timeframe,
            cache_dir=self.test_cache_dir,
            datasets_dir=self.test_datasets_dir
        )
        plugin = MockCustomEngine()
        builder.register_engine(plugin)

        # Verify it got added to pipeline
        self.assertIn(plugin, builder.pipeline.stages)

    def test_deterministic_sample_ids(self):
        builder = HistoricalDatasetBuilder(
            input_dir=self.test_input_dir,
            output_dir=self.test_output_dir,
            window_size=35,
            timeframe=self.timeframe,
            cache_dir=self.test_cache_dir,
            datasets_dir=self.test_datasets_dir
        )
        df, _ = builder.build_dataset(max_workers=1)

        self.assertIn("sample_id", df.columns)
        first_sample_id = df.iloc[0]["sample_id"]
        # Pattern check, e.g., XAUUSD_M5_2026-01-01T02:50
        self.assertTrue(first_sample_id.startswith("XAUUSD_M5_2026-01-"))

    def test_manifest_and_statistics(self):
        builder = HistoricalDatasetBuilder(
            input_dir=self.test_input_dir,
            output_dir=self.test_output_dir,
            window_size=35,
            timeframe=self.timeframe,
            version="v005",
            cache_dir=self.test_cache_dir,
            datasets_dir=self.test_datasets_dir
        )
        builder.build_dataset(max_workers=1)

        v_dir = os.path.join(self.test_datasets_dir, "v005")

        # Verify existence of the 8 required files
        self.assertTrue(os.path.exists(os.path.join(v_dir, "dataset.parquet")))
        self.assertTrue(os.path.exists(os.path.join(v_dir, "dataset.csv")))
        self.assertTrue(os.path.exists(os.path.join(v_dir, "metadata.json")))
        self.assertTrue(os.path.exists(os.path.join(v_dir, "feature_registry.json")))
        self.assertTrue(os.path.exists(os.path.join(v_dir, "engine_versions.json")))
        self.assertTrue(os.path.exists(os.path.join(v_dir, "label_config.json")))
        self.assertTrue(os.path.exists(os.path.join(v_dir, "statistics.json")))
        self.assertTrue(os.path.exists(os.path.join(v_dir, "manifest.json")))

        with open(os.path.join(v_dir, "manifest.json"), "r") as f:
            manifest = json.load(f)
            self.assertEqual(manifest["version"], "v005")
            self.assertEqual(manifest["window_size"], 35)

        with open(os.path.join(v_dir, "statistics.json"), "r") as f:
            stats = json.load(f)
            self.assertIn("Rows", stats)
            self.assertIn("Columns", stats)
            self.assertIn("Features", stats)
            self.assertIn("Generation_Time_Sec", stats)

    def test_fingerprints_in_output(self):
        builder = HistoricalDatasetBuilder(
            input_dir=self.test_input_dir,
            output_dir=self.test_output_dir,
            window_size=35,
            timeframe=self.timeframe,
            version="v006",
            cache_dir=self.test_cache_dir,
            datasets_dir=self.test_datasets_dir
        )
        _, metadata = builder.build_dataset(max_workers=1)

        # Check that fingerprints exist
        self.assertIn("fingerprint", metadata)
        fingerprint = metadata["fingerprint"]
        self.assertIn("dataset_hash", fingerprint)
        self.assertIn("feature_hash", fingerprint)
        self.assertIn("engine_hash", fingerprint)
        self.assertIn("git_commit", fingerprint)
        self.assertIn("creation_time", fingerprint)

    def test_html_quality_report_saved(self):
        builder = HistoricalDatasetBuilder(
            input_dir=self.test_input_dir,
            output_dir=self.test_output_dir,
            window_size=35,
            timeframe=self.timeframe,
            version="v007",
            cache_dir=self.test_cache_dir,
            datasets_dir=self.test_datasets_dir
        )
        builder.build_dataset(max_workers=1)

        report_path = os.path.join(self.test_datasets_dir, "v007", "dataset_quality_report.html")
        self.assertTrue(os.path.exists(report_path))
        with open(report_path, "r") as f:
            html = f.read()
            self.assertIn("Dataset Quality & Health Report", html)
            self.assertIn("dataset_hash", html)
            self.assertIn("feature_hash", html)

    def test_directory_layout_initialized(self):
        from Configs.path_manager import PathManager
        builder = HistoricalDatasetBuilder(
            input_dir=self.test_input_dir,
            output_dir=self.test_output_dir,
            timeframe=self.timeframe,
            cache_dir=self.test_cache_dir,
            datasets_dir=self.test_datasets_dir
        )
        for folder in PathManager.PATHS.values():
            self.assertTrue(os.path.exists(folder))

if __name__ == "__main__":
    unittest.main()
