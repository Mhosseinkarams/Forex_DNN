import unittest
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from Core.trend_context import TrendContext, TrendContextBuilder


class TestTrendContext(unittest.TestCase):
    def setUp(self):
        self.builder = TrendContextBuilder(
            slope_threshold=0.1,
            strong_sep_atr=1.5,
            very_strong_sep_atr=4.0,
            weak_sep_atr=0.5
        )

    def generate_mock_df(self, n_bars=100, trend_type="bull", cross_idx=None, trend_change_idx=None):
        """
        Helper to generate a mock indicator DataFrame.
        """
        datetimes = pd.date_range("2024-01-01", periods=n_bars, freq="5min")

        # We need: Datetime, ema_50, ema_600, ema_slope_600, atr_14, cross_ema_50, dist_ema_50
        df = pd.DataFrame({
            "Datetime": [dt.isoformat() for dt in datetimes],
            "Close": [1.1000] * n_bars,
            "atr_14": [0.0010] * n_bars,
            "cross_ema_50": [0] * n_bars,
            "dist_ema_50": [0.0] * n_bars,
        })

        if trend_type == "bull":
            df["ema_50"] = 1.1051
            df["ema_600"] = 1.1010
            df["ema_slope_600"] = 0.2
        elif trend_type == "bear":
            df["ema_50"] = 1.1009
            df["ema_600"] = 1.1050
            df["ema_slope_600"] = 0.2
        elif trend_type == "weak_flat":
            df["ema_50"] = 1.1010
            df["ema_600"] = 1.1012
            df["ema_slope_600"] = 0.02
        else:
            df["ema_50"] = 1.1010
            df["ema_600"] = 1.1010
            df["ema_slope_600"] = 0.0

        if cross_idx is not None:
            df.loc[cross_idx, "cross_ema_50"] = 1

        if trend_change_idx is not None:
            # Shift ema_50 relative to ema_600 before/after trend_change_idx
            # Before trend_change_idx, let's say trend is Opposite to current
            current_is_bull = trend_type == "bull"
            for i in range(trend_change_idx):
                if current_is_bull:
                    # Opposite is Bear (ema_50 < ema_600)
                    df.loc[i, "ema_50"] = 1.1000
                    df.loc[i, "ema_600"] = 1.1020
                else:
                    # Opposite is Bull (ema_50 > ema_600)
                    df.loc[i, "ema_50"] = 1.1020
                    df.loc[i, "ema_600"] = 1.1000

        return df

    def test_bull_trend_strong_metrics(self):
        # 1.1051 vs 1.1010 = 0.0041. ATR is 0.0010. Distance ATR = 4.1. Slope is 0.2 >= 0.1
        # Thus trend_strength should be 'Very Strong'
        df = self.generate_mock_df(trend_type="bull")
        context = self.builder.build("EURUSD_o", "M5", df)

        self.assertEqual(context.symbol, "EURUSD_o")
        self.assertEqual(context.timeframe, "M5")
        self.assertEqual(context.trend_direction, "Bull")
        self.assertAlmostEqual(context.ema_distance, 0.0041)
        self.assertAlmostEqual(context.ema_distance_atr, 4.1, places=4)
        self.assertEqual(context.trend_strength, "Very Strong")
        self.assertTrue(context.is_strong_trend)
        self.assertFalse(context.is_weak_trend)

    def test_bear_trend_metrics(self):
        df = self.generate_mock_df(trend_type="bear")
        context = self.builder.build("EURUSD_o", "M5", df)

        self.assertEqual(context.trend_direction, "Bear")
        self.assertTrue(context.is_strong_trend)

    def test_weak_trend_metrics(self):
        # 1.1010 vs 1.1012 = 0.0002. ATR is 0.0010. Distance ATR = 0.2. Slope is 0.02 < 0.1
        df = self.generate_mock_df(trend_type="weak_flat")
        context = self.builder.build("EURUSD_o", "M5", df)

        self.assertEqual(context.trend_direction, "Bear")
        self.assertEqual(context.trend_strength, "Weak")
        self.assertFalse(context.is_strong_trend)
        self.assertTrue(context.is_weak_trend)

    def test_bars_since_cross(self):
        # Set a cross at index 80 of 100 bars. Index -1 (99) is 19 bars since cross
        df = self.generate_mock_df(n_bars=100, cross_idx=80)
        context = self.builder.build("EURUSD_o", "M5", df, idx=-1)
        self.assertEqual(context.bars_since_cross, 19)

    def test_bars_since_trend_change(self):
        # Set a trend change at index 75 of 100 bars. Index -1 (99) is 24 bars since trend change
        df = self.generate_mock_df(n_bars=100, trend_type="bull", trend_change_idx=75)
        context = self.builder.build("EURUSD_o", "M5", df, idx=-1)
        self.assertEqual(context.bars_since_trend_change, 24)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            self.builder.build("EURUSD_o", "M5", None)

        df = self.generate_mock_df(n_bars=5)
        with self.assertRaises(IndexError):
            self.builder.build("EURUSD_o", "M5", df, idx=10)


if __name__ == "__main__":
    unittest.main()
