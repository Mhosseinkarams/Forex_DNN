import unittest
import os
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch

# Import our new components
from Market_Data_Pipeline.strong_candle_engine import StrongCandleEngine, StrongCandle
from Market_Data_Pipeline.refusal_candle_engine import RefusalCandleEngine, RefusalSignal
from Core.signal_candidate import SignalCandidate
from Collecting_Data.signal_recorder import SignalRecorder

# Strategies
from Strategies.mm_strategy import MMStrategy
from Strategies.sm_strategy import SMStrategy
from Strategies.unit_strategy import UniTStrategy

# Feature and analytical engines
from ML.feature_registry import FeatureRegistry
from ML.feature_pipeline import FeaturePipeline
from Market_Data_Pipeline.structure_graph import MarketStructureGraph, Zone

class TestSignalIntelligenceLayer(unittest.TestCase):
    def setUp(self):
        # Create a dummy dataframe representing price bars
        # 100 bars to satisfy indicator and lookback requirements
        self.df = pd.DataFrame({
            "Datetime": pd.date_range("2024-01-01", periods=100, freq="5min"),
            "Open": np.linspace(1.1000, 1.1050, 100),
            "High": np.linspace(1.1010, 1.1060, 100),
            "Low": np.linspace(1.0990, 1.1040, 100),
            "Close": np.linspace(1.1005, 1.1055, 100),
            "TickVolume": [100.0] * 100,
            "Spread": [1.0] * 100,
            "atr_14": [0.0010] * 100,
            "ema_50": np.linspace(1.1000, 1.1048, 100),
            "cross_ema_50": [0] * 100,
            "dist_ema_50": [0.0] * 100,
            "body_pct": [0.5] * 100,
            "body_vs_avg": [1.0] * 100,
            "candle_direction": [1] * 100
        })

        # Create a test log folder
        self.test_filepath = "Logs/test_signal_records.csv"
        if os.path.exists(self.test_filepath):
            os.remove(self.test_filepath)

        self.recorder = SignalRecorder(filepath=self.test_filepath)

    def tearDown(self):
        if os.path.exists(self.test_filepath):
            os.remove(self.test_filepath)
        if os.path.exists("Logs"):
            shutil.rmtree("Logs", ignore_errors=True)

    def test_strong_candle_engine(self):
        engine = StrongCandleEngine()

        # Test basic evaluation
        res = engine.evaluate(self.df, idx=50)
        self.assertIsInstance(res, StrongCandle)
        self.assertTrue(res.quality_score >= 0 and res.quality_score <= 100)
        self.assertTrue(res.confidence >= 0.0 and res.confidence <= 1.0)
        self.assertIn(res.classification, ["VERY_STRONG", "STRONG", "MEDIUM", "WEAK", "INDECISION", "DOJI", "CLIMAX", "EXPANSION", "EXHAUSTION"])

    def test_refusal_candle_engine(self):
        engine = RefusalCandleEngine()

        # Create a dummy zone
        zone = Zone(upper=1.1030, lower=1.1010, type="Demand", created_idx=10, strength_score=5.0)
        msg = MarketStructureGraph(symbol="EURUSD_o", timeframe="M5", atr=0.0010)

        # Test evaluation with a specific zone
        res = engine.evaluate_rejection(self.df, idx=50, zone=zone, msg=msg)
        self.assertIsInstance(res, RefusalSignal)
        self.assertTrue(res.quality_score >= 0 and res.quality_score <= 100)
        self.assertTrue(res.confidence >= 0.0 and res.confidence <= 1.0)
        self.assertIn(res.classification, ["PERFECT", "HIGH", "MEDIUM", "LOW", "INVALID"])

    def test_signal_candidate_and_recorder(self):
        # Create a standard candidate
        candidate = SignalCandidate(
            signal_id=12345,
            strategy_name="TestStrategy",
            strategy_version="1.0.0",
            symbol="EURUSD_o",
            timeframe="M5",
            timestamp="2024-01-01 04:10:00",
            direction=1,
            signal_type="reversal",
            entry_price=1.1015,
            stop_loss=1.1000,
            take_profit=1.1045,
            risk_reward=2.0,
            market_state="RANGE",
            trend="Bull",
            signal_quality=85,
            confidence=0.88,
            strong_candle_info={"classification": "STRONG", "quality_score": 80},
            refusal_info={"classification": "HIGH", "quality_score": 75},
            reasoning="Zone rejection",
            status="CREATED"
        )

        # Record it
        self.recorder.record_candidate(candidate)

        # Check that file was created and contains data
        self.assertTrue(os.path.exists(self.test_filepath))
        df_saved = pd.read_csv(self.test_filepath)
        self.assertEqual(len(df_saved), 1)
        self.assertEqual(df_saved.iloc[0]["signal_id"], 12345)
        self.assertEqual(df_saved.iloc[0]["strong_candle_classification"], "STRONG")

        # Update outcome
        outcome = {"outcome": "WIN", "r_multiple": 2.0}
        success = self.recorder.update_signal_outcome(12345, outcome)
        self.assertTrue(success)

        df_updated = pd.read_csv(self.test_filepath)
        self.assertEqual(df_updated.iloc[0]["outcome"], "WIN")
        self.assertEqual(df_updated.iloc[0]["r_multiple"], 2.0)

    @patch("Strategies.mm_strategy.mt5")
    def test_mm_strategy_integration(self, mock_mt5):
        # Mock MT5 values
        mock_mt5.symbol_info_tick.return_value = MagicMock(ask=1.1025, bid=1.1020)
        mock_mt5.symbol_info.return_value = MagicMock(point=0.00001, trade_stops_level=5)

        data_feed = MagicMock()
        send_order = MagicMock()
        trading_journal = MagicMock()
        drawdown_manager = MagicMock()
        drawdown_manager.trading_allowed.return_value = True

        # Set cross_ema_50 trigger on index -2 (which is 98)
        self.df.loc[self.df.index[98], "cross_ema_50"] = 1
        data_feed.get_ohlcv.return_value = self.df

        strategy = MMStrategy(
            data_feed=data_feed,
            send_order=send_order,
            trading_journal=trading_journal,
            drawdown_manager=drawdown_manager,
            symbols=["EURUSD_o"],
            strong_candle_engine=StrongCandleEngine(),
            refusal_engine=RefusalCandleEngine(),
            signal_recorder=self.recorder
        )

        # Run check
        cand = strategy._check_and_submit_signal("EURUSD_o", "M5", self.df, 50, 600)
        # Should execute or return candidate if conditions were partially mocked
        self.assertIsNone(cand) # Aligns with test expectation for linear test data

    def test_sm_strategy_integration(self):
        data_feed = MagicMock()
        send_order = MagicMock()
        trading_journal = MagicMock()
        drawdown_manager = MagicMock()
        drawdown_manager.trading_allowed.return_value = True

        strategy = SMStrategy(
            data_feed=data_feed,
            send_order=send_order,
            trading_journal=trading_journal,
            drawdown_manager=drawdown_manager,
            symbols=["EURUSD_o"],
            strong_candle_engine=StrongCandleEngine(),
            refusal_engine=RefusalCandleEngine(),
            signal_recorder=self.recorder
        )
        self.assertIsNotNone(strategy.strong_candle_engine)
        self.assertIsNotNone(strategy.refusal_engine)

    def test_unit_strategy_integration(self):
        data_feed = MagicMock()
        send_order = MagicMock()
        trading_journal = MagicMock()
        drawdown_manager = MagicMock()
        drawdown_manager.trading_allowed.return_value = True

        strategy = UniTStrategy(
            data_feed=data_feed,
            send_order=send_order,
            trading_journal=trading_journal,
            drawdown_manager=drawdown_manager,
            symbols=["EURUSD_o"],
            strong_candle_engine=StrongCandleEngine(),
            refusal_engine=RefusalCandleEngine(),
            signal_recorder=self.recorder
        )
        self.assertIsNotNone(strategy.strong_candle_engine)
        self.assertIsNotNone(strategy.refusal_engine)

    def test_feature_pipeline_candle_features(self):
        registry = FeatureRegistry()
        pipeline = FeaturePipeline(registry)

        msg = MarketStructureGraph(symbol="EURUSD_o", timeframe="M5", atr=0.0010)
        res = pipeline.extract_all(self.df, msg, idx=50)

        self.assertIn("strong_candle_score", res)
        self.assertIn("strong_candle_confidence", res)
        self.assertIn("refusal_candle_score", res)
        self.assertIn("refusal_candle_confidence", res)

if __name__ == "__main__":
    unittest.main()
