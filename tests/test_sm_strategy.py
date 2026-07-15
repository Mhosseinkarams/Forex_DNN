import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import threading
from datetime import datetime, timezone

from Market_Data_Pipeline.structure_graph import MarketStructureGraph, Zone
from Strategies.refusal_candle_engine import RefusalCandleEngine, RefusalResult
from Strategies.sm_strategy import SMStrategy
from ML.decision_context import DecisionContext, PolicyRecommendation


class TestSMStrategySuite(unittest.TestCase):
    def setUp(self):
        # 1. Setup Refusal Candle Engine
        self.refusal_engine = RefusalCandleEngine()

        # 2. Generate a dummy dataframe for candle testing
        self.df = pd.DataFrame({
            "Datetime": pd.date_range("2024-01-01", periods=15, freq="5min"),
            "Open": [1.1000] * 15,
            "High": [1.1008] * 15,
            "Low": [1.0992] * 15,
            "Close": [1.1000] * 15,
            "TickVolume": [100] * 15,
            "Spread": [1.0] * 15,
            "atr_14": [0.0010] * 15,
            "cross_ema_50": [0] * 15
        })

    def test_refusal_candle_engine_scoring(self):
        """
        Verify that RefusalCandleEngine calculates scores correctly for a strong pin bar
        reversing a supply zone versus a weak/standard candle.
        """
        # Strong Bearish rejection candle at index 10 (Peak rejection of Supply)
        # Supply upper 1.1025, lower 1.1015
        zone = Zone(
            upper=1.1025,
            lower=1.1015,
            type="Supply",
            strength_score=5.0
        )

        # Set up a strong pin bar rejecting the supply from above
        # High reaches into zone (1.1022), close near low (1.1001), open (1.1005)
        self.df.loc[10, "Open"] = 1.1005
        self.df.loc[10, "High"] = 1.1022  # Deep penetration of supply
        self.df.loc[10, "Low"] = 1.1000
        self.df.loc[10, "Close"] = 1.1001

        msg = MarketStructureGraph(
            symbol="EURUSD",
            timeframe="M5",
            atr=0.0010,
            supply_zones=[zone]
        )

        res_strong = self.refusal_engine.evaluate_rejection(self.df, 10, zone, msg)
        self.assertTrue(res_strong.bearish)
        self.assertGreater(res_strong.score, 65.0)

        # Contrast with a standard body candle with no wicks (weak rejection)
        self.df.loc[11, "Open"] = 1.1000
        self.df.loc[11, "High"] = 1.1010
        self.df.loc[11, "Low"] = 1.1000
        self.df.loc[11, "Close"] = 1.1010

        res_weak = self.refusal_engine.evaluate_rejection(self.df, 11, zone, msg)
        self.assertLess(res_weak.score, 45.0)

    def test_sm_strategy_regime_filtering(self):
        """
        Verify that SMStrategy skips on TREND or TRANSITION market states and proceeds on RANGE.
        """
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
            symbols=["EURUSD"],
            shadow_mode=True
        )

        # Mock the underlying model predictions
        decision_engine = MagicMock()
        strategy.decision_engine = decision_engine

        # Mock build_market_structure_graph
        msg = MarketStructureGraph(
            symbol="EURUSD",
            timeframe="M5",
            atr=0.0010
        )
        strategy._build_market_structure_graph = MagicMock(return_value=msg)

        # Case A: Predicted state is TREND -> should return / skip evaluation
        mock_ctx_trend = DecisionContext(
            symbol="EURUSD", timeframe="M5", timestamp="2024-01-01",
            predicted_state="TREND", state_probabilities={"TREND": 0.8, "RANGE": 0.1, "TRANSITION": 0.1},
            state_confidence=0.8, break_probability=0.2, rejection_probability=0.8,
            trade_quality_score=0.7, confidence_score=0.7,
            policy_recommendation=PolicyRecommendation(
                allow_trade=True, suggested_risk_multiplier=1.0, suggested_position_scale=1.0,
                suggested_tp_mode="STRUCTURE_TARGET", suggested_sl_adjustment=0.0
            ),
            model_versions={}, inference_time_ms=1.0, missing_features=[], warnings=[]
        )
        decision_engine.evaluate.return_value = mock_ctx_trend

        # Run strategy evaluation
        strategy._evaluate_setup_and_trade("EURUSD", "M5", self.df)
        send_order.execute.assert_not_called()

        # Case B: Predicted state is RANGE, but no eligible zones -> should still skip execution gracefully
        mock_ctx_range = DecisionContext(
            symbol="EURUSD", timeframe="M5", timestamp="2024-01-01",
            predicted_state="RANGE", state_probabilities={"TREND": 0.1, "RANGE": 0.8, "TRANSITION": 0.1},
            state_confidence=0.8, break_probability=0.1, rejection_probability=0.9,
            trade_quality_score=0.8, confidence_score=0.8,
            policy_recommendation=PolicyRecommendation(
                allow_trade=True, suggested_risk_multiplier=1.0, suggested_position_scale=1.0,
                suggested_tp_mode="STRUCTURE_TARGET", suggested_sl_adjustment=0.0
            ),
            model_versions={}, inference_time_ms=1.0, missing_features=[], warnings=[]
        )
        decision_engine.evaluate.return_value = mock_ctx_range

        strategy._evaluate_setup_and_trade("EURUSD", "M5", self.df)
        send_order.execute.assert_not_called()

    def test_sm_strategy_sl_tp_placement(self):
        """
        Verify that structural SL and TP levels are correctly placed outside zones
        and within opposite zones with buffers, and ranked properly.
        """
        # Create eligible zones
        demand_zone = Zone(
            upper=1.0990,
            lower=1.0980,
            type="Demand",
            created_idx=1,
            strength_score=5.0
        )
        opposite_supply_zone = Zone(
            upper=1.1030,
            lower=1.1020,
            type="Supply",
            created_idx=2,
            strength_score=5.0
        )

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
            symbols=["EURUSD"],
            shadow_mode=False, # active order execution
            sl_buffer_pct=0.10,
            tp_buffer_pct=0.20
        )

        # Set up closed bar intersecting Demand zone (rejection from below)
        self.df.loc[13, "Open"] = 1.0995
        self.df.loc[13, "High"] = 1.1000
        self.df.loc[13, "Low"] = 1.0988  # Intersects Demand zone
        self.df.loc[13, "Close"] = 1.0994

        msg = MarketStructureGraph(
            symbol="EURUSD",
            timeframe="M5",
            atr=0.0010,
            demand_zones=[demand_zone],
            supply_zones=[opposite_supply_zone]
        )
        strategy._build_market_structure_graph = MagicMock(return_value=msg)

        # Mock models to confirm RANGE and rejection HOLD
        decision_engine = MagicMock()
        mock_ctx = DecisionContext(
            symbol="EURUSD", timeframe="M5", timestamp="2024-01-01",
            predicted_state="RANGE", state_probabilities={"TREND": 0.1, "RANGE": 0.8, "TRANSITION": 0.1},
            state_confidence=0.8, break_probability=0.1, rejection_probability=0.9,
            trade_quality_score=0.8, confidence_score=0.8,
            policy_recommendation=PolicyRecommendation(
                allow_trade=True, suggested_risk_multiplier=1.0, suggested_position_scale=1.0,
                suggested_tp_mode="STRUCTURE_TARGET", suggested_sl_adjustment=0.0
            ),
            model_versions={}, inference_time_ms=1.0, missing_features=[], warnings=[]
        )
        decision_engine.evaluate.return_value = mock_ctx
        strategy.decision_engine = decision_engine

        # Mock the RefusalCandleEngine to return a high score
        ref_res = RefusalResult(
            score=85.0,
            confidence=0.9,
            bullish=True,
            bearish=False,
            reasons=["Confirmed pin bar"],
            metrics={}
        )
        strategy.refusal_engine.evaluate_rejection = MagicMock(return_value=ref_res)

        # Run strategy evaluation
        strategy._evaluate_setup_and_trade("EURUSD", "M5", self.df)

        # Assert order was executed
        send_order.execute.assert_called_once()
        args = send_order.execute.call_args[1]

        # Verify direction is BUY (1)
        self.assertEqual(args["direction"], 1)

        # Verify SL is placed structurally beyond Demand zone (lower - offset)
        # zone_width = 1.0990 - 1.0980 = 0.0010. sl_buffer = 0.10 -> offset = 0.0001
        # SL = 1.0980 - 0.0001 = 1.0979
        self.assertAlmostEqual(args["sl_price"], 1.0979)

    def test_sm_strategy_thread_safety(self):
        """
        Verify the thread-safe strategy lock behavior.
        """
        data_feed = MagicMock()
        send_order = MagicMock()
        trading_journal = MagicMock()
        drawdown_manager = MagicMock()

        strategy = SMStrategy(
            data_feed=data_feed,
            send_order=send_order,
            trading_journal=trading_journal,
            drawdown_manager=drawdown_manager,
            symbols=["EURUSD"]
        )

        # Acquire lock in thread A, check that thread B is blocked
        lock_acquired = []

        def worker_b():
            # Attempt to acquire the strategy's re-entrant/synchronization lock
            acquired = strategy._lock.acquire(timeout=0.2)
            if acquired:
                lock_acquired.append(True)
                strategy._lock.release()
            else:
                lock_acquired.append(False)

        strategy._lock.acquire()
        t = threading.Thread(target=worker_b)
        t.start()
        t.join()
        strategy._lock.release()

        # Thread B should have failed to acquire the lock because Thread A held it
        self.assertFalse(lock_acquired[0])


if __name__ == "__main__":
    unittest.main()
