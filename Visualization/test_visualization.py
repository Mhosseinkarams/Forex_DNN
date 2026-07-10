import unittest
import os
from datetime import datetime, timezone
from Market_Data_Pipeline.structure_graph import MarketStructureGraph, StructureLevel, Zone, BOS, CHOCH
from Market_Data_Pipeline.state_engine import StateContext
from Visualization.debug_config import DebugConfig
from Visualization.chart_annotator import ChartAnnotationEngine

class TestVisualizationEngineV2(unittest.TestCase):
    def setUp(self):
        self.config = DebugConfig()
        self.annotator = ChartAnnotationEngine(config=self.config)
        self.symbol = "EURUSD"

        # Build simple mock graph
        self.graph = MarketStructureGraph(
            symbol=self.symbol,
            timeframe="M5",
            timestamp=datetime.now(timezone.utc),
            swing_highs=[StructureLevel(price=1.1250, index=10, timestamp=datetime.now(timezone.utc), level_type="SwingHigh")],
            swing_lows=[StructureLevel(price=1.1200, index=5, timestamp=datetime.now(timezone.utc), level_type="SwingLow")],
            bos=[BOS(index=15, direction=1, broken_level=1.1260, timestamp=datetime.now(timezone.utc))],
            choch=[CHOCH(index=18, previous_trend=-1, new_trend=1, timestamp=datetime.now(timezone.utc), price=1.1270)],
            supply_zones=[Zone(upper=1.1300, lower=1.1280, type="Supply", created_time=datetime.now(timezone.utc), strength_score=3.0)],
            demand_zones=[Zone(upper=1.1180, lower=1.1160, type="Demand", created_time=datetime.now(timezone.utc), strength_score=2.0)],
            atr=0.0010
        )

        self.state_ctx = StateContext(
            regime="Trending",
            trend_direction="Bull",
            volatility_regime="Normal",
            confidence_score=0.95,
            ema_slope=0.0,
            ema_distance_atr=1.5,
            atr=0.0010,
            timestamp=datetime.now(timezone.utc)
        )

    def test_render_creates_csv_files(self):
        # We enforce fallback folder output/Files/ for offline testing
        target_dir = os.path.join("output", "Files")

        trade_plan = {"entry_price": 1.1220, "sl_price": 1.1200, "tp_price": 1.1260}
        decision = {"direction": 1, "accepted": True, "reason": "Test setup"}
        ml_output = {"quality_score": 0.85, "break_prob": 0.1}

        self.annotator.render(
            symbol=self.symbol,
            structure_graph=self.graph,
            state_context=self.state_ctx,
            trade_plan=trade_plan,
            decision=decision,
            ml_output=ml_output
        )

        expected_files = [
            f"{self.symbol}_structure.csv",
            f"{self.symbol}_levels.csv",
            f"{self.symbol}_zones.csv",
            f"{self.symbol}_signals.csv",
            f"{self.symbol}_state.csv",
            f"{self.symbol}_ml.csv"
        ]

        for ef in expected_files:
            fp = os.path.join(target_dir, ef)
            self.assertTrue(os.path.exists(fp), f"Expected file {fp} does not exist!")

            # Clean up
            try:
                os.remove(fp)
            except Exception:
                pass

if __name__ == "__main__":
    unittest.main()
