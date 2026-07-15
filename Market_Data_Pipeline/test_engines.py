import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from Market_Data_Pipeline.structure_engine import MarketStructureEngine
from Market_Data_Pipeline.supply_demand_engine import SupplyDemandEngine
from Market_Data_Pipeline.replay_validator import StructureReplayValidator

class TestMarketDataPipelineEngines(unittest.TestCase):
    def setUp(self):
        # Using lookback=3
        self.ms_engine = MarketStructureEngine(lookback=3)
        self.sd_engine = SupplyDemandEngine(atr_period=14, impulse_threshold=2.0, use_fractal=False)

    def _make_base_df(self, n_bars: int = 100, base_price: float = 1.1000) -> pd.DataFrame:
        times = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15 * i) for i in range(n_bars)]
        df = pd.DataFrame({
            'Datetime': times,
            'Open': np.full(n_bars, base_price),
            'High': np.full(n_bars, base_price + 0.0005),
            'Low': np.full(n_bars, base_price - 0.0005),
            'Close': np.full(n_bars, base_price),
            'TickVolume': np.full(n_bars, 100.0),
            'Spread': np.full(n_bars, 1.0)
        })
        df['ema_50'] = df['Close']
        df['ema_600'] = df['Close']
        df['ema_800'] = df['Close']
        df['atr_14'] = 0.0010
        return df

    def test_swing_high_low_detection(self):
        # With lookback=3, we can confirm a pivot at idx 10 at index 13
        df = self._make_base_df(n_bars=30)
        # Create a confirmed Swing High at index 10 (lookback=3)
        df.loc[10, 'High'] = 1.1050
        df.loc[10, 'Close'] = 1.1040
        # Make sure surrounding bars are lower
        for idx in range(7, 14):
            if idx != 10:
                df.loc[idx, 'High'] = 1.1000

        # Create a confirmed Swing Low at index 20
        df.loc[20, 'Low'] = 1.0950
        df.loc[20, 'Close'] = 1.0960
        # Make sure surrounding bars are higher
        for idx in range(17, 24):
            if idx != 20:
                df.loc[idx, 'Low'] = 1.1000

        df_processed = self.ms_engine.process(df)

        # Verify swing high was detected
        sh_swings = [s for s in self.ms_engine.swings if s.level_type == 'SwingHigh']
        self.assertTrue(len(sh_swings) > 0)
        sh = max(sh_swings, key=lambda s: s.price)
        self.assertEqual(sh.index, 10)
        self.assertAlmostEqual(sh.price, 1.1050)

    def test_bos_and_choch_detection(self):
        # We need two pivot highs to trigger a breakout under MQL5 logic
        df = self._make_base_df(n_bars=50)
        # Swing High 1 at index 10
        df.loc[10, 'High'] = 1.1030
        for idx in range(7, 14):
            if idx != 10:
                df.loc[idx, 'High'] = 1.1000

        # Swing High 2 at index 30 (pivPrice > previous pivPrice)
        df.loc[30, 'High'] = 1.1050
        for idx in range(27, 34):
            if idx != 30:
                df.loc[idx, 'High'] = 1.1000

        df_processed = self.ms_engine.process(df)

        # Verify that either a BOS or CHOCH was registered
        total_breaks = len(self.ms_engine.bos_list) + len(self.ms_engine.choch_list)
        self.assertTrue(total_breaks > 0)

    def test_supply_demand_zone_mitigation_and_break(self):
        # Tests fallback impulse mode
        df = self._make_base_df(n_bars=100)

        # Base candle at index 24: Open/Close are base prices, Low is 1.0980, High/Upper is 1.1002
        df.loc[24, 'Open'] = 1.1000
        df.loc[24, 'Close'] = 1.1002
        df.loc[24, 'Low'] = 1.0980
        df.loc[24, 'High'] = 1.1005

        # Create a massive demand zone via a bullish impulsive move at index 25
        df.loc[25, 'Open'] = 1.1010
        df.loc[25, 'High'] = 1.1060
        df.loc[25, 'Low'] = 1.1008
        df.loc[25, 'Close'] = 1.1055

        # Make all subsequent bars from 26 to 44 stay above the zone upper limit of 1.1002
        for idx in range(26, 45):
            df.loc[idx, 'Open'] = 1.1020
            df.loc[idx, 'High'] = 1.1030
            df.loc[idx, 'Low'] = 1.1015
            df.loc[idx, 'Close'] = 1.1025

        # Touches demand zone at index 45 without breaking close limit (lower price is 1.0980)
        df.loc[45, 'Low'] = 1.0990
        df.loc[45, 'Close'] = 1.1020

        # Keep subsequent bars before 60 above the zone lower limit of 1.0980
        for idx in range(46, 60):
            df.loc[idx, 'Open'] = 1.1020
            df.loc[idx, 'High'] = 1.1030
            df.loc[idx, 'Low'] = 1.1015
            df.loc[idx, 'Close'] = 1.1025

        # Breaks demand zone at index 60 (Close below lower limit 1.0980)
        df.loc[60, 'Close'] = 1.0970
        df.loc[60, 'Low'] = 1.0965

        # Process supply demand
        df_processed = self.sd_engine.process(df)

        demand_zones = [z for z in self.sd_engine.zones if z.type == 'Demand']
        self.assertEqual(len(demand_zones), 1)
        z = demand_zones[0]

        # Verify mitigation
        self.assertTrue(z.mitigated)
        self.assertEqual(z.mitigated_idx, 45)
        self.assertEqual(z.touch_count, 2)

        # Verify breakage
        self.assertTrue(z.broken)
        self.assertEqual(z.broken_idx, 60)
        self.assertFalse(z.active)

    def test_edge_cases_weekend_gaps_and_spikes(self):
        df = self._make_base_df(n_bars=40)

        # Weekend Gap: index 20 has large jump in time
        df.loc[20, 'Datetime'] = df.loc[19, 'Datetime'] + timedelta(days=2, hours=12)

        # Giant Flash Crash Spike: low goes extremely deep but closes near open
        df.loc[15, 'Open'] = 1.1000
        df.loc[15, 'Low'] = 1.0500
        df.loc[15, 'High'] = 1.1010
        df.loc[15, 'Close'] = 1.0995

        df_ms = self.ms_engine.process(df)
        df_sd = self.sd_engine.process(df_ms)

        # Check stability
        self.assertIsNotNone(df_sd)
        # Verify core pricing columns do not have NaNs
        self.assertEqual(df_sd[['Open', 'High', 'Low', 'Close']].isnull().sum().sum(), 0)

    def test_nested_zones(self):
        df = self._make_base_df(n_bars=80)

        # Create larger demand zone at index 25
        df.loc[24, 'Low'] = 1.0900
        df.loc[25, 'Open'] = 1.1000
        df.loc[25, 'Close'] = 1.1060

        # Create nested smaller demand zone inside it at index 45
        df.loc[44, 'Low'] = 1.0950
        df.loc[45, 'Open'] = 1.1000
        df.loc[45, 'Close'] = 1.1050

        df_processed = self.sd_engine.process(df)
        demands = [z for z in self.sd_engine.zones if z.type == 'Demand']

        self.assertTrue(len(demands) >= 2)
        # Find the smaller one
        nested = [z for z in demands if z.nested_inside_idx is not None]
        self.assertTrue(len(nested) > 0)
        self.assertEqual(nested[0].nested_inside_idx, 24)

    def test_fractal_supply_demand_zone_mitigation_and_break(self):
        # Dedicated test for MQL5 Order Block creation on structure breakout
        fractal_engine = SupplyDemandEngine(atr_period=14, use_fractal=True)
        df = self._make_base_df(n_bars=80)

        # We need two confirmed pivot highs to trigger a breakout and Bullish Order Block creation
        # Swing High 1 at index 15
        df.loc[15, 'High'] = 1.1030
        for idx in range(5, 26):
            if idx != 15:
                df.loc[idx, 'High'] = 1.1000

        # Swing High 2 at index 40 (pivPrice > previous pivPrice)
        df.loc[40, 'High'] = 1.1060
        for idx in range(30, 51):
            if idx != 40:
                df.loc[idx, 'High'] = 1.1000
            # Opp-color candle candidate going backwards from breakout pivot index 40: index 39 bearish
            if idx == 39:
                df.loc[idx, 'Open'] = 1.1010
                df.loc[idx, 'Close'] = 1.0990

        # Run processing
        df_processed = fractal_engine.process(df)

        # Verify that a Bullish Order Block (Demand Zone) was created!
        demand_zones = [z for z in fractal_engine.zones if z.type == 'Demand']
        self.assertTrue(len(demand_zones) >= 1)
        z = demand_zones[0]
        self.assertEqual(z.created_idx, 39)  # Backwards opposite-color candle index from breakout pivot high 40

if __name__ == '__main__':
    unittest.main()
