import sys
from pathlib import Path

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Allow direct execution via ``python scripts/verify_market_structure_logic.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MarketStructure.market_structure import MarketStructureEngine
from MarketStructure.supply_demand import SupplyDemandEngine

def create_synthetic_data(n=200):
    np.random.seed(42)
    base_price = 1.1000
    prices = base_price + np.cumsum(np.random.randn(n) * 0.001)

    df = pd.DataFrame({
        'Datetime': [datetime(2024, 1, 1) + timedelta(minutes=5*i) for i in range(n)],
        'Open': prices + np.random.randn(n) * 0.0001,
        'High': prices + np.abs(np.random.randn(n)) * 0.0005,
        'Low': prices - np.abs(np.random.randn(n)) * 0.0005,
        'Close': prices,
        'TickVolume': np.random.randint(100, 1000, n)
    })
    return df

def test_market_structure():
    print("Testing MarketStructureEngine...")
    df = create_synthetic_data()
    engine = MarketStructureEngine(lookback=3)
    df_result = engine.process(df)

    # Check for expected columns
    expected_cols = ['trend', 'bos', 'choch', 'bars_since_bos', 'bars_since_choch']
    for col in expected_cols:
        assert col in df_result.columns, f"Missing column: {col}"

    summary = engine.get_summary(df_result)
    assert isinstance(summary, dict)
    assert 'trend' in summary
    assert 'swing_high' in summary

    print("MarketStructureEngine tests passed.")

def test_supply_demand():
    print("Testing SupplyDemandEngine...")
    df = create_synthetic_data()
    engine = SupplyDemandEngine(impulse_threshold=1.5)
    df_result = engine.process(df)

    expected_cols = [
        'nearest_supply_distance', 'nearest_demand_distance',
        'inside_supply', 'inside_demand',
        'supply_strength', 'demand_strength',
        'bars_since_supply', 'bars_since_demand'
    ]
    for col in expected_cols:
        assert col in df_result.columns, f"Missing column: {col}"

    print("SupplyDemandEngine tests passed.")

def test_lookahead_bias():
    print("Testing Look-ahead Bias...")
    df = create_synthetic_data(100)
    engine = MarketStructureEngine(lookback=3)

    # Full process
    df_full = engine.process(df)

    # Partial process (up to index 50)
    engine_partial = MarketStructureEngine(lookback=3)
    df_partial = engine_partial.process(df.iloc[:51])

    # Check that common rows are identical for deterministic outputs
    # Note: Market structure can change as new bars confirm previous swings,
    # but the output for index 40 should be the same whether we have 50 bars or 100 bars.
    # Swing confirmation at index i happens at index i+lookback.
    # So for lookback 3, index 40 is confirmed by index 43.
    # If we have 51 bars, everything up to 51-3=48 should be identical to the full run.

    test_idx = 40
    cols_to_check = ['trend', 'bars_since_bos', 'bars_since_choch']
    for col in cols_to_check:
        full_val = df_full.iloc[test_idx][col]
        part_val = df_partial.iloc[test_idx][col]
        assert full_val == part_val, f"Look-ahead bias detected in {col} at index {test_idx}! Full: {full_val}, Partial: {part_val}"

    print("Look-ahead bias tests passed.")

if __name__ == "__main__":
    try:
        test_market_structure()
        test_supply_demand()
        test_lookahead_bias()
        print("\nAll Market Structure logic verifications passed successfully!")
    except Exception as e:
        print(f"\nVerification failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
