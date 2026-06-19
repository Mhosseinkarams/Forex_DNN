import sys
import unittest.mock as mock

# 1. Create a comprehensive MT5 mock
mt5_mock = mock.MagicMock()
mt5_mock.TIMEFRAME_M1 = 1
mt5_mock.ORDER_TYPE_BUY = 0
mt5_mock.ORDER_TYPE_SELL = 1
mt5_mock.TRADE_RETCODE_DONE = 10009
sys.modules["MetaTrader5"] = mt5_mock

# 2. Import the classes to test
from PositionManager.drawdown import DrawdownManager
from PositionManager.position_tracker import PositionTracker
from PositionManager.position_manager import PositionManager
from Collecting_Data.indicators import IndicatorEngine

def test_drawdown():
    print("Running DrawdownManager tests...")
    import time
    import os
    import logging

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    mock_tracker = mock.MagicMock()
    mock_tracker.get_open_positions.return_value = []
    mock_tracker.get_open_risk.return_value = 0.0

    with mock.patch("MetaTrader5.account_info") as mock_acc, \
         mock.patch("MetaTrader5.symbol_info_tick") as mock_tick, \
         mock.patch("MetaTrader5.symbol_select") as mock_select:

        mock_select.return_value = True
        mock_acc.return_value = mock.MagicMock(balance=10000.0)
        mock_tick.return_value = mock.MagicMock(time=time.mktime(time.strptime("2024-05-20", "%Y-%m-%d")))

        dm = DrawdownManager(
            initial_balance=10000.0,
            position_tracker=mock_tracker,
            daily_limit_pct=0.03,
            total_limit_pct=0.10,
            state_file="test_drawdown_state.json"
        )

        assert dm.trading_allowed() == True
        assert abs(dm.max_risk_pct() - 0.03) < 1e-6

        # Position with $200 risk
        mock_tracker.get_open_risk.return_value = 200.0
        dm.check()
        assert abs(dm.max_risk_pct() - 0.01) < 1e-6

        # Loss and breach
        mock_acc.return_value = mock.MagicMock(balance=9850.0)
        dm.check()
        assert dm.trading_allowed() == False

        if os.path.exists("test_drawdown_state.json"):
            os.remove("test_drawdown_state.json")
    print("DrawdownManager tests passed.")

def test_indicators():
    print("Running IndicatorEngine tests...")
    import pandas as pd
    import numpy as np
    n = 700
    df_test = pd.DataFrame({
        "Datetime":   pd.date_range("2024-01-01", periods=n, freq="5min"),
        "Open":       1.1 + np.random.randn(n) * 0.0001,
        "High":       1.1 + 0.0003,
        "Low":        1.1 - 0.0003,
        "Close":      1.1,
        "TickVolume": 500,
        "Spread":     0,
    })
    engine = IndicatorEngine()
    result = engine.calculate(df_test)
    assert "ema_600" in result.columns
    print("IndicatorEngine tests passed.")

if __name__ == "__main__":
    test_drawdown()
    test_indicators()
    print("All tests passed successfully.")
