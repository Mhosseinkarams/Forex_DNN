import logging
import math
import MetaTrader5 as mt5

logger = logging.getLogger("PositionSizer")

class PositionSizer:
    """
    Module 6 — Risk Sizing
    Computes lot size from a risk percentage and SL distance.
    """

    def calculate_lot_size(
        self,
        symbol: str,
        entry_price: float,
        sl_price: float,
        risk_pct: float,
        account_balance: float,
    ) -> dict:
        """
        Calculates the lot size based on risk parameters and symbol specifications.
        """
        # 1. Validate inputs
        if risk_pct <= 0 or account_balance <= 0:
            logger.error(f"Invalid input: risk_pct={risk_pct}, account_balance={account_balance}")
            return self._result(False, 0.0, 0.0, 0.0, False, "invalid_input")

        sl_distance = abs(entry_price - sl_price)
        if sl_distance == 0:
            logger.error(f"Invalid SL distance: entry={entry_price}, sl={sl_price}")
            return self._result(False, 0.0, 0.0, 0.0, False, "invalid_sl_distance")

        # 2. Fetch symbol information
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Symbol info unavailable for {symbol}")
            return self._result(False, 0.0, 0.0, 0.0, False, "symbol_info_unavailable")

        # 3. Lot size calculation
        risk_dollars = account_balance * risk_pct
        contract_size = info.trade_contract_size

        raw_lot = risk_dollars / (sl_distance * contract_size)

        # 4. Rounding (always round down)
        volume_step = info.volume_step
        volume_min = info.volume_min
        volume_max = info.volume_max

        # Round down to the nearest multiple of volume_step
        # math.floor(raw_lot / volume_step) * volume_step
        lot_size = math.floor(round(raw_lot / volume_step, 10)) * volume_step

        # Correct for floating-point drift by rounding to the same decimal precision as volume_step
        step_str = f"{volume_step:.8f}".rstrip('0').rstrip('.')
        precision = len(step_str.split('.')[1]) if '.' in step_str else 0
        lot_size = round(lot_size, precision)

        # 5. Constraints handling
        capped_at_max = False
        if lot_size < volume_min:
            logger.error(f"Lot size {lot_size} below minimum {volume_min} for {symbol}")
            return self._result(False, 0.0, 0.0, 0.0, False, "lot_size_below_minimum")

        if lot_size > volume_max:
            logger.warning(f"Lot size {lot_size} capped at maximum {volume_max} for {symbol}")
            lot_size = volume_max
            capped_at_max = True

        # Final metrics
        actual_risk_dollars = lot_size * sl_distance * contract_size
        actual_risk_pct = actual_risk_dollars / account_balance

        logger.info(f"SUCCESS: {symbol} lot_size={lot_size} risk=${actual_risk_dollars:.2f} ({actual_risk_pct*100:.2f}%)")

        return self._result(
            success=True,
            lot_size=lot_size,
            risk_dollars=actual_risk_dollars,
            risk_pct_actual=actual_risk_pct,
            capped_at_max=capped_at_max,
            error=None
        )

    def _result(self, success: bool, lot_size: float, risk_dollars: float, risk_pct_actual: float, capped_at_max: bool, error: str | None) -> dict:
        return {
            "success": success,
            "lot_size": lot_size,
            "risk_dollars": risk_dollars,
            "risk_pct_actual": risk_pct_actual,
            "capped_at_max": capped_at_max,
            "error": error
        }

if __name__ == "__main__":
    import unittest.mock as mock
    import sys

    # Structural test
    logging.basicConfig(level=logging.INFO)

    sizer = PositionSizer()

    # Case 1: Success
    with mock.patch("MetaTrader5.symbol_info") as mock_info:
        mock_info.return_value = mock.MagicMock(
            trade_contract_size=100000,
            volume_step=0.01,
            volume_min=0.01,
            volume_max=100.0
        )
        # 1% of 10,000 = $100
        # SL distance = 1.1000 - 1.0950 = 0.0050
        # lot = 100 / (0.0050 * 100000) = 100 / 500 = 0.2
        res = sizer.calculate_lot_size("EURUSD", 1.1000, 1.0950, 0.01, 10000.0)
        assert res["success"] == True
        assert res["lot_size"] == 0.2
        assert abs(res["risk_dollars"] - 100.0) < 1e-9
        assert res["error"] is None

    # Case 2: Rounding down
    with mock.patch("MetaTrader5.symbol_info") as mock_info:
        mock_info.return_value = mock.MagicMock(
            trade_contract_size=100000,
            volume_step=0.01,
            volume_min=0.01,
            volume_max=100.0
        )
        # $150 risk, 0.0050 SL -> 0.3 lot exactly
        # Let's try $151 risk -> 0.302 lot -> round down to 0.30
        res = sizer.calculate_lot_size("EURUSD", 1.1000, 1.0950, 0.0151, 10000.0)
        assert res["success"] == True
        assert res["lot_size"] == 0.3
        assert abs(res["risk_dollars"] - 150.0) < 1e-9
        assert res["risk_pct_actual"] < 0.0151

    # Case 3: Below minimum
    with mock.patch("MetaTrader5.symbol_info") as mock_info:
        mock_info.return_value = mock.MagicMock(
            trade_contract_size=100000,
            volume_step=0.01,
            volume_min=0.1,
            volume_max=100.0
        )
        # $10 risk, 0.0050 SL -> 0.02 lot. Min is 0.1.
        res = sizer.calculate_lot_size("EURUSD", 1.1000, 1.0950, 0.001, 10000.0)
        assert res["success"] == False
        assert res["error"] == "lot_size_below_minimum"
        assert res["lot_size"] == 0.0

    # Case 4: Capped at maximum
    with mock.patch("MetaTrader5.symbol_info") as mock_info:
        mock_info.return_value = mock.MagicMock(
            trade_contract_size=100000,
            volume_step=0.01,
            volume_min=0.01,
            volume_max=1.0
        )
        # $1000 risk, 0.0050 SL -> 2.0 lots. Max is 1.0.
        res = sizer.calculate_lot_size("EURUSD", 1.1000, 1.0950, 0.1, 10000.0)
        assert res["success"] == True
        assert res["capped_at_max"] == True
        assert res["lot_size"] == 1.0
        assert abs(res["risk_dollars"] - 500.0) < 1e-9

    # Case 5: Symbol info unavailable
    with mock.patch("MetaTrader5.symbol_info") as mock_info:
        mock_info.return_value = None
        res = sizer.calculate_lot_size("UNKNOWN", 1.1000, 1.0950, 0.01, 10000.0)
        assert res["success"] == False
        assert res["error"] == "symbol_info_unavailable"

    # Case 6: Invalid SL distance
    with mock.patch("MetaTrader5.symbol_info") as mock_info:
        res = sizer.calculate_lot_size("EURUSD", 1.1000, 1.1000, 0.01, 10000.0)
        assert res["success"] == False
        assert res["error"] == "invalid_sl_distance"

    print("PositionSizer structural tests passed.")
