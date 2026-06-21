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
        if risk_pct <= 0 or account_balance <= 0:
            logger.error(f"Invalid input: risk_pct={risk_pct}, account_balance={account_balance}")
            return self._result(False, 0.0, 0.0, 0.0, False, "invalid_input")

        sl_distance = abs(entry_price - sl_price)
        if sl_distance == 0:
            logger.error(f"Invalid SL distance: entry={entry_price}, sl={sl_price}")
            return self._result(False, 0.0, 0.0, 0.0, False, "invalid_sl_distance")

        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Symbol info unavailable for {symbol}")
            return self._result(False, 0.0, 0.0, 0.0, False, "symbol_info_unavailable")

        risk_dollars = account_balance * risk_pct
        contract_size = info.trade_contract_size

        raw_lot = risk_dollars / (sl_distance * contract_size)

        volume_step = info.volume_step
        volume_min = info.volume_min
        volume_max = info.volume_max

        lot_size = math.floor(round(raw_lot / volume_step, 10)) * volume_step

        step_str = f"{volume_step:.8f}".rstrip('0').rstrip('.')
        precision = len(step_str.split('.')[1]) if '.' in step_str else 0
        lot_size = round(lot_size, precision)

        capped_at_max = False
        if lot_size < volume_min:
            logger.error(f"Lot size {lot_size} below minimum {volume_min} for {symbol}")
            return self._result(False, 0.0, 0.0, 0.0, False, "lot_size_below_minimum")

        if lot_size > volume_max:
            logger.warning(f"Lot size {lot_size} capped at maximum {volume_max} for {symbol}")
            lot_size = volume_max
            capped_at_max = True

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

    def _result(self, success, lot_size, risk_dollars, risk_pct_actual, capped_at_max, error):
        return {
            "success": success,
            "lot_size": lot_size,
            "risk_dollars": risk_dollars,
            "risk_pct_actual": risk_pct_actual,
            "capped_at_max": capped_at_max,
            "error": error
        }
