import logging
import MetaTrader5 as mt5

# Constants for when MetaTrader5 is not installed (e.g. during local testing)
if not hasattr(mt5, "ORDER_FILLING_FOK"):
    ORDER_FILLING_FOK = 0
else:
    ORDER_FILLING_FOK = mt5.ORDER_FILLING_FOK

if not hasattr(mt5, "ORDER_TYPE_BUY"):
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
else:
    ORDER_TYPE_BUY = mt5.ORDER_TYPE_BUY
    ORDER_TYPE_SELL = mt5.ORDER_TYPE_SELL

if not hasattr(mt5, "TRADE_ACTION_DEAL"):
    TRADE_ACTION_DEAL = 1
else:
    TRADE_ACTION_DEAL = mt5.TRADE_ACTION_DEAL

if not hasattr(mt5, "TRADE_ACTION_SLTP"):
    TRADE_ACTION_SLTP = 6
else:
    TRADE_ACTION_SLTP = mt5.TRADE_ACTION_SLTP

if not hasattr(mt5, "ORDER_TIME_GTC"):
    ORDER_TIME_GTC = 0
else:
    ORDER_TIME_GTC = mt5.ORDER_TIME_GTC

logger = logging.getLogger("PositionManager")

class PositionManager:
    def __init__(
        self,
        magic_unity: int,
        magic_mm: int,
        deviation: int = 10,
        filling_mode: int = ORDER_FILLING_FOK,
    ):
        self.magic_unity = magic_unity
        self.magic_mm = magic_mm
        self.deviation = deviation
        self.filling_mode = filling_mode
        logger.info(f"PositionManager initialized (magic_unity={magic_unity}, magic_mm={magic_mm})")

    def open_position(
        self,
        symbol: str,
        direction: int,          # 1 = buy, -1 = sell
        lot_size: float,
        sl_price: float,
        tp_price: float,
        strategy: str,           # "unity" or "mm"
        comment: str = "",
    ) -> dict:
        """
        Executes a buy or sell order.
        """
        # 1. Validate inputs
        if direction not in [1, -1]:
            return self._result(False, None, symbol, direction, lot_size, None, sl_price, tp_price, strategy, comment, error_code=-1, retcode=None, msg="Invalid direction")
        if lot_size <= 0:
            return self._result(False, None, symbol, direction, lot_size, None, sl_price, tp_price, strategy, comment, error_code=-1, retcode=None, msg="Invalid lot size")
        if sl_price == 0 or tp_price == 0:
            return self._result(False, None, symbol, direction, lot_size, None, sl_price, tp_price, strategy, comment, error_code=-1, retcode=None, msg="SL and TP must be non-zero")

        # 2. Determine order type and price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            err = mt5.last_error()
            logger.error(f"Failed to get tick for {symbol}: {err}")
            return self._result(False, None, symbol, direction, lot_size, None, sl_price, tp_price, strategy, comment, error_code=err[0], retcode=None, msg="Failed to get tick info")

        if direction == 1:
            order_type = ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = ORDER_TYPE_SELL
            price = tick.bid

        # 3. Determine magic number
        magic = self.magic_unity if strategy == "unity" else self.magic_mm

        # 4. Construct MqlTradeRequest
        request = {
            "action": TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot_size),
            "type": order_type,
            "price": price,
            "sl": float(sl_price),
            "tp": float(tp_price),
            "deviation": self.deviation,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.filling_mode,
        }

        # 5. Send via OrderSend()
        result = mt5.order_send(request)
        
        if result is None:
            err = mt5.last_error()
            logger.error(f"order_send returned None for {symbol}. Error: {err}")
            return self._result(False, None, symbol, direction, lot_size, None, sl_price, tp_price, strategy, comment, error_code=err[0], retcode=None, msg="order_send returned None")

        success = result.retcode == mt5.TRADE_RETCODE_DONE
        ticket = result.order if success else None
        entry_price = result.price if success else None

        res_dict = self._result(
            success=success,
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            lot_size=lot_size,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            strategy=strategy,
            comment=comment,
            error_code=0 if success else mt5.last_error()[0],
            retcode=result.retcode
        )

        if success:
            logger.info(f"OPEN SUCCESS: {symbol} {strategy} {direction} {lot_size} SL:{sl_price} TP:{tp_price} Ticket:{ticket}")
        else:
            logger.error(f"OPEN FAILED: {symbol} {strategy} {direction} Lot:{lot_size} Retcode:{result.retcode} Comment:{result.comment}")
            res_dict["comment"] = result.comment

        return res_dict

    def close_position(
        self,
        ticket: int,
        volume: float = None,    # None = full close, float = partial close
    ) -> dict:
        """
        Closes an open position.
        """
        # 1. Query MT5 for position by ticket
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            logger.error(f"CLOSE FAILED: Position {ticket} not found.")
            return self._result(False, None, "UNKNOWN", 0, 0, None, 0, 0, "UNKNOWN", f"Position {ticket} not found", error_code=-1, retcode=None)

        pos = position[0]
        symbol = pos.symbol
        pos_volume = pos.volume
        pos_type = pos.type
        pos_magic = pos.magic
        strategy = "unity" if pos_magic == self.magic_unity else "mm" if pos_magic == self.magic_mm else "unknown"

        # 2. Determine volume to close
        close_vol = volume if volume is not None else pos_volume
        if close_vol > pos_volume:
            logger.error(f"CLOSE FAILED: Requested volume {close_vol} exceeds position volume {pos_volume}")
            return self._result(False, ticket, symbol, 1 if pos_type == ORDER_TYPE_BUY else -1, pos_volume, None, pos.sl, pos.tp, strategy, "Volume too large", error_code=-1, retcode=None)

        # 3. Determine price (opposite of entry)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            err = mt5.last_error()
            return self._result(False, ticket, symbol, 1 if pos_type == ORDER_TYPE_BUY else -1, close_vol, None, pos.sl, pos.tp, strategy, "Tick info failed", error_code=err[0], retcode=None)

        price = tick.bid if pos_type == ORDER_TYPE_BUY else tick.ask
        order_type = ORDER_TYPE_SELL if pos_type == ORDER_TYPE_BUY else ORDER_TYPE_BUY

        # 4. Send close order
        request = {
            "action": TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(close_vol),
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": self.deviation,
            "magic": pos_magic,
            "comment": "Close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.filling_mode,
        }

        result = mt5.order_send(request)
        if result is None:
            err = mt5.last_error()
            return self._result(False, ticket, symbol, 1 if pos_type == ORDER_TYPE_BUY else -1, close_vol, None, pos.sl, pos.tp, strategy, "order_send failed", error_code=err[0], retcode=None)

        success = result.retcode == mt5.TRADE_RETCODE_DONE
        
        res_dict = self._result(
            success=success,
            ticket=ticket,
            symbol=symbol,
            direction=1 if pos_type == ORDER_TYPE_BUY else -1,
            lot_size=close_vol,
            entry_price=result.price if success else None,
            sl_price=pos.sl,
            tp_price=pos.tp,
            strategy=strategy,
            comment=result.comment if not success else "Success",
            error_code=0 if success else mt5.last_error()[0],
            retcode=result.retcode
        )

        if success:
            logger.info(f"CLOSE SUCCESS: Ticket {ticket} Symbol {symbol} Vol {close_vol}")
        else:
            logger.error(f"CLOSE FAILED: Ticket {ticket} Retcode {result.retcode} Comment {result.comment}")

        return res_dict

    def modify_position(
        self,
        ticket: int,
        sl_price: float = None,
        tp_price: float = None,
    ) -> dict:
        """
        Modifies SL/TP of an open position.
        """
        # 1. Query MT5 for position by ticket
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            return self._result(False, None, "UNKNOWN", 0, 0, None, 0, 0, "UNKNOWN", "Position not found", error_code=-1, retcode=None)

        pos = position[0]
        symbol = pos.symbol
        pos_magic = pos.magic
        strategy = "unity" if pos_magic == self.magic_unity else "mm" if pos_magic == self.magic_mm else "unknown"

        # 2. Construct SLTP request
        new_sl = float(sl_price) if sl_price is not None else pos.sl
        new_tp = float(tp_price) if tp_price is not None else pos.tp

        request = {
            "action": TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": new_tp,
            "magic": pos_magic,
        }

        # 3. Send modification
        result = mt5.order_send(request)
        if result is None:
            err = mt5.last_error()
            return self._result(False, ticket, symbol, 1 if pos.type == ORDER_TYPE_BUY else -1, pos.volume, None, new_sl, new_tp, strategy, "order_send failed", error_code=err[0], retcode=None)

        success = result.retcode == mt5.TRADE_RETCODE_DONE

        res_dict = self._result(
            success=success,
            ticket=ticket,
            symbol=symbol,
            direction=1 if pos.type == ORDER_TYPE_BUY else -1,
            lot_size=pos.volume,
            entry_price=pos.price_open,
            sl_price=new_sl,
            tp_price=new_tp,
            strategy=strategy,
            comment=result.comment if not success else "Success",
            error_code=0 if success else mt5.last_error()[0],
            retcode=result.retcode
        )

        if success:
            logger.info(f"MODIFY SUCCESS: Ticket {ticket} SL:{new_sl} TP:{new_tp}")
        else:
            logger.error(f"MODIFY FAILED: Ticket {ticket} Retcode {result.retcode} Comment {result.comment}")

        return res_dict

    def _result(self, success, ticket, symbol, direction, lot_size, entry_price, sl_price, tp_price, strategy, comment, error_code=None, retcode=None, msg=None) -> dict:
        return {
            "success": success,
            "ticket": ticket,
            "symbol": symbol,
            "direction": direction,
            "lot_size": lot_size,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "strategy": strategy,
            "error_code": error_code,
            "retcode": retcode,
            "comment": comment if msg is None else f"{msg}: {comment}",
        }

if __name__ == "__main__":
    # Structural test only — no live MT5 required.
    # Verifies return schema shape on a mocked failed order.
    import unittest.mock as mock

    # Mocking mt5 module
    with mock.patch("MetaTrader5.order_send") as mock_send, \
         mock.patch("MetaTrader5.symbol_info_tick") as mock_tick, \
         mock.patch("MetaTrader5.last_error") as mock_last_err:
        
        mock_tick.return_value = mock.MagicMock(ask=1.1000, bid=1.0990)
        mock_last_err.return_value = (10004, "Requote")
        
        mock_result = mock.MagicMock()
        mock_result.retcode = 10004  # TRADE_RETCODE_REQUOTE
        mock_result.order = 0
        mock_result.comment = "Requote"
        mock_send.return_value = mock_result

        pm = PositionManager(magic_unity=100001, magic_mm=100002)
        result = pm.open_position(
            symbol="EURUSD_o",
            direction=1,
            lot_size=0.01,
            sl_price=1.0950,
            tp_price=1.1050,
            strategy="unity",
        )

        assert result["success"] == False
        assert result["error_code"] is not None
        assert "sl_price" in result
        assert "tp_price" in result
        assert "lot_size" in result
        print("PositionManager schema test passed.")
