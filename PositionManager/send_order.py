import os
import json
import logging
import sys

# Optional MT5 import for environments where it's not installed (e.g. Linux CI)
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

logger = logging.getLogger("SendOrder")

class SendOrder:
    def __init__(
        self,
        position_manager,    # PositionManager instance
        position_tracker,    # PositionTracker instance
        drawdown_manager,    # DrawdownManager instance
        position_sizer,      # PositionSizer instance
        exit_manager,        # ExitManager instance
        trading_journal,     # TradingJournal instance
        state_file: str = "send_order_state.json"
    ):
        self.pm = position_manager
        self.pt = position_tracker
        self.dm = drawdown_manager
        self.ps = position_sizer
        self.em = exit_manager
        self.tj = trading_journal
        self.state_file = state_file

        self.ticket_categories = {}  # ticket (int) -> category (str)
        self._load_state()

    def _load_state(self):
        """Loads persisted category metadata from JSON."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    # JSON keys are strings, convert back to int for tickets
                    self.ticket_categories = {int(k): v for k, v in data.items()}
                logger.info(f"Loaded {len(self.ticket_categories)} tickets from state file.")
            except Exception as e:
                logger.error(f"Failed to load state file {self.state_file}: {e}")

    def _save_state(self):
        """Saves category metadata to JSON atomically."""
        try:
            temp_file = self.state_file + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(self.ticket_categories, f, indent=4)
            os.replace(temp_file, self.state_file)
        except Exception as e:
            logger.error(f"Failed to save state to {self.state_file}: {e}")

    def execute(
        self,
        symbol: str,
        direction: int,          # 1=buy, -1=sell
        entry_price: float,      # 0.0 or None = market order; specific price = pending (future)
        sl_price: float,
        tp_level: int,           # 1, 2, 3, or 4 — maps to 1R, 2R, 3R, 4R
        stage: str,              # "single" or "multi"
        strategy: str,           # "unity" or "mm"
        signal_category: str,    # "standard", "high_risk", or "reversal"
        signal_id: str,          # from TradingJournal.log_signal(), already logged upstream
        comment: str = "",
    ) -> dict:
        # 1. Validation
        if entry_price is not None and entry_price != 0.0:
            logger.warning(f"Pending orders not yet implemented. Requested entry: {entry_price}")
            return self._failure("invalid_input", "Pending orders not yet implemented", symbol, direction, signal_category)

        # 2. Fetch Live Price for Market Order
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                err = mt5.last_error()
                return self._failure("invalid_input", f"Failed to fetch tick for {symbol}: {err}", symbol, direction, signal_category)

            market_price = tick.ask if direction == 1 else tick.bid
            logger.debug(f"Fetched market price for {symbol}: {market_price}")
        except Exception as e:
            logger.error(f"Error fetching tick for {symbol}: {e}")
            return self._failure("invalid_input", f"Error fetching tick: {e}", symbol, direction, signal_category)

        # 3. Drawdown and Risk Check
        if not self.dm.trading_allowed():
            logger.warning(f"Trade blocked by DrawdownManager for {symbol}")
            return self._failure("drawdown_blocked", "Drawdown limit reached", symbol, direction, signal_category)

        risk_pct = self.dm.max_risk_pct()
        DEFAULT_RISK = {"standard": 0.01, "high_risk": 0.005, "reversal": 0.003}
        risk_pct = min(risk_pct, DEFAULT_RISK.get(signal_category, 0.01))

        # Account Balance
        acc = mt5.account_info()
        if acc is None:
            return self._failure("invalid_input", "Failed to fetch account balance", symbol, direction, signal_category)
        balance = acc.balance

        # 4. Position Conflict Checks
        open_positions = self.pt.get_open_positions()

        # Clean up stale tickets from internal category state
        live_tickets = {p["ticket"] for p in open_positions}
        stale_tickets = [t for t in self.ticket_categories if t not in live_tickets]
        if stale_tickets:
            for t in stale_tickets:
                del self.ticket_categories[t]
            self._save_state()

        symbol_positions = [p for p in open_positions if p["symbol"] == symbol]

        # Calculate R and TP price based on market_price
        R = abs(market_price - sl_price)
        tp_price = market_price + (1 if direction == 1 else -1) * tp_level * R

        for pos in symbol_positions:
            existing_ticket = pos["ticket"]
            existing_category = self.ticket_categories.get(existing_ticket, "unknown")
            existing_direction = pos["direction"]
            existing_sl = pos["sl_price"]
            existing_tp = pos["tp_price"]

            # Rule 1: Same category + same direction -> block
            if existing_category == signal_category and existing_direction == direction:
                logger.warning(f"Conflict Block (Rule 1): {symbol} already has {signal_category} in direction {direction}")
                return self._failure("conflict_blocked", "Rule 1 violation", symbol, direction, signal_category, sl=sl_price, tp=tp_price)

            # Rule 2: Different category + same direction -> SL check
            if existing_direction == direction:
                if direction == 1: # Buy
                    if sl_price < existing_sl:
                        logger.warning(f"Conflict Block (Rule 2): New SL {sl_price} below existing SL {existing_sl}")
                        return self._failure("conflict_blocked", "Rule 2 violation", symbol, direction, signal_category, sl=sl_price, tp=tp_price)
                else: # Sell
                    if sl_price > existing_sl:
                        logger.warning(f"Conflict Block (Rule 2): New SL {sl_price} above existing SL {existing_sl}")
                        return self._failure("conflict_blocked", "Rule 2 violation", symbol, direction, signal_category, sl=sl_price, tp=tp_price)

            # Rule 3: Different category + opposite direction -> TP/SL check
            if existing_direction != direction and existing_category != "reversal":
                if direction == 1: # Buy new vs sell existing
                    if tp_price > existing_sl:
                        logger.warning(f"Conflict Block (Rule 3): New TP {tp_price} crosses existing SL {existing_sl}")
                        return self._failure("conflict_blocked", "Rule 3 violation", symbol, direction, signal_category, sl=sl_price, tp=tp_price)
                else: # Sell new vs buy existing
                    if tp_price < existing_sl:
                        logger.warning(f"Conflict Block (Rule 3): New TP {tp_price} crosses existing SL {existing_sl}")
                        return self._failure("conflict_blocked", "Rule 3 violation", symbol, direction, signal_category, sl=sl_price, tp=tp_price)

        # 5. Lot Sizing
        sizing_res = self.ps.calculate_lot_size(symbol, market_price, sl_price, risk_pct, balance)
        if not sizing_res["success"]:
            return self._failure("sizing_failed", sizing_res["error"], symbol, direction, signal_category, sl=sl_price, tp=tp_price, risk=risk_pct)

        lot_size = sizing_res["lot_size"]
        actual_risk_pct = sizing_res["risk_pct_actual"]

        # 6. Execute Open
        try:
            open_res = self.pm.open_position(symbol, direction, lot_size, sl_price, tp_price, strategy, comment)
        except Exception as e:
            logger.error(f"Exception during open_position: {e}")
            return self._failure("open_failed", str(e), symbol, direction, signal_category, lot=lot_size, sl=sl_price, tp=tp_price, risk=actual_risk_pct)

        # 7. Post-Execution Handling
        if open_res["success"]:
            ticket = open_res["ticket"]
            actual_entry = open_res["entry_price"]
            actual_sl = open_res["sl_price"]
            actual_tp = open_res["tp_price"]

            # Update Category State
            self.ticket_categories[ticket] = signal_category
            self._save_state()

            # Register with ExitManager
            try:
                # Note: register_position signature follows the 6-arg interface provided in the prompt
                self.em.register_position(
                    ticket=ticket,
                    entry_price=actual_entry,
                    sl_price=actual_sl,
                    direction=direction,
                    stage=stage,
                    final_tp=tp_level
                )
                logger.info(f"Registered ticket {ticket} with ExitManager")
            except Exception as e:
                logger.error(f"Failed to register ticket {ticket} with ExitManager: {e}")
                # We don't fail the whole trade if just registration fails, but we note it.
                # Actually, requirement 8 says "Registers with ExitManager on success",
                # and schema says reason="register_failed" is a possibility.
                # Let's return success but with a note or consider it a failure?
                # Usually if ExitManager fails, the trade has no exit logic.
                # The schema has "register_failed", so let's use it.
                return self._failure("register_failed", f"ExitManager registration failed: {e}", symbol, direction, signal_category, lot=lot_size, entry=actual_entry, sl=actual_sl, tp=actual_tp, risk=actual_risk_pct)

            # Log to TradingJournal
            self.tj.log_order_open(
                signal_id=signal_id,
                ticket=ticket,
                actual_entry=actual_entry,
                actual_sl=actual_sl,
                actual_tp=actual_tp,
                lot_size=lot_size,
                risk_pct=actual_risk_pct
            )

            logger.info(f"SUCCESS: {symbol} {direction} Lot:{lot_size} Ticket:{ticket} Category:{signal_category}")

            return {
                "success": True,
                "reason": "ok",
                "ticket": ticket,
                "symbol": symbol,
                "direction": direction,
                "lot_size": lot_size,
                "entry_price": actual_entry,
                "sl_price": actual_sl,
                "tp_price": actual_tp,
                "risk_pct": actual_risk_pct,
                "signal_category": signal_category,
                "error_detail": "",
            }
        else:
            # open_position failed
            error_detail = open_res.get("comment", "Unknown MT5 error")
            self.tj.log_order_failure(signal_id, f"MT5 open failed: {error_detail}")
            return self._failure("open_failed", error_detail, symbol, direction, signal_category, lot=lot_size, sl=sl_price, tp=tp_price, risk=actual_risk_pct)

    def _failure(self, reason, detail, symbol, direction, category, lot=0.0, entry=None, sl=0.0, tp=0.0, risk=0.0):
        res = {
            "success": False,
            "reason": reason,
            "ticket": None,
            "symbol": symbol,
            "direction": direction,
            "lot_size": lot,
            "entry_price": entry,
            "sl_price": sl,
            "tp_price": tp,
            "risk_pct": risk,
            "signal_category": category,
            "error_detail": detail,
        }
        return res

if __name__ == "__main__":
    import unittest.mock as mock
    import sys
    # Mock MT5 for local testing in environments where it's not installed
    if mt5 is None:
        mock_mt5 = mock.MagicMock()
        sys.modules["MetaTrader5"] = mock_mt5
        import MetaTrader5 as mt5

    import shutil

    logging.basicConfig(level=logging.INFO)

    def setup_mocks():
        pm = mock.MagicMock()
        pt = mock.MagicMock()
        dm = mock.MagicMock()
        ps = mock.MagicMock()
        em = mock.MagicMock()
        tj = mock.MagicMock()

        # Default MT5 mocks
        mt5.symbol_info_tick.return_value = mock.MagicMock(ask=1.1000, bid=1.0990)
        mt5.account_info.return_value = mock.MagicMock(balance=10000.0)

        return pm, pt, dm, ps, em, tj

    def cleanup():
        if os.path.exists("test_send_order_state.json"):
            os.remove("test_send_order_state.json")

    print("\n--- Starting SendOrder Tests ---")

    # 1. Market order success (buy, standard, multi, tp_level=2)
    print("Test 1: Market order success (buy, standard, multi, tp_level=2)")
    pm, pt, dm, ps, em, tj = setup_mocks()
    dm.trading_allowed.return_value = True
    dm.max_risk_pct.return_value = 0.03
    pt.get_open_positions.return_value = []
    ps.calculate_lot_size.return_value = {"success": True, "lot_size": 0.1, "risk_pct_actual": 0.01}
    pm.open_position.return_value = {
        "success": True, "ticket": 1001, "symbol": "EURUSD_o", "direction": 1,
        "lot_size": 0.1, "entry_price": 1.1005, "sl_price": 1.0950, "tp_price": 1.1105,
        "strategy": "unity"
    }

    mt5.symbol_info_tick.return_value = mock.MagicMock(ask=1.1000, bid=1.0990)
    mt5.account_info.return_value = mock.MagicMock(balance=10000.0)

    so = SendOrder(pm, pt, dm, ps, em, tj, "test_send_order_state.json")
    res = so.execute("EURUSD_o", 1, 0.0, 1.0950, 2, "multi", "unity", "standard", "sig_123")

    assert res["success"] == True
    assert res["ticket"] == 1001
    assert res["reason"] == "ok"
    assert 1001 in so.ticket_categories
    em.register_position.assert_called_once()
    tj.log_order_open.assert_called_once()
    print("Test 1 Passed.")

    # 2. Drawdown blocked
    print("Test 2: Drawdown blocked")
    pm, pt, dm, ps, em, tj = setup_mocks()
    dm.trading_allowed.return_value = False

    mt5.symbol_info_tick.return_value = mock.MagicMock(ask=1.1000, bid=1.0990)
    so = SendOrder(pm, pt, dm, ps, em, tj, "test_send_order_state.json")
    res = so.execute("EURUSD_o", 1, 0.0, 1.0950, 2, "multi", "unity", "standard", "sig_123")

    assert res["success"] == False
    assert res["reason"] == "drawdown_blocked"
    assert res["ticket"] is None
    print("Test 2 Passed.")

    # 3. Same category + same direction conflict
    print("Test 3: Same category + same direction conflict")
    pm, pt, dm, ps, em, tj = setup_mocks()
    dm.trading_allowed.return_value = True
    dm.max_risk_pct.return_value = 0.03
    pt.get_open_positions.return_value = [{"ticket": 1001, "symbol": "EURUSD_o", "direction": 1, "sl_price": 1.0950, "tp_price": 1.1100}]

    mt5.symbol_info_tick.return_value = mock.MagicMock(ask=1.1000, bid=1.0990)
    so = SendOrder(pm, pt, dm, ps, em, tj, "test_send_order_state.json")
    so.ticket_categories[1001] = "standard"

    res = so.execute("EURUSD_o", 1, 0.0, 1.0950, 2, "multi", "unity", "standard", "sig_123")
    assert res["success"] == False
    assert res["reason"] == "conflict_blocked"
    print("Test 3 Passed.")

    # 4. Different category + same direction — SL conflict
    print("Test 4: Different category + same direction — SL conflict")
    pm, pt, dm, ps, em, tj = setup_mocks()
    dm.trading_allowed.return_value = True
    dm.max_risk_pct.return_value = 0.03
    pt.get_open_positions.return_value = [{"ticket": 1001, "symbol": "EURUSD_o", "direction": 1, "sl_price": 1.0950, "tp_price": 1.1100}]

    mt5.symbol_info_tick.return_value = mock.MagicMock(ask=1.1000, bid=1.0990)
    so = SendOrder(pm, pt, dm, ps, em, tj, "test_send_order_state.json")
    so.ticket_categories[1001] = "standard"

    # New buy with SL lower than 1.0950 -> blocked
    res = so.execute("EURUSD_o", 1, 0.0, 1.0940, 2, "multi", "unity", "high_risk", "sig_124")
    assert res["success"] == False
    assert res["reason"] == "conflict_blocked"
    print("Test 4 Passed.")

    # 5. Different category + same direction — SL OK
    print("Test 5: Different category + same direction — SL OK")
    pm, pt, dm, ps, em, tj = setup_mocks()
    dm.trading_allowed.return_value = True
    dm.max_risk_pct.return_value = 0.03
    pt.get_open_positions.return_value = [{"ticket": 1001, "symbol": "EURUSD_o", "direction": 1, "sl_price": 1.0950, "tp_price": 1.1100}]
    ps.calculate_lot_size.return_value = {"success": True, "lot_size": 0.1, "risk_pct_actual": 0.005}
    pm.open_position.return_value = {"success": True, "ticket": 1002, "symbol": "EURUSD_o", "entry_price": 1.1001, "sl_price": 1.0960, "tp_price": 1.1100}

    mt5.symbol_info_tick.return_value = mock.MagicMock(ask=1.1000, bid=1.0990)
    mt5.account_info.return_value = mock.MagicMock(balance=10000.0)

    so = SendOrder(pm, pt, dm, ps, em, tj, "test_send_order_state.json")
    so.ticket_categories[1001] = "standard"

    # New buy with SL higher than 1.0950 -> allowed
    res = so.execute("EURUSD_o", 1, 0.0, 1.0960, 2, "multi", "unity", "high_risk", "sig_124")
    assert res["success"] == True
    print("Test 5 Passed.")

    # 6. Opposite direction — TP crosses existing SL (non-reversal existing)
    print("Test 6: Opposite direction — TP crosses existing SL (non-reversal existing)")
    pm, pt, dm, ps, em, tj = setup_mocks()
    dm.trading_allowed.return_value = True
    dm.max_risk_pct.return_value = 0.03
    pt.get_open_positions.return_value = [{"ticket": 1001, "symbol": "EURUSD_o", "direction": 1, "sl_price": 1.0950, "tp_price": 1.1100}]

    mt5.symbol_info_tick.return_value = mock.MagicMock(ask=1.1000, bid=1.0990) # bid 1.0990 for sell
    so = SendOrder(pm, pt, dm, ps, em, tj, "test_send_order_state.json")
    so.ticket_categories[1001] = "standard"

    # New sell at 1.0990, SL at 1.1050. TP at 2R: 1.0990 - 2*(1.1050-1.0990) = 1.0990 - 0.0120 = 1.0870.
    # Wait, if TP is 1.0940 (below existing SL 1.0950) it should block.
    # R = 1.1050 - 1.0990 = 0.0060. tp_level=1 -> TP = 1.0990 - 0.0060 = 1.0930. 1.0930 < 1.0950 -> blocked.
    res = so.execute("EURUSD_o", -1, 0.0, 1.1050, 1, "multi", "unity", "high_risk", "sig_125")
    assert res["success"] == False
    assert res["reason"] == "conflict_blocked"
    print("Test 6 Passed.")

    # 7. Opposite direction — existing is reversal (exception applies)
    print("Test 7: Opposite direction — existing is reversal (exception applies)")
    pm, pt, dm, ps, em, tj = setup_mocks()
    dm.trading_allowed.return_value = True
    dm.max_risk_pct.return_value = 0.03
    pt.get_open_positions.return_value = [{"ticket": 1001, "symbol": "EURUSD_o", "direction": 1, "sl_price": 1.0950, "tp_price": 1.1100}]
    ps.calculate_lot_size.return_value = {"success": True, "lot_size": 0.1, "risk_pct_actual": 0.01}
    pm.open_position.return_value = {"success": True, "ticket": 1002, "symbol": "EURUSD_o", "entry_price": 1.0991, "sl_price": 1.1050, "tp_price": 1.0930}

    mt5.symbol_info_tick.return_value = mock.MagicMock(ask=1.1000, bid=1.0990)
    mt5.account_info.return_value = mock.MagicMock(balance=10000.0)

    so = SendOrder(pm, pt, dm, ps, em, tj, "test_send_order_state.json")
    so.ticket_categories[1001] = "reversal"

    # Should be allowed because existing is reversal
    res = so.execute("EURUSD_o", -1, 0.0, 1.1050, 1, "multi", "unity", "standard", "sig_126")
    assert res["success"] == True
    print("Test 7 Passed.")

    # 8. Lot size below minimum
    print("Test 8: Lot size below minimum")
    pm, pt, dm, ps, em, tj = setup_mocks()
    dm.trading_allowed.return_value = True
    dm.max_risk_pct.return_value = 0.03
    pt.get_open_positions.return_value = []
    ps.calculate_lot_size.return_value = {"success": False, "error": "lot_size_below_minimum"}

    mt5.symbol_info_tick.return_value = mock.MagicMock(ask=1.1000, bid=1.0990)
    mt5.account_info.return_value = mock.MagicMock(balance=10000.0)

    so = SendOrder(pm, pt, dm, ps, em, tj, "test_send_order_state.json")
    res = so.execute("EURUSD_o", 1, 0.0, 1.0950, 2, "multi", "unity", "standard", "sig_127")
    assert res["success"] == False
    assert res["reason"] == "sizing_failed"
    print("Test 8 Passed.")

    # 9. MT5 open_position failure
    print("Test 9: MT5 open_position failure")
    pm, pt, dm, ps, em, tj = setup_mocks()
    dm.trading_allowed.return_value = True
    dm.max_risk_pct.return_value = 0.03
    pt.get_open_positions.return_value = []
    ps.calculate_lot_size.return_value = {"success": True, "lot_size": 0.1, "risk_pct_actual": 0.01}
    pm.open_position.return_value = {"success": False, "comment": "Invalid volume"}

    mt5.symbol_info_tick.return_value = mock.MagicMock(ask=1.1000, bid=1.0990)
    mt5.account_info.return_value = mock.MagicMock(balance=10000.0)

    so = SendOrder(pm, pt, dm, ps, em, tj, "test_send_order_state.json")
    res = so.execute("EURUSD_o", 1, 0.0, 1.0950, 2, "multi", "unity", "standard", "sig_128")
    assert res["success"] == False
    assert res["reason"] == "open_failed"
    tj.log_order_failure.assert_called_once()
    print("Test 9 Passed.")

    # 10. State file persistence
    print("Test 10: State file persistence")
    cleanup()
    pm, pt, dm, ps, em, tj = setup_mocks()
    dm.trading_allowed.return_value = True
    dm.max_risk_pct.return_value = 0.03
    pt.get_open_positions.return_value = []
    ps.calculate_lot_size.return_value = {"success": True, "lot_size": 0.1, "risk_pct_actual": 0.01}
    pm.open_position.return_value = {"success": True, "ticket": 5005, "symbol": "EURUSD_o", "entry_price": 1.1000, "sl_price": 1.0950, "tp_price": 1.1100}

    mt5.symbol_info_tick.return_value = mock.MagicMock(ask=1.1000, bid=1.0990)
    mt5.account_info.return_value = mock.MagicMock(balance=10000.0)

    so1 = SendOrder(pm, pt, dm, ps, em, tj, "test_send_order_state.json")
    so1.execute("EURUSD_o", 1, 0.0, 1.0950, 2, "multi", "unity", "standard", "sig_999")
    assert 5005 in so1.ticket_categories

    # New instance, same file
    so2 = SendOrder(pm, pt, dm, ps, em, tj, "test_send_order_state.json")
    assert 5005 in so2.ticket_categories
    assert so2.ticket_categories[5005] == "standard"
    print("Test 10 Passed.")

    cleanup()
    print("\n--- All SendOrder Tests Passed Successfully ---")
