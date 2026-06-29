import os
import json
import logging
import threading
import math
from datetime import datetime, timezone
import MetaTrader5 as mt5

# Constants for when MetaTrader5 is not installed (e.g. during local testing),
# matching the fallback convention already used in position_manager.py
if not hasattr(mt5, "DEAL_REASON_CLIENT"):
    DEAL_REASON_CLIENT = 0
    DEAL_REASON_MOBILE = 1
    DEAL_REASON_WEB = 2
    DEAL_REASON_EXPERT = 3
    DEAL_REASON_SL = 4
    DEAL_REASON_TP = 5
    DEAL_REASON_SO = 6
else:
    DEAL_REASON_CLIENT = mt5.DEAL_REASON_CLIENT
    DEAL_REASON_MOBILE = mt5.DEAL_REASON_MOBILE
    DEAL_REASON_WEB = mt5.DEAL_REASON_WEB
    DEAL_REASON_EXPERT = mt5.DEAL_REASON_EXPERT
    DEAL_REASON_SL = mt5.DEAL_REASON_SL
    DEAL_REASON_TP = mt5.DEAL_REASON_TP
    DEAL_REASON_SO = mt5.DEAL_REASON_SO

# Reasons that represent an expected, broker-driven closure (no human/bot action needed)
EXPECTED_CLOSE_REASONS = {DEAL_REASON_SL, DEAL_REASON_TP, DEAL_REASON_SO}

REASON_LABELS = {
    DEAL_REASON_CLIENT: "manual_client",
    DEAL_REASON_MOBILE: "manual_mobile",
    DEAL_REASON_WEB: "manual_web",
    DEAL_REASON_EXPERT: "expert_advisor",
    DEAL_REASON_SL: "stop_loss",
    DEAL_REASON_TP: "take_profit",
    DEAL_REASON_SO: "stop_out",
}

logger = logging.getLogger("ExitManager")

class ExitManager:
    def __init__(
        self,
        position_tracker,
        position_manager,
        poll_interval_seconds: float = 1.0,
        state_file: str = "exit_manager_state.json",
        trading_journal = None,
    ):
        self.position_tracker = position_tracker
        self.position_manager = position_manager
        self.poll_interval_seconds = poll_interval_seconds
        self.state_file = state_file
        self.trading_journal = trading_journal
        
        self.tracked_tickets = {}  # ticket -> state dict
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        
        self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    # JSON keys are always strings, convert tickets back to int
                    raw_tickets = data.get("tracked_tickets", {})
                    restored = {}
                    for k, v in raw_tickets.items():
                        ticket = int(k)
                        # Also convert tp_prices keys to int
                        if "tp_prices" in v:
                            v["tp_prices"] = {int(tp_k): tp_v for tp_k, tp_v in v["tp_prices"].items()}
                        restored[ticket] = v
                    self.tracked_tickets = restored
                    logger.info(f"ExitManager state restored from {self.state_file}. {len(self.tracked_tickets)} tickets loaded.")
            except Exception as e:
                logger.error(f"Failed to load ExitManager state: {e}")

    def _save_state(self):
        try:
            with self._lock:
                data = {
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "tracked_tickets": self.tracked_tickets
                }
            temp_file = self.state_file + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=4, default=str)
            os.replace(temp_file, self.state_file)
        except Exception as e:
            logger.error(f"Failed to save ExitManager state: {e}")

    def start(self) -> None:
        """Start background polling loop in a daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("ExitManager is already running.")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("ExitManager started.")

    def stop(self) -> None:
        """Stop the polling loop cleanly."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("ExitManager stopped.")

    def _poll_loop(self):
        while not self._stop_event.is_set():
            try:
                self._poll_cycle()
            except Exception as e:
                logger.exception(f"Unexpected error in ExitManager poll loop: {e}")
            
            self._stop_event.wait(self.poll_interval_seconds)

    def _poll_cycle(self):
        """
        Main evaluation cycle: reconciles with PositionTracker and evaluates TP/SL logic.
        """
        open_positions = self.position_tracker.get_open_positions()
        live_tickets = {p["ticket"]: p for p in open_positions}
        
        with self._lock:
            tracked_list = list(self.tracked_tickets.items())

        for ticket, state in tracked_list:
            if state["closed"]:
                continue

            if ticket not in live_tickets:
                self._handle_disappeared_ticket(ticket)
                continue

            # Initialization if needed
            if state["original_lot_size"] is None:
                live_pos = live_tickets[ticket]
                if not self._initialize_ticket_shares(ticket, live_pos["symbol"], live_pos["lot_size"]):
                    continue
                self._save_state()

            self._evaluate_ticket(ticket, live_tickets[ticket])

    def _handle_disappeared_ticket(self, ticket: int) -> None:
        """
        Called when a tracked ticket is no longer present among live positions.
        Classifies the closure as expected (broker-native SL/TP/stop-out) or
        genuinely unexpected (manual/external/EA closure, desync), and logs
        accordingly before removing it from active tracking.
        """
        label, reason_code = self._get_close_reason(ticket)

        if reason_code in EXPECTED_CLOSE_REASONS:
            logger.info(f"Ticket {ticket} closed by broker ({label}). Removing from active tracking.")
        elif reason_code is None:
            logger.warning(f"Ticket {ticket} disappeared from tracker; close reason could not be determined ({label}).")
        else:
            logger.warning(f"Ticket {ticket} disappeared via unexpected closure ({label}). Possible manual intervention, EA conflict, or desync.")

        # Journal Hook
        state = self.tracked_tickets.get(ticket)
        if state and self.trading_journal and state.get("signal_id"):
            try:
                pnl = 0.0
                deals = mt5.history_deals_get(position=ticket)
                if deals:
                    pnl = sum(d.profit for d in deals)

                duration = 0
                if state.get("open_time"):
                    ot = datetime.fromisoformat(state["open_time"])
                    duration = int((datetime.now(timezone.utc) - ot).total_seconds())

                self.trading_journal.log_outcome(
                    signal_id=state["signal_id"],
                    ticket=ticket,
                    outcome=label,
                    close_price=0.0, # Will be 0 if we can't find it easily from disappeared
                    pnl_dollars=pnl,
                    duration_seconds=duration
                )
            except Exception as e:
                logger.error(f"Failed to log outcome for disappeared ticket {ticket}: {e}")

        with self._lock:
            if ticket in self.tracked_tickets:
                del self.tracked_tickets[ticket]
        self._save_state()

    def _get_close_reason(self, ticket: int):
        """
        Looks up the most recent closing deal for a position ticket via MT5
        deal history. Returns (label: str, reason_code: int | None).
        reason_code is None when no deal history could be retrieved, in which
        case the closure cannot be classified and should be treated cautiously.
        """
        try:
            deals = mt5.history_deals_get(position=ticket)
        except Exception as e:
            logger.error(f"Failed to query deal history for ticket {ticket}: {e}")
            return ("query_failed", None)

        if not deals:
            return ("no_deal_history", None)

        last_deal = max(deals, key=lambda d: d.time)
        reason_code = last_deal.reason
        label = REASON_LABELS.get(reason_code, f"unknown_reason_{reason_code}")
        return (label, reason_code)

    def _evaluate_ticket(self, ticket: int, live_pos: dict):
        """
        Evaluates TP/SL logic for a single ticket.
        """
        state = self.tracked_tickets[ticket]
        current_price = live_pos["current_price"]
        direction = state["direction"]
        
        if state["stage"] == "multi":
            self._handle_multi_stage(ticket, current_price)
        else:
            self._handle_single_stage(ticket, current_price)

    def _handle_multi_stage(self, ticket: int, current_price: float):
        state = self.tracked_tickets[ticket]
        next_stage = state["current_stage_reached"] + 1
        
        if next_stage > state["final_tp"]:
            return

        tp_level = state["tp_prices"][next_stage]
        direction = state["direction"]

        # direction-aware "reached" check
        reached = (current_price >= tp_level) if direction == 1 else (current_price <= tp_level)

        if reached:
            is_final = (next_stage == state["final_tp"])
            
            if not is_final:
                # Partial close
                vol = state["share"]
                res = self.position_manager.close_position(ticket, volume=vol)
                if not res["success"]:
                    logger.error(f"Failed to execute partial close for ticket {ticket} at stage {next_stage}")
                    return

                # Journal Hook
                if self.trading_journal and state.get("signal_id"):
                    self.trading_journal.log_partial_close(
                        signal_id=state["signal_id"],
                        ticket=ticket,
                        stage_reached=next_stage,
                        closed_volume=vol,
                        close_price=current_price,
                        new_sl=state["entry_price"] if next_stage == 1 else state["tp_prices"][next_stage - 1]
                    )

                # Move SL
                new_sl = state["entry_price"] if next_stage == 1 else state["tp_prices"][next_stage - 1]
                mod_res = self.position_manager.modify_position(ticket, sl_price=new_sl)
                if not mod_res["success"]:
                    logger.error(f"Failed to move SL for ticket {ticket} after stage {next_stage}")
                    # Continue anyway, we already closed the partial lot
                
                state["current_stage_reached"] = next_stage
                logger.info(f"Multi-stage {next_stage} reached for ticket {ticket}. Closed {vol}, moved SL to {new_sl}")
                self._save_state()
            else:
                # Final stage
                # volume=None or the remaining_lot_size_including_leftover
                # Spec says: remaining_lot_size_including_leftover. 
                # Since PositionManager handles volume=None as full close, that's safer.
                res = self.position_manager.close_position(ticket, volume=None)
                if not res["success"]:
                    logger.error(f"Failed to execute final close for ticket {ticket}")
                    return
                
                self._finalize_ticket(ticket)
                logger.info(f"Final stage {next_stage} reached for ticket {ticket}. Position fully closed.")

    def _handle_single_stage(self, ticket: int, current_price: float):
        state = self.tracked_tickets[ticket]
        direction = state["direction"]
        final_tp_stage = state["final_tp"]
        
        # 1. Check for breakeven move (only if final_tp > 1)
        if final_tp_stage > 1 and not state["sl_moved_to_breakeven"]:
            tp1_level = state["tp_prices"][1]
            if (direction == 1 and current_price >= tp1_level) or (direction == -1 and current_price <= tp1_level):
                res = self.position_manager.modify_position(ticket, sl_price=state["entry_price"])
                if res["success"]:
                    state["sl_moved_to_breakeven"] = True
                    logger.info(f"Breakeven moved for ticket {ticket} (Single-stage mode, TP1 hit)")
                    self._save_state()
                else:
                    logger.error(f"Failed to move SL to breakeven for ticket {ticket}")

        # 2. Check for final close
        final_tp_level = state["tp_prices"][final_tp_stage]
        if (direction == 1 and current_price >= final_tp_level) or (direction == -1 and current_price <= final_tp_level):
            res = self.position_manager.close_position(ticket, volume=None)
            if res["success"]:
                self._finalize_ticket(ticket)
                logger.info(f"Final TP {final_tp_stage} reached for ticket {ticket}. Single-stage exit complete.")
            else:
                logger.error(f"Failed to execute final close for ticket {ticket}")

    def _finalize_ticket(self, ticket: int):
        state = self.tracked_tickets.get(ticket)
        if state and self.trading_journal and state.get("signal_id"):
            try:
                pnl = 0.0
                deals = mt5.history_deals_get(position=ticket)
                if deals:
                    pnl = sum(d.profit for d in deals)

                duration = 0
                if state.get("open_time"):
                    ot = datetime.fromisoformat(state["open_time"])
                    duration = int((datetime.now(timezone.utc) - ot).total_seconds())

                # Get outcome label
                next_stage = state.get("current_stage_reached", 0) + 1
                outcome_label = f"tp{next_stage}"

                self.trading_journal.log_outcome(
                    signal_id=state["signal_id"],
                    ticket=ticket,
                    outcome=outcome_label,
                    close_price=0.0, # Handled by journal context for routing
                    pnl_dollars=pnl,
                    duration_seconds=duration
                )
            except Exception as e:
                logger.error(f"Failed to log outcome for ticket {ticket}: {e}")

        with self._lock:
            if ticket in self.tracked_tickets:
                self.tracked_tickets[ticket]["closed"] = True
                # Optional: del self.tracked_tickets[ticket] if we don't want it in JSON anymore
                # But spec says "mark closed=True", so let's keep it for history until next restart?
                # Actually, better to remove from active tracking to keep JSON lean.
                del self.tracked_tickets[ticket]
        self._save_state()

    def register_position(
        self,
        ticket: int,
        entry_price: float,
        sl_price: float,
        direction: int,       # 1 = buy, -1 = sell
        stage: str,           # "single" or "multi"
        final_tp: int,        # 1, 2, 3, or 4
        signal_id: str = None,
    ) -> None:
        """
        Registers an open position for exit management.
        Calculates TP price ladder and initializes internal tracking state.
        """
        with self._lock:
            if ticket in self.tracked_tickets:
                return

            # TP price ladder calculation
            R = abs(entry_price - sl_price)
            sign = 1 if direction == 1 else -1
            tp_prices = {
                i: float(entry_price + sign * i * R)
                for i in range(1, final_tp + 1)
            }

            self.tracked_tickets[ticket] = {
                "signal_id": signal_id,
                "open_time": datetime.now(timezone.utc).isoformat(),
                "entry_price": float(entry_price),
                "sl_price_original": float(sl_price),
                "direction": int(direction),
                "stage": stage,
                "final_tp": int(final_tp),
                "tp_prices": tp_prices,
                "original_lot_size": None,        # Captured from tracker in _poll_cycle
                "symbol": None,                   # Captured from tracker
                "share": 0.0,
                "leftover": 0.0,
                "current_stage_reached": 0,
                "sl_moved_to_breakeven": False,
                "closed": False,
            }
        
        self._save_state()
        logger.info(f"Registered ticket {ticket} for {stage} exit (final_tp={final_tp})")

    def _initialize_ticket_shares(self, ticket: int, symbol: str, original_lot: float) -> bool:
        """
        Calculates stage shares and leftover lot size once symbol info is available.
        """
        state = self.tracked_tickets[ticket]
        
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Failed to get symbol info for {symbol} to initialize ticket {ticket}")
            return False

        volume_step = info.volume_step
        
        state["symbol"] = symbol
        state["original_lot_size"] = original_lot
        
        if state["stage"] == "multi":
            n = state["final_tp"]
            # share = original_lot / N, rounded DOWN to volume_step
            share = math.floor(round((original_lot / n) / volume_step, 10)) * volume_step
            
            # Precision-drift guard for rounding
            step_str = f"{volume_step:.8f}".rstrip('0').rstrip('.')
            precision = len(step_str.split('.')[1]) if '.' in step_str else 0
            share = round(share, precision)
            
            # Leftover = original_lot - sum(rounded shares)
            leftover = original_lot - (share * n)
            leftover = round(leftover, precision)
            
            state["share"] = share
            state["leftover"] = leftover
            logger.info(f"Initialized multi-stage shares for ticket {ticket}: {n} stages, share={share}, leftover={leftover}")
        else:
            logger.info(f"Initialized single-stage exit for ticket {ticket}: {original_lot} lot")

        return True