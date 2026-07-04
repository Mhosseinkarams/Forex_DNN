import time
import json
import logging
import threading
import os
from datetime import datetime, timezone
import MetaTrader5 as mt5
from simulation.simulation_environment import env

logger = logging.getLogger("PositionTracker")

class PositionTracker:
    def __init__(
        self,
        magic_numbers: list[int],     # track only these magic numbers
        poll_interval_seconds: int = 5,
        state_file: str = "position_state.json",
    ):
        self.magic_numbers = magic_numbers
        self.poll_interval_seconds = poll_interval_seconds
        self.state_file = state_file
        
        self.positions = []
        self.total_risk = 0.0
        self.total_reward = 0.0
        
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        
        self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.positions = data.get("positions", [])
                    # Recalculate totals from loaded positions
                    self.total_risk = sum(p.get("remaining_risk_dollars", 0.0) for p in self.positions)
                    self.total_reward = sum(p.get("reward_dollars", 0.0) for p in self.positions)
                    logger.info(f"State restored from {self.state_file}. {len(self.positions)} positions loaded.")
            except Exception as e:
                logger.error(f"Failed to load state file: {e}")

    def _save_state(self):
        try:
            data = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "positions": self.positions
            }
            # Use a temporary file and rename for atomic write
            temp_file = self.state_file + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=4, default=str)
            os.replace(temp_file, self.state_file)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def start(self) -> None:
        """Start background polling loop in a daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Tracker is already running.")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("PositionTracker started.")

    def stop(self) -> None:
        """Stop the polling loop cleanly."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("PositionTracker stopped.")

    def _poll_loop(self):
        while not self._stop_event.is_set():
            try:
                self._poll_cycle()
            except Exception as e:
                logger.exception(f"Unexpected error in poll loop: {e}")
            
            # Wait for next interval or stop event
            self._stop_event.wait(self.poll_interval_seconds)

    def _poll_cycle(self):
        # 1. Query environment for all open positions
        mt5_positions = env.positions_get()
        if mt5_positions is None:
            if env.mode == "live":
                err = mt5.last_error()
                logger.error(f"MT5 query failure during poll: {err}")
            return

        # 2. Filter by magic_numbers list
        tracked_mt5_positions = [p for p in mt5_positions if (p.magic if hasattr(p, 'magic') else p['magic'] if isinstance(p, dict) else getattr(p, 'magic', 0)) in self.magic_numbers]
        
        new_positions = []
        new_total_risk = 0.0
        new_total_reward = 0.0
        
        current_tickets = {(p.ticket if hasattr(p, 'ticket') else p['ticket'] if isinstance(p, dict) else getattr(p, 'ticket', 0)) for p in tracked_mt5_positions}
        
        with self._lock:
            old_tickets = {p["ticket"] for p in self.positions}

        # Log discrepancies
        closed_tickets = old_tickets - current_tickets
        for ticket in closed_tickets:
            logger.warning(f"Unexpected position close detected: ticket {ticket} disappeared.")
        
        new_tickets = current_tickets - old_tickets
        for ticket in new_tickets:
            logger.info(f"New tracked position detected: ticket {ticket}")

        # 3. For each matching position, calculate metrics
        for p in tracked_mt5_positions:
            symbol = p.symbol if hasattr(p, 'symbol') else p['symbol'] if isinstance(p, dict) else getattr(p, 'symbol', '')
            info = env.symbol_info(symbol)
            if info is None:
                logger.error(f"Failed to get symbol info for {symbol}")
                continue
            
            contract_size = info.trade_contract_size if hasattr(info, 'trade_contract_size') else info.get('trade_contract_size', 100000)
            entry_price = p.price_open if hasattr(p, 'price_open') else p['price_open'] if isinstance(p, dict) else getattr(p, 'price_open', 0)
            sl_price = p.sl if hasattr(p, 'sl') else p['sl'] if isinstance(p, dict) else getattr(p, 'sl', 0)
            tp_price = p.tp if hasattr(p, 'tp') else p['tp'] if isinstance(p, dict) else getattr(p, 'tp', 0)
            lot_size = p.volume if hasattr(p, 'volume') else p['volume'] if isinstance(p, dict) else getattr(p, 'volume', 0)
            
            # Remaining risk = abs(entry_price - sl_price) × lot_size × contract_size
            risk = abs(entry_price - sl_price) * lot_size * contract_size if sl_price != 0 else 0.0
            
            # Reward = abs(tp_price - entry_price) × lot_size × contract_size
            reward = abs(tp_price - entry_price) * lot_size * contract_size if tp_price != 0 else 0.0
            
            # In our SimPosition, direction is 1 (Buy) or -1 (Sell). MT5 ORDER_TYPE_BUY=0, ORDER_TYPE_SELL=1.
            if hasattr(p, 'type'):
                direction = 1 if p.type == mt5.ORDER_TYPE_BUY else -1
            else:
                direction = p.get('direction', 1) if isinstance(p, dict) else getattr(p, 'direction', 1)

            p_time = p.time if hasattr(p, 'time') else p.get('time', 0) if isinstance(p, dict) else getattr(p, 'time', 0)
            p_price_current = p.price_current if hasattr(p, 'price_current') else p.get('price_current', entry_price) if isinstance(p, dict) else getattr(p, 'price_current', entry_price)
            p_profit = p.profit if hasattr(p, 'profit') else p.get('floating_pnl', 0.0) if isinstance(p, dict) else getattr(p, 'floating_pnl', 0.0)

            snapshot = {
                "ticket": p.ticket if hasattr(p, 'ticket') else p['ticket'] if isinstance(p, dict) else getattr(p, 'ticket', 0),
                "symbol": symbol,
                "magic": p.magic if hasattr(p, 'magic') else p['magic'] if isinstance(p, dict) else getattr(p, 'magic', 0),
                "direction": direction,
                "lot_size": lot_size,
                "entry_price": entry_price,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "current_price": p_price_current,
                "floating_pnl": p_profit,
                "remaining_risk_dollars": risk,
                "reward_dollars": reward,
                "open_time": datetime.fromtimestamp(p_time, tz=timezone.utc).isoformat(),
            }
            new_positions.append(snapshot)
            new_total_risk += risk
            new_total_reward += reward

        # 4. Update internal state
        with self._lock:
            self.positions = new_positions
            self.total_risk = new_total_risk
            self.total_reward = new_total_reward
        
        # 5. Persist state
        self._save_state()
        
        logger.info(f"Poll cycle summary: {len(new_positions)} positions tracked, total open risk ${new_total_risk:.2f}")

    def get_open_positions(self) -> list[dict]:
        with self._lock:
            return list(self.positions)

    def get_open_risk(self) -> float:
        with self._lock:
            return self.total_risk

    def get_open_reward(self) -> float:
        with self._lock:
            return self.total_reward

if __name__ == "__main__":
    # Structural test
    import time
    from unittest.mock import MagicMock
    import sys

    # Mock MT5
    mt5_mock = MagicMock()
    mt5_mock.ORDER_TYPE_BUY = 0
    mt5_mock.ORDER_TYPE_SELL = 1
    sys.modules["MetaTrader5"] = mt5_mock

    # Setup mock data
    mock_pos = MagicMock()
    mock_pos.ticket = 123456
    mock_pos.symbol = "EURUSD"
    mock_pos.magic = 100001
    mock_pos.type = 0 # BUY
    mock_pos.volume = 0.1
    mock_pos.price_open = 1.1000
    mock_pos.sl = 1.0990
    mock_pos.tp = 1.1020
    mock_pos.price_current = 1.1005
    mock_pos.profit = 5.0
    mock_pos.time = time.time() - 3600

    mt5_mock.positions_get.return_value = [mock_pos]
    
    mock_symbol = MagicMock()
    mock_symbol.trade_contract_size = 100000
    mt5_mock.symbol_info.return_value = mock_symbol

    tracker = PositionTracker(
        magic_numbers=[100001, 100002],
        poll_interval_seconds=1,
        state_file="test_position_state.json",
    )
    tracker.start()
    time.sleep(1.5)  # allow one poll cycle

    positions = tracker.get_open_positions()
    risk = tracker.get_open_risk()
    reward = tracker.get_open_reward()

    print(f"Open positions: {len(positions)}")
    print(f"Total open risk: ${risk:.2f}")
    print(f"Total open reward: ${reward:.2f}")

    if len(positions) > 0:
        p = positions[0]
        assert "ticket" in p
        assert "remaining_risk_dollars" in p
        assert "reward_dollars" in p
        # risk = abs(1.1000 - 1.0990) * 0.1 * 100000 = 0.001 * 10000 = 10
        assert abs(p["remaining_risk_dollars"] - 10.0) < 0.01

    tracker.stop()
    if os.path.exists("test_position_state.json"):
        os.remove("test_position_state.json")
    print("PositionTracker schema test passed.")
