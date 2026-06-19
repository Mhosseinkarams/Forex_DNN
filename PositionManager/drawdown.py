import os
import json
import logging
import threading
from datetime import datetime, timezone
import MetaTrader5 as mt5

logger = logging.getLogger("DrawdownManager")

class DrawdownManager:
    """
    Enforces daily and total drawdown ceilings by monitoring account balance and open risk.
    """

    def __init__(
        self,
        initial_balance: float,
        position_tracker,
        daily_limit_pct: float = 0.03,
        total_limit_pct: float = 0.10,
        state_file: str = "drawdown_state.json",
    ):
        self.initial_balance = initial_balance
        self.position_tracker = position_tracker
        self.daily_limit_pct = daily_limit_pct
        self.total_limit_pct = total_limit_pct
        self.state_file = state_file

        self.start_of_day_balance = initial_balance
        self.snapshot_date = ""  # YYYY-MM-DD

        self._lock = threading.Lock()

        # Internal metric states
        self._trading_allowed = True
        self._max_risk_pct = 0.0
        self._daily_loss_pct = 0.0
        self._total_loss_pct = 0.0
        self._remaining_daily_risk_pct = 0.0
        self._remaining_total_risk_pct = 0.0

        self._load_state()
        # Initialize/check day boundary on startup
        self._check_day_boundary()
        # Initial calculation
        self.check()

    def _load_state(self):
        """Loads persisted state from JSON file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    with self._lock:
                        self.start_of_day_balance = data.get("start_of_day_balance", self.initial_balance)
                        self.snapshot_date = data.get("snapshot_date", "")
                    logger.info(f"Drawdown state restored: date={self.snapshot_date}, start_balance={self.start_of_day_balance}")
            except Exception as e:
                logger.error(f"Failed to load state file: {e}")

    def _save_state(self):
        """Saves current state to JSON file atomically."""
        try:
            with self._lock:
                data = {
                    "start_of_day_balance": self.start_of_day_balance,
                    "snapshot_date": self.snapshot_date,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }
            temp_file = self.state_file + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=4, default=str)
            os.replace(temp_file, self.state_file)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _get_server_date(self) -> str | None:
        """
        Retrieves MT5 server date as 'YYYY-MM-DD'.
        Tries any selected symbol to get a tick, falling back to recent bars.
        """
        # Try to find a symbol to query for server time
        symbols_to_try = []

        # 1. Check open positions symbols
        positions = self.position_tracker.get_open_positions()
        if positions:
            symbols_to_try.extend(list({p['symbol'] for p in positions}))

        # 2. Add some common ones as fallback
        symbols_to_try.extend(["GBPUSD", "EURUSD", "USDJPY", "GBPUSD_i", "EURUSD_i"])

        for symbol in symbols_to_try:
            # Check if symbol is selected/available
            if not mt5.symbol_select(symbol, True):
                continue

            tick = mt5.symbol_info_tick(symbol)
            if tick is not None:
                return datetime.fromtimestamp(tick.time, tz=timezone.utc).strftime('%Y-%m-%d')

            # Fallback to last bar if tick unavailable
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
            if rates is not None and len(rates) > 0:
                return datetime.fromtimestamp(rates[0]['time'], tz=timezone.utc).strftime('%Y-%m-%d')

        return None

    def _check_day_boundary(self):
        """Detects day rollover and snapshots balance if needed."""
        server_date = self._get_server_date()
        if server_date is None:
            logger.warning("Could not retrieve MT5 server date for boundary check.")
            return

        with self._lock:
            needs_snapshot = (server_date != self.snapshot_date)

        if needs_snapshot:
            acc = mt5.account_info()
            if acc is not None:
                with self._lock:
                    self.start_of_day_balance = acc.balance
                    self.snapshot_date = server_date
                self._save_state()
                logger.info(f"Day boundary detected. New snapshot: date={server_date}, balance={acc.balance}")
            else:
                logger.warning("Failed to retrieve account info for balance snapshot.")

    def check(self):
        """
        Recomputes all drawdown metrics and updates trading permission.
        Should be called frequently in the main trading loop.
        """
        self._check_day_boundary()

        acc = mt5.account_info()
        if acc is None:
            logger.error("Failed to retrieve account info during drawdown check.")
            return

        current_balance = acc.balance
        open_risk_dollars = self.position_tracker.get_open_risk()

        with self._lock:
            # Core calculations
            open_risk_pct = open_risk_dollars / current_balance if current_balance > 0 else 0

            # Daily Branch
            daily_loss_pct = max((self.start_of_day_balance - current_balance) / self.start_of_day_balance, 0) if self.start_of_day_balance > 0 else 0
            total_committed_daily_pct = daily_loss_pct + open_risk_pct
            self._remaining_daily_risk_pct = self.daily_limit_pct - total_committed_daily_pct

            # Overall Branch
            total_loss_pct = max((self.initial_balance - current_balance) / self.initial_balance, 0) if self.initial_balance > 0 else 0
            total_committed_overall_pct = total_loss_pct + open_risk_pct
            self._remaining_total_risk_pct = self.total_limit_pct - total_committed_overall_pct

            # Determination
            new_max_risk_pct = min(self._remaining_daily_risk_pct, self._remaining_total_risk_pct)
            new_trading_allowed = new_max_risk_pct > 0

            # State Update and Logging
            if new_trading_allowed != self._trading_allowed:
                if not new_trading_allowed:
                    logger.warning(f"Drawdown Limit Breached! Trading blocked. Max Risk Remaining: {new_max_risk_pct:.4%}")
                else:
                    logger.info(f"Drawdown Limit Recovered. Trading permitted. Max Risk Remaining: {new_max_risk_pct:.4%}")

            self._trading_allowed = new_trading_allowed
            self._max_risk_pct = new_max_risk_pct
            self._daily_loss_pct = daily_loss_pct
            self._total_loss_pct = total_loss_pct

    # --- Output Methods ---

    def trading_allowed(self) -> bool:
        with self._lock:
            return self._trading_allowed

    def max_risk_pct(self) -> float:
        with self._lock:
            return self._max_risk_pct

    def daily_loss_pct(self) -> float:
        with self._lock:
            return self._daily_loss_pct

    def total_loss_pct(self) -> float:
        with self._lock:
            return self._total_loss_pct

    def remaining_daily_risk_pct(self) -> float:
        with self._lock:
            return self._remaining_daily_risk_pct

    def remaining_total_risk_pct(self) -> float:
        with self._lock:
            return self._remaining_total_risk_pct

if __name__ == "__main__":
    # Structural and logic test with mocks
    import unittest.mock as mock
    import time

    # Configure logging for test output
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # 1. Mock PositionTracker
    mock_tracker = mock.MagicMock()
    mock_tracker.get_open_positions.return_value = []
    mock_tracker.get_open_risk.return_value = 0.0

    # 2. Mock MetaTrader5
    with mock.patch("MetaTrader5.account_info") as mock_acc, \
         mock.patch("MetaTrader5.symbol_info_tick") as mock_tick, \
         mock.patch("MetaTrader5.symbol_select") as mock_select:

        mock_select.return_value = True

        # Scenario: Start of day, initial balance 10k
        mock_acc.return_value = mock.MagicMock(balance=10000.0)

        # Set server time to 2024-05-20
        mock_tick.return_value = mock.MagicMock(time=time.mktime(time.strptime("2024-05-20", "%Y-%m-%d")))

        dm = DrawdownManager(
            initial_balance=10000.0,
            position_tracker=mock_tracker,
            daily_limit_pct=0.03, # $300
            total_limit_pct=0.10, # $1000
            state_file="test_drawdown_state.json"
        )

        print(f"Initial: Allowed={dm.trading_allowed()}, MaxRisk={dm.max_risk_pct():.4%}")
        assert dm.trading_allowed() == True
        assert abs(dm.max_risk_pct() - 0.03) < 1e-6

        # Scenario: Open a position with $200 risk
        mock_tracker.get_open_risk.return_value = 200.0
        dm.check()
        print(f"Post-Position ($200 risk): Allowed={dm.trading_allowed()}, MaxRisk={dm.max_risk_pct():.4%}")
        # Daily risk remaining: 0.03 - (0 + 200/10000) = 0.03 - 0.02 = 0.01
        assert abs(dm.max_risk_pct() - 0.01) < 1e-6
        assert dm.trading_allowed() == True

        # Scenario: Position still open, balance drops to $9850 (closed $150 loss)
        mock_acc.return_value = mock.MagicMock(balance=9850.0)
        dm.check()
        print(f"Post-Loss ($150 loss + $200 risk): Allowed={dm.trading_allowed()}, MaxRisk={dm.max_risk_pct():.4%}")
        # current_balance = 9850
        # open_risk_pct = 200 / 9850 = 0.0203045
        # daily_loss_pct = (10000 - 9850) / 10000 = 0.015
        # total_committed_daily = 0.015 + 0.0203045 = 0.0353045
        # remaining_daily = 0.03 - 0.0353045 = -0.0053045
        assert dm.trading_allowed() == False
        assert dm.max_risk_pct() < 0

        # Scenario: Close position, balance remains $9850.
        mock_tracker.get_open_risk.return_value = 0.0
        dm.check()
        print(f"Post-Close ($150 loss, 0 risk): Allowed={dm.trading_allowed()}, MaxRisk={dm.max_risk_pct():.4%}")
        # remaining_daily = 0.03 - (0.015 + 0) = 0.015
        assert dm.trading_allowed() == True
        assert abs(dm.max_risk_pct() - 0.015) < 1e-6

        # Scenario: Day rolls over. Snapshot balance at $9850.
        mock_tick.return_value = mock.MagicMock(time=time.mktime(time.strptime("2024-05-21", "%Y-%m-%d")))
        dm.check()
        print(f"New Day (Start balance $9850): Allowed={dm.trading_allowed()}, MaxRisk={dm.max_risk_pct():.4%}")
        # daily_loss_pct = (9850 - 9850) / 9850 = 0
        # remaining_daily = 0.03 - 0 = 0.03
        # total_loss_pct = (10000 - 9850) / 10000 = 0.015
        # remaining_total = 0.10 - (0.015 + 0) = 0.085
        # max_risk = min(0.03, 0.085) = 0.03
        assert abs(dm.max_risk_pct() - 0.03) < 1e-6
        assert dm.start_of_day_balance == 9850.0

        # Cleanup
        if os.path.exists("test_drawdown_state.json"):
            os.remove("test_drawdown_state.json")
        if os.path.exists("test_drawdown_state.json.tmp"):
            os.remove("test_drawdown_state.json.tmp")

        print("DrawdownManager logic test passed.")
