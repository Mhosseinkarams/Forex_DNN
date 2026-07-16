import os
import json
import unittest
from unittest import mock

from Trade_Execution.drawdown import DrawdownManager
from Simulation.simulation_environment import env as mt5

class TestSymbolAndDrawdown(unittest.TestCase):
    def setUp(self):
        self.state_file = "test_drawdown_state.json"
        self._cleanup()

    def tearDown(self):
        self._cleanup()

    def _cleanup(self):
        for f in [self.state_file, self.state_file + ".tmp"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass

    def test_symbol_preservation_cli(self):
        """
        Verify that user-supplied symbols containing lowercase suffixes
        and custom broker naming conventions are preserved exactly
        without uppercase normalization.
        """
        symbols_arg = "XAUUSD_o,EURUSD.r,GBPUSD.m,YM"
        parsed_symbols = [s.strip() for s in symbols_arg.split(",")]

        # Ensure lowercase suffixes are preserved exactly
        self.assertEqual(parsed_symbols, ["XAUUSD_o", "EURUSD.r", "GBPUSD.m", "YM"])

    def test_drawdown_manager_state_restoration(self):
        """
        Verify that DrawdownManager restores state correctly.
        """
        # Save a mock state
        state_data = {
            "start_of_day_balance": 10000.0,
            "snapshot_date": "2026-07-10",
            "account_id": 123456
        }
        with open(self.state_file, "w") as f:
            json.dump(state_data, f)

        mock_tracker = mock.MagicMock()
        mock_tracker.get_open_positions.return_value = []
        mock_tracker.get_open_risk.return_value = 0.0

        # Mock MT5 account info to match restored state
        mock_acc_info = mock.MagicMock()
        mock_acc_info.balance = 10000.0
        mock_acc_info.login = 123456

        mock_tick = mock.MagicMock()
        mock_tick.time = 1783684800  # July 10, 2026

        with mock.patch("Simulation.simulation_environment.SimulationEnvironment.account_info", return_value=mock_acc_info), \
             mock.patch("Simulation.simulation_environment.SimulationEnvironment.symbol_info_tick", return_value=mock_tick), \
             mock.patch("Simulation.simulation_environment.SimulationEnvironment.symbol_select", return_value=True):

            dm = DrawdownManager(
                initial_balance=10000.0,
                position_tracker=mock_tracker,
                state_file=self.state_file,
                symbols=["EURUSD.r"]
            )

            # Restored values should match saved state, no reset triggered
            self.assertEqual(dm.start_of_day_balance, 10000.0)
            self.assertEqual(dm.snapshot_date, "2026-07-10")
            self.assertEqual(dm.saved_account_id, 123456)

    def test_drawdown_manager_changed_account_id(self):
        """
        Verify that a changed account ID triggers a reset/reinitialization.
        """
        # Saved state with ID 123456 and balance 10000.0
        state_data = {
            "start_of_day_balance": 10000.0,
            "snapshot_date": "2026-07-10",
            "account_id": 123456
        }
        with open(self.state_file, "w") as f:
            json.dump(state_data, f)

        mock_tracker = mock.MagicMock()
        mock_tracker.get_open_positions.return_value = []
        mock_tracker.get_open_risk.return_value = 0.0

        # Mock MT5 account info with a DIFFERENT login/ID (99999) and balance 5000.0
        mock_acc_info = mock.MagicMock()
        mock_acc_info.balance = 5000.0
        mock_acc_info.login = 99999

        mock_tick = mock.MagicMock()
        mock_tick.time = 1783684800

        with mock.patch("Simulation.simulation_environment.SimulationEnvironment.account_info", return_value=mock_acc_info), \
             mock.patch("Simulation.simulation_environment.SimulationEnvironment.symbol_info_tick", return_value=mock_tick), \
             mock.patch("Simulation.simulation_environment.SimulationEnvironment.symbol_select", return_value=True):

            dm = DrawdownManager(
                initial_balance=10000.0,
                position_tracker=mock_tracker,
                state_file=self.state_file
            )

            # DrawdownManager should have run check() inside __init__ which invokes sync_live_state()
            # This should have triggered a reset to the new account's balance (5000.0) and new account ID (99999)
            self.assertEqual(dm.saved_account_id, 99999)
            self.assertEqual(dm.initial_balance, 5000.0)
            self.assertEqual(dm.start_of_day_balance, 5000.0)
            self.assertTrue(dm.trading_allowed())

    def test_drawdown_manager_changed_balance_startup(self):
        """
        Verify that a changed account balance on startup triggers a reset/reinitialization.
        """
        # Saved state with balance 10000.0, same account ID
        state_data = {
            "start_of_day_balance": 10000.0,
            "snapshot_date": "2026-07-10",
            "account_id": 123456
        }
        with open(self.state_file, "w") as f:
            json.dump(state_data, f)

        mock_tracker = mock.MagicMock()
        mock_tracker.get_open_positions.return_value = []
        mock_tracker.get_open_risk.return_value = 0.0

        # Mock MT5 account info with SAME login/ID but DIFFERENT balance 5000.0
        mock_acc_info = mock.MagicMock()
        mock_acc_info.balance = 5000.0
        mock_acc_info.login = 123456

        mock_tick = mock.MagicMock()
        mock_tick.time = 1783684800

        with mock.patch("Simulation.simulation_environment.SimulationEnvironment.account_info", return_value=mock_acc_info), \
             mock.patch("Simulation.simulation_environment.SimulationEnvironment.symbol_info_tick", return_value=mock_tick), \
             mock.patch("Simulation.simulation_environment.SimulationEnvironment.symbol_select", return_value=True):

            dm = DrawdownManager(
                initial_balance=10000.0,
                position_tracker=mock_tracker,
                state_file=self.state_file
            )

            # Difference in balance at startup triggers a reset to prevent spurious violations
            self.assertEqual(dm.saved_account_id, 123456)
            self.assertEqual(dm.initial_balance, 5000.0)
            self.assertEqual(dm.start_of_day_balance, 5000.0)
            self.assertTrue(dm.trading_allowed())

    def test_live_account_synchronized_before_check(self):
        """
        Verify that the live account state is synchronized before drawdown checks.
        """
        mock_tracker = mock.MagicMock()
        mock_tracker.get_open_positions.return_value = []
        mock_tracker.get_open_risk.return_value = 0.0

        # Initial account setup
        mock_acc_info = mock.MagicMock()
        mock_acc_info.balance = 10000.0
        mock_acc_info.login = 123456

        mock_tick = mock.MagicMock()
        mock_tick.time = 1783684800

        with mock.patch("Simulation.simulation_environment.SimulationEnvironment.account_info", return_value=mock_acc_info) as mock_acc_call, \
             mock.patch("Simulation.simulation_environment.SimulationEnvironment.symbol_info_tick", return_value=mock_tick), \
             mock.patch("Simulation.simulation_environment.SimulationEnvironment.symbol_select", return_value=True):

            dm = DrawdownManager(
                initial_balance=10000.0,
                position_tracker=mock_tracker,
                state_file=self.state_file
            )

            # Clear calls from init check
            mock_acc_call.reset_mock()

            # Now update the account balance in MT5 during runtime
            mock_acc_info.balance = 9800.0

            # Execute check()
            dm.check()

            # Verify account_info was queried during check()
            mock_acc_call.assert_called()

            # Check values are computed against new synchronized balance
            self.assertEqual(dm.daily_loss_pct(), 0.02)
