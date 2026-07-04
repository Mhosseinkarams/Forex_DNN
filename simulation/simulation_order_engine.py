import os
import json
import logging
from simulation.simulation_environment import env

logger = logging.getLogger("SimOrderEngine")

class SimulationOrderEngine:
    """
    Drop-in replacement for SendOrder in backtest mode.
    """
    def __init__(
        self,
        simulation_broker,
        position_tracker,
        drawdown_manager,
        position_sizer,
        exit_manager,
        trading_journal,
        state_file: str = "sim_send_order_state.json"
    ):
        self.broker = simulation_broker
        self.pt = position_tracker
        self.dm = drawdown_manager
        self.ps = position_sizer
        self.em = exit_manager
        self.tj = trading_journal
        self.state_file = state_file

        self.ticket_categories = {}
        # We might not need state persistence for backtest if it runs in one go,
        # but let's keep the interface similar.

    def execute(
        self,
        symbol: str,
        direction: int,
        entry_price: float,
        sl_price: float,
        tp_level: int,
        stage: str,
        strategy: str,
        signal_category: str,
        signal_id: str,
        comment: str = "",
    ) -> dict:
        # 1. Fetch Price
        tick = env.symbol_info_tick(symbol)
        if tick is None:
            return self._failure("invalid_input", "No tick data", symbol, direction, signal_category)
        market_price = tick.ask if direction == 1 else tick.bid

        # 2. Drawdown Check
        if not self.dm.trading_allowed():
            return self._failure("drawdown_blocked", "Drawdown limit", symbol, direction, signal_category)

        # 3. Risk %
        risk_pct = self.dm.max_risk_pct()
        DEFAULT_RISK = {"standard": 0.01, "high_risk": 0.005, "reversal": 0.003}
        risk_pct = min(risk_pct, DEFAULT_RISK.get(signal_category, 0.01))

        # 4. Lot Sizing
        acc = env.get_account_info()
        balance = acc["balance"]

        sizing_res = self.ps.calculate_lot_size(symbol, market_price, sl_price, risk_pct, balance)
        if not sizing_res["success"]:
            return self._failure("sizing_failed", sizing_res["error"], symbol, direction, signal_category)

        lot_size = sizing_res["lot_size"]
        actual_risk_pct = sizing_res["risk_pct_actual"]

        # 5. Position Conflict Checks (Skip for now to keep it simple, or implement if needed)
        # Rule implementation would mirror SendOrder

        # 6. Execute via SimBroker
        R = abs(market_price - sl_price)
        tp_price = market_price + (1 if direction == 1 else -1) * tp_level * R

        ticket = self.broker.open_position(
            symbol=symbol,
            direction=direction,
            volume=lot_size,
            price=market_price,
            sl=sl_price,
            tp=tp_price,
            magic=100002 if strategy == "mm" else 100001,
            time_ts=tick.time,
            comment=comment
        )

        # 7. Post-Execution
        self.ticket_categories[ticket] = signal_category

        # Register with ExitManager
        self.em.register_position(
            ticket=ticket,
            entry_price=market_price,
            sl_price=sl_price,
            direction=direction,
            stage=stage,
            final_tp=tp_level,
            signal_id=signal_id
        )

        # Log to Journal
        self.tj.log_order_open(
            signal_id=signal_id,
            ticket=ticket,
            actual_entry=market_price,
            actual_sl=sl_price,
            actual_tp=tp_price,
            lot_size=lot_size,
            risk_pct=actual_risk_pct
        )

        return {
            "success": True,
            "reason": "ok",
            "ticket": ticket,
            "symbol": symbol,
            "direction": direction,
            "lot_size": lot_size,
            "entry_price": market_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "risk_pct": actual_risk_pct,
            "signal_category": signal_category,
            "error_detail": "",
        }

    def _failure(self, reason, detail, symbol, direction, category):
        return {
            "success": False,
            "reason": reason,
            "ticket": None,
            "symbol": symbol,
            "direction": direction,
            "lot_size": 0.0,
            "entry_price": 0.0,
            "sl_price": 0.0,
            "tp_price": 0.0,
            "risk_pct": 0.0,
            "signal_category": category,
            "error_detail": detail,
        }
