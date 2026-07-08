import logging
from simulation.simulation_environment import env

logger = logging.getLogger("SimulationOrderEngine")

class SimulationOrderEngine:
    def __init__(
        self,
        position_manager,
        position_tracker,
        drawdown_manager,
        position_sizer,
        exit_manager,
        trading_journal
    ):
        self.pm = position_manager
        self.pt = position_tracker
        self.dm = drawdown_manager
        self.ps = position_sizer
        self.em = exit_manager
        self.tj = trading_journal

    def execute(
        self,
        symbol: str,
        direction: int,
        entry_price: float,
        sl_price: float,
        exit_profile: str,
        strategy: str,
        signal_category: str,
        signal_id: str,
        comment: str = ""
    ) -> dict:
        tick = env.symbol_info_tick(symbol)
        if tick is None:
            return {"success": False, "reason": "no_tick", "error_detail": f"No tick for {symbol}"}

        market_price = tick.ask if direction == 1 else tick.bid

        if not self.dm.trading_allowed():
            return {"success": False, "reason": "drawdown_blocked"}

        risk_pct = self.dm.max_risk_pct()
        from Collecting_Data.position_lifecycle import EXIT_PROFILES
        profile_cfg = EXIT_PROFILES.get(exit_profile)
        default_risk = profile_cfg.get("default_risk", 0.01) if profile_cfg else 0.01
        risk_pct = min(risk_pct, default_risk)

        acc = env.account_info()
        balance = acc.balance

        sizing_res = self.ps.calculate_lot_size(symbol, market_price, sl_price, risk_pct, balance)
        if not sizing_res["success"]:
            return {"success": False, "reason": "sizing_failed", "error_detail": sizing_res["error"]}

        lot_size = sizing_res["lot_size"]
        actual_risk_pct = sizing_res["risk_pct_actual"]

        tp_level = profile_cfg["broker_tp_level"] if profile_cfg else 1
        R = abs(market_price - sl_price)
        tp_price = market_price + (1 if direction == 1 else -1) * tp_level * R

        open_res = self.pm.open_position(symbol, direction, lot_size, sl_price, tp_price, strategy, comment)

        if open_res["success"]:
            ticket = open_res["ticket"]
            actual_entry = open_res["entry_price"]
            actual_sl = open_res["sl_price"]
            actual_tp = open_res["tp_price"]

            self.em.register_position(
                ticket=ticket,
                entry_price=actual_entry,
                sl_price=actual_sl,
                direction=direction,
                exit_profile=exit_profile,
                signal_id=signal_id
            )

            self.tj.log_order_open(
                signal_id=signal_id,
                ticket=ticket,
                actual_entry=actual_entry,
                actual_sl=actual_sl,
                actual_tp=actual_tp,
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
                "entry_price": actual_entry,
                "sl_price": actual_sl,
                "tp_price": actual_tp,
                "risk_pct": actual_risk_pct,
                "signal_category": signal_category
            }

        return {"success": False, "reason": "open_failed"}
