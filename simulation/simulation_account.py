import logging

logger = logging.getLogger("SimulationAccount")

class SimulationAccount:
    def __init__(self, initial_balance: float = 10000.0, leverage: int = 100):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage

        self.equity = initial_balance
        self.margin_used = 0.0
        self.free_margin = initial_balance

        self.open_pnl = 0.0
        self.closed_pnl = 0.0

        self.max_drawdown_dollars = 0.0
        self.max_drawdown_pct = 0.0
        self.peak_balance = initial_balance

    def update(self, open_pnl: float, margin_used: float):
        self.open_pnl = open_pnl
        self.equity = self.balance + open_pnl
        self.margin_used = margin_used
        self.free_margin = self.equity - margin_used

        if self.equity > self.peak_balance:
            self.peak_balance = self.equity

        dd_dollars = self.peak_balance - self.equity
        dd_pct = (dd_dollars / self.peak_balance) * 100 if self.peak_balance > 0 else 0

        if dd_dollars > self.max_drawdown_dollars:
            self.max_drawdown_dollars = dd_dollars
        if dd_pct > self.max_drawdown_pct:
            self.max_drawdown_pct = dd_pct

    def apply_deal(self, profit: float):
        self.balance += profit
        self.closed_pnl += profit
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

    def get_info(self):
        return {
            "balance": self.balance,
            "equity": self.equity,
            "margin": self.margin_used,
            "margin_free": self.free_margin,
            "profit": self.open_pnl,
        }
