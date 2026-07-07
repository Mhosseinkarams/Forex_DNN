class SimulationAccount:
    def __init__(self, initial_balance: float = 10000.0, leverage: int = 100):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.equity = initial_balance
        self.margin_used = 0.0
        self.free_margin = initial_balance
        self.closed_profit = 0.0
        self.open_profit = 0.0
        
        self.max_balance = initial_balance
        self.max_equity = initial_balance
        self.max_drawdown_dollars = 0.0
        self.max_drawdown_pct = 0.0

    def update(self, open_positions_profit: float, margin_used: float):
        self.open_profit = open_positions_profit
        self.equity = self.balance + self.open_profit
        self.margin_used = margin_used
        self.free_margin = self.equity - self.margin_used
        
        if self.equity > self.max_equity:
            self.max_equity = self.equity
            
        dd_dollars = self.max_equity - self.equity
        dd_pct = (dd_dollars / self.max_equity) * 100 if self.max_equity > 0 else 0
        
        if dd_dollars > self.max_drawdown_dollars:
            self.max_drawdown_dollars = dd_dollars
        if dd_pct > self.max_drawdown_pct:
            self.max_drawdown_pct = dd_pct

    def apply_deal(self, profit: float):
        self.balance += profit
        self.closed_profit += profit
        if self.balance > self.max_balance:
            self.max_balance = self.balance

    def reset(self):
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.margin_used = 0.0
        self.free_margin = self.initial_balance
        self.closed_profit = 0.0
        self.open_profit = 0.0
        self.max_balance = self.initial_balance
        self.max_equity = self.initial_balance
        self.max_drawdown_dollars = 0.0
        self.max_drawdown_pct = 0.0
