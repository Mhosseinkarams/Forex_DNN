import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

class SimulationEnvironment:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SimulationEnvironment, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.mode = "live" # live, backtest
        self.clock = None
        self.account = None
        self.broker = None
        self.data_feed = None
        self._initialized = True

    def set_backtest_mode(self, clock, account, broker, data_feed):
        self.mode = "backtest"
        self.clock = clock
        self.account = account
        self.broker = broker
        self.data_feed = data_feed

    def set_live_mode(self):
        self.mode = "live"

    def get_now(self) -> datetime:
        if self.mode == "backtest" and self.clock:
            return self.clock.current_time()
        return datetime.now(timezone.utc)

    def get_account_info(self):
        if self.mode == "backtest" and self.account:
            return self.account.get_info()
        import MetaTrader5 as mt5
        acc = mt5.account_info()
        if acc is None: return None
        return {
            "balance": acc.balance,
            "equity": acc.equity,
            "margin": acc.margin,
            "margin_free": acc.margin_free,
            "profit": acc.profit,
        }

    def positions_get(self, ticket=None):
        if self.mode == "backtest" and self.broker:
            return self.broker.get_positions(ticket=ticket)
        import MetaTrader5 as mt5
        if ticket is not None:
            return mt5.positions_get(ticket=ticket)
        return mt5.positions_get()

    def history_deals_get(self, position=None):
        if self.mode == "backtest" and self.broker:
            return self.broker.get_history_deals(position_ticket=position)
        import MetaTrader5 as mt5
        return mt5.history_deals_get(position=position)

    def symbol_info(self, symbol):
        if self.mode == "backtest" and self.broker:
            return self.broker.get_symbol_info(symbol)
        import MetaTrader5 as mt5
        info = mt5.symbol_info(symbol)
        if info is None: return None
        # Return a dict-like object or the object itself
        return info

    def symbol_info_tick(self, symbol):
        if self.mode == "backtest" and self.data_feed:
            bar = self.data_feed.get_current_bar(symbol, "M1") # Use M1 for "ticks" in backtest
            if bar is None:
                # Fallback to any available timeframe
                for tf in self.data_feed.symbol_data.get(symbol, {}):
                    bar = self.data_feed.get_current_bar(symbol, tf)
                    if bar is not None: break

            if bar is not None:
                # Mock a tick object
                class MockTick:
                    def __init__(self, bid, ask, time_ts):
                        self.bid = bid
                        self.ask = ask
                        self.time = time_ts
                return MockTick(bar["Close"], bar["Close"], int(bar["Datetime"].timestamp()))

        import MetaTrader5 as mt5
        return mt5.symbol_info_tick(symbol)

# Global singleton
env = SimulationEnvironment()
