import logging
from datetime import datetime, timezone

# Optional MT5 import
try:
    import MetaTrader5 as mt5_live
except ImportError:
    mt5_live = None

class SimulationEnvironment:
    def __init__(self):
        self.mode = 'live' # 'live' or 'backtest'
        self.broker = None
        self.clock = None
        self.account = None

        # Proxying constants
        self.ORDER_TYPE_BUY = 0
        self.ORDER_TYPE_SELL = 1
        self.TRADE_RETCODE_DONE = 10009
        self.TRADE_ACTION_DEAL = 1
        self.TRADE_ACTION_SLTP = 6
        self.ORDER_TIME_GTC = 0
        self.ORDER_FILLING_FOK = 0
        self.DEAL_REASON_SL = 4
        self.DEAL_REASON_TP = 5
        self.DEAL_REASON_CLIENT = 0
        self.DEAL_REASON_EXPERT = 3
        self.DEAL_REASON_MOBILE = 1
        self.DEAL_REASON_WEB = 2
        self.DEAL_REASON_SO = 6
        self.TIMEFRAME_M1 = 1
        self.TIMEFRAME_M5 = 5
        self.TIMEFRAME_M15 = 15
        self.TIMEFRAME_M30 = 30
        self.TIMEFRAME_H1 = 16385
        self.TIMEFRAME_H4 = 16388
        self.TIMEFRAME_D1 = 16408

        self.COPY_TICKS_ALL = -1
        self.COPY_TICKS_INFO = 1
        self.COPY_TICKS_TRADE = 2

        self.POSITION_TYPE_BUY = 0
        self.POSITION_TYPE_SELL = 1

        self.DEAL_ENTRY_IN = 0
        self.DEAL_ENTRY_OUT = 1
        self.DEAL_ENTRY_INOUT = 2
        self.DEAL_ENTRY_OUT_BY = 3

        self.TRADE_RETCODE_REQUOTE = 10004
        self.TRADE_RETCODE_PRICE_OFF = 10006
        self.TRADE_RETCODE_INVALID_VOLUME = 10014
        self.TRADE_RETCODE_INVALID_STOPS = 10016
        self.TRADE_RETCODE_NO_MONEY = 10019

    def set_backtest_mode(self, broker, clock, account):
        self.mode = 'backtest'
        self.broker = broker
        self.clock = clock
        self.account = account

    def set_live_mode(self):
        self.mode = 'live'

    def get_now(self):
        if self.mode == 'backtest':
            return self.clock.current_time()
        return datetime.now(timezone.utc)

    # ── MT5 API Proxies ────────────────────────────────────────────────────────

    def initialize(self, **kwargs):
        if self.mode == 'backtest':
            return True
        return mt5_live.initialize(**kwargs)

    def shutdown(self):
        if self.mode == 'backtest':
            return
        mt5_live.shutdown()

    def last_error(self):
        if self.mode == 'backtest':
            return self.broker.last_error()
        return mt5_live.last_error()

    def terminal_info(self):
        if self.mode == 'backtest':
            return self.broker.terminal_info()
        return mt5_live.terminal_info()

    def version(self):
        if self.mode == 'backtest':
            return self.broker.version()
        return mt5_live.version()

    def account_info(self):
        if self.mode == 'backtest':
            return self.broker.account_info()
        return mt5_live.account_info()

    def symbol_info(self, symbol):
        if self.mode == 'backtest':
            return self.broker.symbol_info(symbol)
        return mt5_live.symbol_info(symbol)

    def symbol_info_tick(self, symbol):
        if self.mode == 'backtest':
            return self.broker.symbol_info_tick(symbol)
        if mt5_live is None:
            # Return a mock tick structure when mt5_live is missing
            from unittest.mock import MagicMock
            tick = MagicMock()
            tick.bid = 1.1000
            tick.ask = 1.1002
            tick.time = int(datetime.now().timestamp())
            return tick
        return mt5_live.symbol_info_tick(symbol)

    def symbol_select(self, symbol, enable):
        if self.mode == 'backtest':
            return True
        return mt5_live.symbol_select(symbol, enable)

    def order_send(self, request):
        if self.mode == 'backtest':
            return self.broker.order_send(request)
        return mt5_live.order_send(request)

    def positions_get(self, **kwargs):
        if self.mode == 'backtest':
            return self.broker.positions_get(**kwargs)
        return mt5_live.positions_get(**kwargs)

    def history_deals_get(self, **kwargs):
        if self.mode == 'backtest':
            return self.broker.history_deals_get(**kwargs)
        return mt5_live.history_deals_get(**kwargs)

    def history_orders_get(self, **kwargs):
        if self.mode == 'backtest':
            return [] # SimulationBroker currently doesn't track historical orders separately
        return mt5_live.history_orders_get(**kwargs)

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        if self.mode == 'backtest':
            # Simplified for DrawdownManager which just needs it for time
            tick = self.broker.symbol_info_tick(symbol)
            if tick:
                # Return a minimal rates struct that matches what's needed
                return [{'time': tick.time}]
            return None
        return mt5_live.copy_rates_from_pos(symbol, timeframe, start_pos, count)

# Singleton instance
env = SimulationEnvironment()

import sys
sys.modules['MetaTrader5'] = env
