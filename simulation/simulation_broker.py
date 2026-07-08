import logging
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

logger = logging.getLogger("SimulationBroker")

# Mock MT5 Constants
TRADE_ACTION_DEAL = 1
TRADE_ACTION_SLTP = 6
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
ORDER_TIME_GTC = 0
ORDER_FILLING_FOK = 0
TRADE_RETCODE_DONE = 10009

DEAL_ENTRY_IN = 0
DEAL_ENTRY_OUT = 1

DEAL_REASON_CLIENT = 0
DEAL_REASON_EXPERT = 3
DEAL_REASON_SL = 4
DEAL_REASON_TP = 5

@dataclass
class SimPosition:
    def _asdict(self):
        return asdict(self)

    ticket: int
    symbol: str
    magic: int
    type: int
    volume: float
    price_open: float
    sl: float
    tp: float
    price_current: float
    profit: float
    time: int # timestamp
    comment: str = ""
    reason: int = DEAL_REASON_EXPERT

@dataclass
class SimDeal:
    def _asdict(self):
        return asdict(self)

    ticket: int
    order: int
    symbol: str
    type: int
    entry: int # 0=in, 1=out
    magic: int
    price: float
    volume: float
    profit: float
    time: int = 0
    time_msc: int = 0
    commission: float = 0.0
    swap: float = 0.0
    reason: int = DEAL_REASON_EXPERT
    comment: str = ""
    position_id: int = 0

class SimulationBroker:
    def __init__(self, account, clock):
        self.account = account
        self.clock = clock
        self.positions = {} # ticket -> SimPosition
        self.deals = [] # list of SimDeal
        self.next_ticket = 1000000
        self.current_prices = {} # symbol -> {"bid": float, "ask": float}
        self.symbol_info_dict = {} # symbol -> info dict
        self._last_deal_time_msc = 0

        # MT5 Constants equivalents
        self.ORDER_TYPE_BUY = ORDER_TYPE_BUY
        self.ORDER_TYPE_SELL = ORDER_TYPE_SELL
        self.TRADE_RETCODE_DONE = TRADE_RETCODE_DONE
        self.DEAL_REASON_SL = DEAL_REASON_SL
        self.DEAL_REASON_TP = DEAL_REASON_TP
        self.DEAL_REASON_CLIENT = DEAL_REASON_CLIENT
        self.DEAL_REASON_EXPERT = DEAL_REASON_EXPERT
        self.ORDER_TIME_GTC = ORDER_TIME_GTC

    def set_symbol_info(self, symbol: str, info: dict):
        self.symbol_info_dict[symbol] = info

    def update_market_price(self, symbol: str, bid: float, ask: float):
        self.current_prices[symbol] = {"bid": bid, "ask": ask}

        tickets_to_close = []
        for ticket, pos in self.positions.items():
            if pos.symbol == symbol:
                pos.price_current = bid if pos.type == ORDER_TYPE_BUY else ask

                info = self.symbol_info_dict.get(symbol, {})
                tick_value = info.get("trade_tick_value")
                tick_size = info.get("trade_tick_size")

                if tick_value and tick_size:
                    ticks = (pos.price_current - pos.price_open) / tick_size
                    if pos.type == ORDER_TYPE_SELL:
                        ticks = -ticks
                    pos.profit = ticks * pos.volume * tick_value
                else:
                    contract_size = info.get("trade_contract_size", 100000)
                    if pos.type == ORDER_TYPE_BUY:
                        pos.profit = (pos.price_current - pos.price_open) * pos.volume * contract_size
                    else:
                        pos.profit = (pos.price_open - pos.price_current) * pos.volume * contract_size

                if pos.sl != 0:
                    if (pos.type == ORDER_TYPE_BUY and pos.price_current <= pos.sl - 1e-9) or \
                       (pos.type == ORDER_TYPE_SELL and pos.price_current >= pos.sl + 1e-9):
                        logger.info(f"SIM: SL hit for ticket {ticket} at {pos.sl}")
                        tickets_to_close.append((ticket, pos.sl, DEAL_REASON_SL))

                if pos.tp != 0:
                    if (pos.type == ORDER_TYPE_BUY and pos.price_current >= pos.tp - 1e-9) or \
                       (pos.type == ORDER_TYPE_SELL and pos.price_current <= pos.tp + 1e-9):
                        logger.info(f"SIM: TP hit for ticket {ticket} at {pos.tp}")
                        tickets_to_close.append((ticket, pos.tp, DEAL_REASON_TP))

        for ticket, price, reason in tickets_to_close:
            self._close_sim_position(ticket, price, reason=reason)

        margin_used = self._calculate_total_margin()
        self.account.update(sum(p.profit for p in self.positions.values()), margin_used)

    def order_send(self, request: dict):
        action = request.get("action")
        if action == TRADE_ACTION_DEAL:
            if "position" in request: # Close order
                return self._handle_close_request(request)
            else: # Open order
                return self._handle_open_request(request)
        elif action == TRADE_ACTION_SLTP:
            return self._handle_modify_request(request)
        return None

    def _get_unique_time_msc(self):
        now_msc = int(self.clock.current_time().timestamp() * 1000)
        if now_msc <= self._last_deal_time_msc:
            now_msc = self._last_deal_time_msc + 1
        self._last_deal_time_msc = now_msc
        return now_msc

    def _handle_open_request(self, request: dict):
        symbol = request["symbol"]
        order_type = request["type"]
        volume = request["volume"]
        price = request["price"]
        sl = request.get("sl", 0.0)
        tp = request.get("tp", 0.0)
        magic = request.get("magic", 0)
        comment = request.get("comment", "")

        ticket = self.next_ticket
        self.next_ticket += 1

        now_msc = self._get_unique_time_msc()

        pos = SimPosition(
            ticket=ticket,
            symbol=symbol,
            magic=magic,
            type=order_type,
            volume=volume,
            price_open=price,
            sl=sl,
            tp=tp,
            price_current=price,
            profit=0.0,
            time=now_msc // 1000,
            comment=comment
        )
        self.positions[ticket] = pos

        deal = SimDeal(
            ticket=self.next_ticket,
            order=ticket,
            symbol=symbol,
            type=order_type,
            entry=DEAL_ENTRY_IN,
            magic=magic,
            price=price,
            volume=volume,
            profit=0.0,
            time=now_msc // 1000,
            time_msc=now_msc,
            position_id=ticket,
            comment=comment
        )
        self.next_ticket += 1
        self.deals.append(deal)

        class Result:
            def __init__(self, retcode, order, price, comment):
                self.retcode = retcode
                self.order = order
                self.price = price
                self.comment = comment

        return Result(TRADE_RETCODE_DONE, ticket, price, "Done")

    def _handle_close_request(self, request: dict):
        ticket = request["position"]
        volume = request["volume"]
        price = request["price"]

        res = self._close_sim_position(ticket, price, volume=volume)

        class Result:
            def __init__(self, retcode, order, price, comment):
                self.retcode = retcode
                self.order = order
                self.price = price
                self.comment = comment

        if res:
            return Result(TRADE_RETCODE_DONE, ticket, price, "Done")
        else:
            return Result(10001, ticket, 0.0, "Failed")

    def _handle_modify_request(self, request: dict):
        ticket = request["position"]
        sl = request.get("sl", 0.0)
        tp = request.get("tp", 0.0)

        if ticket in self.positions:
            self.positions[ticket].sl = sl
            self.positions[ticket].tp = tp

            class Result:
                def __init__(self, retcode):
                    self.retcode = retcode
            return Result(TRADE_RETCODE_DONE)
        return None

    def _close_sim_position(self, ticket: int, price: float, volume: float = None, reason: int = DEAL_REASON_EXPERT):
        if ticket not in self.positions:
            return False

        pos = self.positions[ticket]
        close_vol = volume if volume is not None else pos.volume

        if close_vol > pos.volume + 1e-9:
            return False

        now_msc = self._get_unique_time_msc()
        info = self.symbol_info_dict.get(pos.symbol, {})
        tick_value = info.get("trade_tick_value")
        tick_size = info.get("trade_tick_size")

        if tick_value and tick_size:
            ticks = (price - pos.price_open) / tick_size
            if pos.type == ORDER_TYPE_SELL:
                ticks = -ticks
            deal_profit = ticks * close_vol * tick_value
        else:
            contract_size = info.get("trade_contract_size", 100000)
            if pos.type == ORDER_TYPE_BUY:
                deal_profit = (price - pos.price_open) * close_vol * contract_size
            else:
                deal_profit = (pos.price_open - price) * close_vol * contract_size

        exit_comment = "tp" if reason == DEAL_REASON_TP else "sl" if reason == DEAL_REASON_SL else pos.comment
        deal = SimDeal(
            ticket=self.next_ticket,
            order=ticket,
            symbol=pos.symbol,
            type=ORDER_TYPE_SELL if pos.type == ORDER_TYPE_BUY else ORDER_TYPE_BUY,
            entry=DEAL_ENTRY_OUT,
            magic=pos.magic,
            price=price,
            volume=close_vol,
            profit=deal_profit,
            time=now_msc // 1000,
            time_msc=now_msc,
            position_id=ticket,
            reason=reason,
            comment=exit_comment
        )
        self.next_ticket += 1
        self.deals.append(deal)

        self.account.apply_deal(deal_profit)

        if abs(pos.volume - close_vol) < 1e-9:
            del self.positions[ticket]
        else:
            pos.volume -= close_vol
            # Update remaining position profit
            if tick_value and tick_size:
                ticks = (pos.price_current - pos.price_open) / tick_size
                if pos.type == ORDER_TYPE_SELL:
                    ticks = -ticks
                pos.profit = ticks * pos.volume * tick_value
            else:
                c_size = info.get("trade_contract_size", 100000)
                if pos.type == ORDER_TYPE_BUY:
                    pos.profit = (pos.price_current - pos.price_open) * pos.volume * c_size
                else:
                    pos.profit = (pos.price_open - pos.price_current) * pos.volume * c_size

        return True

    def _calculate_total_margin(self):
        total_margin = 0.0
        for pos in self.positions.values():
            contract_size = self.symbol_info_dict.get(pos.symbol, {}).get("trade_contract_size", 100000)
            total_margin += (pos.volume * contract_size) / self.account.leverage
        return total_margin

    def positions_get(self, symbol=None, ticket=None, magic=None):
        res = list(self.positions.values())
        if symbol:
            res = [p for p in res if p.symbol == symbol]
        if ticket:
            res = [p for p in res if p.ticket == ticket]
        if magic:
            res = [p for p in res if p.magic == magic]
        return tuple(res)

    def history_deals_get(self, position=None):
        if position:
            return sorted([d for d in self.deals if d.position_id == position], key=lambda d: d.time_msc)
        return sorted(self.deals, key=lambda d: d.time_msc)

    def symbol_info_tick(self, symbol: str):
        prices = self.current_prices.get(symbol)

        class Tick:
            def __init__(self, bid, ask, time):
                self.bid = bid
                self.ask = ask
                self.time = time

        if not prices:
            # Fallback for DrawdownManager during init if price not yet set
            return Tick(0.0, 0.0, int(self.clock.current_time().timestamp()))

        return Tick(prices["bid"], prices["ask"], int(self.clock.current_time().timestamp()))

    def account_info(self):
        class Info:
            def __init__(self, balance, equity, margin_free):
                self.balance = balance
                self.equity = equity
                self.margin_free = margin_free
        return Info(self.account.balance, self.account.equity, self.account.free_margin)

    def symbol_info(self, symbol: str):
        info = self.symbol_info_dict.get(symbol)
        if not info:
            return None

        class Info:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)
        return Info(info)

    def terminal_info(self):
        class Info:
            def __init__(self):
                self.name = "Simulation"
                self.connected = True
        return Info()

    def version(self):
        return (500, "Sim")

    def last_error(self):
        return (0, "Success")
