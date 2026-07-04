import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger("SimulationBroker")

@dataclass
class SimPosition:
    ticket: int
    symbol: str
    magic: int
    direction: int # 1 = BUY, -1 = SELL
    volume: float
    price_open: float
    sl: float
    tp: float
    time: float # timestamp
    price_current: float = 0.0
    floating_pnl: float = 0.0
    comment: str = ""

@dataclass
class SimDeal:
    ticket: int # deal id
    order: int
    position: int # position ticket
    time: int # timestamp
    type: int # 0=BUY, 1=SELL
    entry: int # 0=IN, 1=OUT
    symbol: str
    volume: float
    price: float
    profit: float
    magic: int
    reason: int
    comment: str = ""

class SimulationBroker:
    """
    Replaces MT5 execution for backtesting.
    """
    def __init__(self, account):
        self.account = account
        self.positions: Dict[int, SimPosition] = {}
        self.deals: List[SimDeal] = []
        self.history_positions: Dict[int, SimPosition] = {}

        self.next_ticket = 1000001
        self.next_order = 2000001
        self.next_deal = 3000001

        # Symbol info cache for contract size, point, etc.
        self.symbol_info_cache = {}

    def set_symbol_info(self, symbol, info):
        self.symbol_info_cache[symbol] = info

    def get_symbol_info(self, symbol):
        return self.symbol_info_cache.get(symbol)

    def open_position(self, symbol, direction, volume, price, sl, tp, magic, time_ts, comment=""):
        ticket = self.next_ticket
        self.next_ticket += 1

        pos = SimPosition(
            ticket=ticket, symbol=symbol, magic=magic, direction=direction,
            volume=volume, price_open=price, sl=sl, tp=tp, time=time_ts,
            price_current=price, floating_pnl=0.0, comment=comment
        )
        self.positions[ticket] = pos

        deal = SimDeal(
            ticket=self.next_deal, order=self.next_order, position=ticket,
            time=int(time_ts), type=0 if direction == 1 else 1, entry=0,
            symbol=symbol, volume=volume, price=price, profit=0.0,
            magic=magic, reason=3, comment=comment
        )
        self.next_deal += 1
        self.next_order += 1
        self.deals.append(deal)
        logger.info(f"SimBroker: Opened {symbol} {direction} Lot:{volume} Ticket:{ticket} Price:{price}")
        return ticket

    def close_position(self, ticket, volume, price, time_ts, reason=3):
        if ticket not in self.positions: return False
        pos = self.positions[ticket]
        close_vol = volume if volume is not None else pos.volume

        info = self.get_symbol_info(pos.symbol)
        contract_size = info["trade_contract_size"] if (info and hasattr(info, 'trade_contract_size')) else info.get('trade_contract_size', 100000) if isinstance(info, dict) else 100000

        profit = (price - pos.price_open) * pos.direction * close_vol * contract_size

        deal = SimDeal(
            ticket=self.next_deal, order=self.next_order, position=ticket,
            time=int(time_ts), type=1 if pos.direction == 1 else 0, entry=1,
            symbol=pos.symbol, volume=close_vol, price=price, profit=profit,
            magic=pos.magic, reason=reason, comment="Close"
        )
        self.next_deal += 1
        self.next_order += 1
        self.deals.append(deal)
        self.account.apply_deal(profit)

        if abs(close_vol - pos.volume) < 1e-8:
            self.history_positions[ticket] = self.positions.pop(ticket)
        else:
            pos.volume -= close_vol

        logger.info(f"SimBroker: Closed {pos.symbol} Ticket:{ticket} Vol:{close_vol} Price:{price} Profit:{profit:.2f} Reason:{reason}")
        return True

    def modify_position(self, ticket, sl, tp):
        if ticket in self.positions:
            if sl is not None:
                self.positions[ticket].sl = sl
            if tp is not None:
                self.positions[ticket].tp = tp
            return True
        return False

    def get_positions(self, ticket=None):
        if ticket is not None:
            if ticket in self.positions: return [self.positions[ticket]]
            return []
        return list(self.positions.values())

    def get_history_deals(self, position_ticket):
        return [d for d in self.deals if d.position == position_ticket]

    def update_market_price(self, symbol, bid, ask, time_ts):
        """
        Updates current prices for all open positions and checks for SL/TP.
        """
        contract_size_default = 100000

        # We need a copy because we might modify the dict during iteration
        for ticket in list(self.positions.keys()):
            pos = self.positions[ticket]
            if pos.symbol != symbol: continue

            # 1. Update Current Price & Floating PnL
            pos.price_current = ask if pos.direction == 1 else bid

            info = self.get_symbol_info(pos.symbol)
            contract_size = info["trade_contract_size"] if (info and hasattr(info, 'trade_contract_size')) else info.get('trade_contract_size', contract_size_default) if isinstance(info, dict) else contract_size_default

            pos.floating_pnl = (pos.price_current - pos.price_open) * pos.direction * pos.volume * contract_size

            # 2. Check SL
            if pos.sl != 0:
                is_sl_hit = (bid <= pos.sl) if pos.direction == 1 else (ask >= pos.sl)
                if is_sl_hit:
                    logger.info(f"SimBroker: SL Hit for Ticket {ticket} at {pos.sl}")
                    self.close_position(ticket, None, pos.sl, time_ts, reason=4) # DEAL_REASON_SL
                    continue

            # 3. Check TP
            if pos.tp != 0:
                is_tp_hit = (bid >= pos.tp) if pos.direction == 1 else (ask <= pos.tp)
                if is_tp_hit:
                    logger.info(f"SimBroker: TP Hit for Ticket {ticket} at {pos.tp}")
                    self.close_position(ticket, None, pos.tp, time_ts, reason=5) # DEAL_REASON_TP
                    continue

        # Update account overall open pnl
        total_open_pnl = sum(p.floating_pnl for p in self.positions.values())
        # Margin used could also be calculated here if we wanted
        self.account.update(open_pnl=total_open_pnl, margin_used=0.0)
