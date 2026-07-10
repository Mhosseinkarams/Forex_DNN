import logging
from typing import Dict, Any, Optional
from Market_Data_Pipeline.structure_graph import MarketStructureGraph

logger = logging.getLogger("TradeLocationEngine")

class TradeLocationEngine:
    """
    Purpose:
        Responsible for determining Entry candidate, Stop Loss, Take Profit, Invalidation level,
        and Reward/Risk using ONLY structural information from MarketStructureGraph.
        Avoids fixed pip multipliers or raw ATR offsets where structural levels are present.
    """
    def __init__(self, fallback_atr_mult: float = 2.0, min_rr: float = 1.0):
        self.fallback_atr_mult = fallback_atr_mult
        self.min_rr = min_rr

    def get_trade_levels(
        self,
        msg: MarketStructureGraph,
        direction: int,
        entry_price: float,
        exit_profile: str = "standard"
    ) -> Dict[str, Any]:
        """
        Determine entry, stop loss, take profit, invalidation level, and RR.
        """
        sl_price = 0.0
        tp_price = 0.0

        atr_val = msg.atr if msg.atr > 0 else 0.0001

        if direction == 1:  # BUY
            # Stop Loss: Select best stop using: Protected Low -> Nearest Demand Lower -> Nearest Swing Low
            candidates = []
            if msg.protected_low:
                candidates.append(msg.protected_low.price)

            # Nearest demand
            nearest_demand = msg.get_nearest_demand(entry_price)
            if nearest_demand:
                candidates.append(nearest_demand.lower)

            # Swing lows
            valid_swings = [s.price for s in msg.swing_lows if s.price < entry_price]
            if valid_swings:
                candidates.append(max(valid_swings))  # closest swing low

            if candidates:
                # Select the best structural stop (lowest of candidate levels to be safe, but below entry)
                sl_price = min(candidates)
            else:
                # Fallback to ATR-based stop
                sl_price = entry_price - (atr_val * self.fallback_atr_mult)

            # Take Profit: Nearest supply zone lower, or nearest swing high/protected high
            tp_candidates = []
            nearest_supply = msg.get_nearest_supply(entry_price)
            if nearest_supply:
                tp_candidates.append(nearest_supply.lower)
            if msg.protected_high and msg.protected_high.price > entry_price:
                tp_candidates.append(msg.protected_high.price)
            valid_high_swings = [s.price for s in msg.swing_highs if s.price > entry_price]
            if valid_high_swings:
                tp_candidates.append(min(valid_high_swings))

            if tp_candidates:
                tp_price = min(tp_candidates)
            else:
                # Fallback to default TP ratio (e.g. 2:1 RR)
                sl_dist = entry_price - sl_price
                tp_price = entry_price + (sl_dist * 2.0)

            # Invalidation is slightly below the SL price
            invalidation_level = sl_price - (atr_val * 0.1)

        else:  # SELL
            # Stop Loss: Protected High -> Nearest Supply Upper -> Nearest Swing High
            candidates = []
            if msg.protected_high:
                candidates.append(msg.protected_high.price)

            # Nearest supply
            nearest_supply = msg.get_nearest_supply(entry_price)
            if nearest_supply:
                candidates.append(nearest_supply.upper)

            # Swing highs
            valid_swings = [s.price for s in msg.swing_highs if s.price > entry_price]
            if valid_swings:
                candidates.append(min(valid_swings))  # closest swing high

            if candidates:
                sl_price = max(candidates)
            else:
                sl_price = entry_price + (atr_val * self.fallback_atr_mult)

            # Take Profit: Nearest demand zone upper, or nearest swing low/protected low
            tp_candidates = []
            nearest_demand = msg.get_nearest_demand(entry_price)
            if nearest_demand:
                tp_candidates.append(nearest_demand.upper)
            if msg.protected_low and msg.protected_low.price < entry_price:
                tp_candidates.append(msg.protected_low.price)
            valid_low_swings = [s.price for s in msg.swing_lows if s.price < entry_price]
            if valid_low_swings:
                tp_candidates.append(max(valid_low_swings))

            if tp_candidates:
                tp_price = max(tp_candidates)
            else:
                sl_dist = sl_price - entry_price
                tp_price = entry_price - (sl_dist * 2.0)

            # Invalidation is slightly above the SL price
            invalidation_level = sl_price + (atr_val * 0.1)

        # RR Calculation
        sl_distance = abs(entry_price - sl_price)
        tp_distance = abs(tp_price - entry_price)
        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0.0

        return {
            "entry_price": float(entry_price),
            "sl_price": float(sl_price),
            "tp_price": float(tp_price),
            "invalidation_level": float(invalidation_level),
            "rr_ratio": float(rr_ratio)
        }
