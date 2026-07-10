from typing import Dict, Any, Optional
import numpy as np
from Market_Data_Pipeline.structure_graph import MarketStructureGraph

class FeaturePipeline:
    """
    Purpose:
        Consume the shared MarketStructureGraph and produce a dictionary of ML-ready features.
        Never calculates indicators directly, instead utilizes pre-computed indicators and
        structural objects present in the graph.
    """
    def __init__(self):
        pass

    def extract_features(self, msg: MarketStructureGraph, current_price: float) -> Dict[str, Any]:
        """
        Extract features from MarketStructureGraph relative to the current market price.
        """
        # Distances to nearest demand/supply zones
        nearest_demand = msg.get_nearest_demand(current_price)
        nearest_supply = msg.get_nearest_supply(current_price)

        distance_to_demand = current_price - nearest_demand.upper if nearest_demand else -1.0
        distance_to_supply = nearest_supply.lower - current_price if nearest_supply else -1.0

        demand_strength = nearest_demand.strength_score if nearest_demand else 0.0
        supply_strength = nearest_supply.strength_score if nearest_supply else 0.0

        zone_age_demand = (msg.timestamp - nearest_demand.created_time).total_seconds() / 60.0 if (nearest_demand and nearest_demand.created_time and msg.timestamp) else -1.0
        zone_age_supply = (msg.timestamp - nearest_supply.created_time).total_seconds() / 60.0 if (nearest_supply and nearest_supply.created_time and msg.timestamp) else -1.0

        zone_freshness_demand = float(nearest_demand.freshness) if nearest_demand else 0.0
        zone_freshness_supply = float(nearest_supply.freshness) if nearest_supply else 0.0

        # Protected high/low distances
        protected_high_dist = msg.protected_high.price - current_price if msg.protected_high else -1.0
        protected_low_dist = current_price - msg.protected_low.price if msg.protected_low else -1.0

        # BOS / CHOCH counts
        bos_count = len(msg.bos)
        choch_count = len(msg.choch)

        # Range Width and volatility
        range_width = msg.range_width_pips
        trend_strength = 1.0 if msg.trend_direction in ["Bull", "Bear"] else 0.0

        # EMA parameters
        ema_separation = msg.ema_distance_atr

        # Liquidity distances
        liquidity_dist = -1.0
        if msg.liquidity_pools:
            active_pools = [p for p in msg.liquidity_pools if not p.is_swept]
            if active_pools:
                # Find nearest pool boundary
                nearest_pool = min(active_pools, key=lambda p: min(abs(p.upper - current_price), abs(p.lower - current_price)))
                liquidity_dist = min(abs(nearest_pool.upper - current_price), abs(nearest_pool.lower - current_price))

        return {
            "distance_to_demand": float(distance_to_demand),
            "distance_to_supply": float(distance_to_supply),
            "demand_strength": float(demand_strength),
            "supply_strength": float(supply_strength),
            "zone_age_demand": float(zone_age_demand),
            "zone_age_supply": float(zone_age_supply),
            "zone_freshness_demand": float(zone_freshness_demand),
            "zone_freshness_supply": float(zone_freshness_supply),
            "liquidity_distance": float(liquidity_dist),
            "protected_low_distance": float(protected_low_dist),
            "protected_high_distance": float(protected_high_dist),
            "bos_count": int(bos_count),
            "choch_count": int(choch_count),
            "range_width": float(range_width),
            "trend_strength": float(trend_strength),
            "ema_separation": float(ema_separation),
            "session": str(msg.session),
            "atr": float(msg.atr),
            "volatility": float(msg.volatility)
        }
