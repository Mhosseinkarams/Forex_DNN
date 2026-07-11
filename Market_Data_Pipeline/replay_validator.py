import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

from Market_Data_Pipeline.structure_engine import MarketStructureEngine
from Market_Data_Pipeline.supply_demand_engine import SupplyDemandEngine
from Market_Data_Pipeline.structure_graph import StructureLevel, BOS, CHOCH, Zone

logger = logging.getLogger('StructureReplayValidator')

class StructureReplayValidator:
    """
    Purpose:
        The "Ground Truth Inspector" for market structure.
        Simulates live trading by feeding candles one-by-one to detect look-ahead bias,
        historical repainting, confirmation delays, and zone lifecycle transitions.
    """
    def __init__(self, lookback_minor: int = 3, lookback_major: int = 10, impulse_threshold: float = 2.0):
        self.lookback_minor = lookback_minor
        self.lookback_major = lookback_major
        self.impulse_threshold = impulse_threshold

        # Historical state captured at each step of the replay loop
        # Format: step_idx -> list of objects
        self.swings_by_step: Dict[int, List[Dict[str, Any]]] = {}
        self.bos_by_step: Dict[int, List[Dict[str, Any]]] = {}
        self.choch_by_step: Dict[int, List[Dict[str, Any]]] = {}
        self.zones_by_step: Dict[int, List[Dict[str, Any]]] = {}

    def run_replay(self, df: pd.DataFrame, start_idx: int = 50) -> Dict[str, Any]:
        """
        Runs the iterative incremental validation process.
        """
        logger.info(f"Starting Structure Replay Validation on {len(df)} candles...")

        self.swings_by_step.clear()
        self.bos_by_step.clear()
        self.choch_by_step.clear()
        self.zones_by_step.clear()

        # Incremental simulation
        for step in range(start_idx, len(df)):
            df_step = df.iloc[:step + 1].copy()

            # Instantiate clean stateless engines
            ms_engine = MarketStructureEngine(
                lookback=self.lookback_minor,
                lookback_major=self.lookback_major
            )
            sd_engine = SupplyDemandEngine(
                atr_period=14,
                impulse_threshold=self.impulse_threshold
            )

            # Process up to current step
            df_ms = ms_engine.process(df_step)
            _ = sd_engine.process(df_ms)

            # Store objects as snapshot dicts
            self.swings_by_step[step] = [
                {
                    "index": s.index,
                    "price": s.price,
                    "level_type": s.level_type,
                    "structure_type": s.structure_type,
                    "confirmation_candle": s.confirmation_candle,
                    "is_valid": s.is_valid,
                    "broken": s.broken,
                    "reason": s.reason
                }
                for s in ms_engine.swings
            ]

            self.bos_by_step[step] = [
                {
                    "index": b.index,
                    "broken_level": b.broken_level,
                    "direction": b.direction,
                    "break_candle": b.break_candle
                }
                for b in ms_engine.bos_list
            ]

            self.choch_by_step[step] = [
                {
                    "index": c.index,
                    "price": c.price,
                    "previous_trend": c.previous_trend,
                    "new_trend": c.new_trend
                }
                for c in ms_engine.choch_list
            ]

            self.zones_by_step[step] = [
                {
                    "created_idx": z.created_idx,
                    "type": z.type,
                    "upper": z.upper,
                    "lower": z.lower,
                    "mitigated": z.mitigated,
                    "mitigated_idx": z.mitigated_idx,
                    "broken": z.broken,
                    "broken_idx": z.broken_idx,
                    "freshness": z.freshness,
                    "touch_count": z.touch_count
                }
                for z in sd_engine.zones
            ]

        # Analyze results
        metrics = self._compute_replay_metrics()
        return metrics

    def _compute_replay_metrics(self) -> Dict[str, Any]:
        """
        Analyze logs across steps to quantify stability, repainting, and lag.
        """
        steps = sorted(self.swings_by_step.keys())
        if not steps:
            return {"status": "No steps to analyze"}

        total_swings_detected = 0
        repainted_swings_count = 0
        swing_confirmation_delays: List[int] = []

        # Analyze Swing stability
        # A swing at a specific index should NEVER change its price or type once it appears
        detected_swings_registry: Dict[str, Dict[str, Any]] = {}  # key: "index_type" -> first seen step & properties

        for step in steps:
            swings = self.swings_by_step[step]
            for s in swings:
                key = f"{s['index']}_{s['level_type']}"
                if key not in detected_swings_registry:
                    # First time this swing is detected
                    detected_swings_registry[key] = {
                        "first_seen_step": step,
                        "price": s["price"],
                        "index": s["index"],
                        "level_type": s["level_type"],
                        "structure_type": s["structure_type"],
                        "confirmation_delay": step - s["index"],
                        "repainted": False
                    }
                    swing_confirmation_delays.append(step - s["index"])
                else:
                    # Check if swing repainted (changed price or went missing)
                    reg_entry = detected_swings_registry[key]
                    if reg_entry["price"] != s["price"]:
                        reg_entry["repainted"] = True
                        repainted_swings_count += 1

        # Check if any registered swings disappeared in the final step (classic repainting)
        final_step = steps[-1]
        final_swings_keys = {f"{s['index']}_{s['level_type']}" for s in self.swings_by_step[final_step]}
        for key, reg_entry in detected_swings_registry.items():
            if key not in final_swings_keys and reg_entry["first_seen_step"] < final_step - 10:
                # Disappeared swing
                if not reg_entry["repainted"]:
                    reg_entry["repainted"] = True
                    repainted_swings_count += 1

        total_swings_detected = len(detected_swings_registry)
        repaint_rate = (repainted_swings_count / total_swings_detected) if total_swings_detected > 0 else 0.0

        # Zone stability analysis
        total_zones_detected = 0
        zone_repainted_count = 0
        detected_zones_registry: Dict[str, Dict[str, Any]] = {}

        for step in steps:
            zones = self.zones_by_step[step]
            for z in zones:
                key = f"{z['created_idx']}_{z['type']}"
                if key not in detected_zones_registry:
                    detected_zones_registry[key] = {
                        "first_seen_step": step,
                        "upper": z["upper"],
                        "lower": z["lower"],
                        "repainted": False
                    }
                else:
                    reg_z = detected_zones_registry[key]
                    if reg_z["upper"] != z["upper"] or reg_z["lower"] != z["lower"]:
                        reg_z["repainted"] = True
                        zone_repainted_count += 1

        total_zones_detected = len(detected_zones_registry)
        zone_repaint_rate = (zone_repainted_count / total_zones_detected) if total_zones_detected > 0 else 0.0

        return {
            "total_swings_detected": total_swings_detected,
            "repainted_swings_count": repainted_swings_count,
            "swing_repaint_rate": repaint_rate,
            "average_swing_confirmation_delay": float(np.mean(swing_confirmation_delays)) if swing_confirmation_delays else 0.0,
            "total_zones_detected": total_zones_detected,
            "repainted_zones_count": zone_repainted_count,
            "zone_repaint_rate": zone_repaint_rate,
            "structure_causality_score": float(1.0 - repaint_rate),
            "zone_causality_score": float(1.0 - zone_repaint_rate)
        }
