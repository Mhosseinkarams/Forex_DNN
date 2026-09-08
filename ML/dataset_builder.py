import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from ML.feature_registry import FeatureRegistry
from ML.feature_pipeline import FeaturePipeline
from ML.market_state_labeler import MarketStateLabeler
from ML.level_event_labeler import LevelEventLabeler
from ML.strategy_outcome_evaluator import StrategyOutcomeEvaluator
from Market_Data_Pipeline.structure_graph import MarketStructureGraph, Zone

logger = logging.getLogger("DatasetBuilder")

class DatasetBuilder:
    """
    Purpose:
        Construct unified, multi-target, causal machine learning datasets
        from historical OHLCV data using MarketStructureGraph and analytical engines.
    """
    def __init__(
        self,
        registry: Optional[FeatureRegistry] = None,
        window_size: int = 35,
        market_state_horizon: int = 20,
        level_event_horizon: int = 20,
        strategy_horizon: int = 50
    ):
        self.registry = registry or FeatureRegistry()
        self.pipeline = FeaturePipeline(self.registry)
        self.window_size = window_size
        self.market_state_horizon = market_state_horizon
        self.level_event_horizon = level_event_horizon
        self.strategy_horizon = strategy_horizon

        self.market_state_labeler = MarketStateLabeler(future_horizon=market_state_horizon)
        self.level_event_labeler = LevelEventLabeler(future_horizon=level_event_horizon)
        self.strategy_outcome_evaluator = StrategyOutcomeEvaluator(future_horizon=strategy_horizon)

    def build_multi_target_dataset(self, df: pd.DataFrame, msg: MarketStructureGraph) -> pd.DataFrame:
        """
        Build unified multi-target dataset anchored at time t.
        Guarantees strict causal separation:
          - Features use data <= t.
          - Targets evaluate future candles > t.
        """
        rows = []
        max_horizon = max(self.market_state_horizon, self.level_event_horizon, self.strategy_horizon)
        warmup = max(100, self.window_size)

        for t in range(warmup, len(df) - max_horizon):
            feats = self.pipeline.extract_all(df, msg, idx=t)

            # Causal Market State (Feature/Metadata)
            curr_state, curr_conf, _ = self.market_state_labeler.evaluate_causal_current_state(
                df, msg, t - self.window_size + 1, t
            )

            # Future Market State Target (> t)
            fut_state, fut_conf, fut_info = self.market_state_labeler.label_window(
                df, msg, t - self.window_size + 1, t
            )

            row_close = df.at[t, "Close"]
            row_atr = df.at[t, "atr_14"] if "atr_14" in df.columns else 0.0001
            if row_atr <= 0:
                row_atr = 0.0001

            # Level Event Target (> t)
            near_supply = msg.get_nearest_supply_at(row_close, t)
            near_demand = msg.get_nearest_demand_at(row_close, t)
            target_zone = near_supply or near_demand

            if target_zone:
                lvl_res = self.level_event_labeler.evaluate_level_event(df, msg, t, target_zone)
                lvl_type = target_zone.type
                lvl_price = target_zone.mid
                lvl_dist = (target_zone.lower - row_close) if lvl_type == "Supply" else (row_close - target_zone.upper)
            else:
                lvl_res = None
                lvl_type = "NONE"
                lvl_price = row_close
                lvl_dist = 999.0

            # Strategy Outcome Target (> t)
            # Evaluate hypothetical trade setup based on trend direction
            trend_val = df.at[t, "trend"] if "trend" in df.columns else 0
            direction = 1 if trend_val >= 0 else -1
            sl_price = row_close - (1.5 * row_atr) if direction == 1 else row_close + (1.5 * row_atr)
            tp_price = row_close + (3.0 * row_atr) if direction == 1 else row_close - (3.0 * row_atr)

            strat_res = self.strategy_outcome_evaluator.evaluate_outcome(
                df, t, direction, row_close, sl_price, tp_price
            )

            row_dt = df.at[t, "Datetime"] if "Datetime" in df.columns else str(t)
            dt_str = row_dt.isoformat() if isinstance(row_dt, pd.Timestamp) else str(row_dt)

            row_data = {
                # IDENTITY
                "symbol": msg.symbol,
                "timeframe": msg.timeframe,
                "datetime": dt_str,
                "anchor_index": t,
                "dataset_version": "2.0.0-causal",

                # INPUT METADATA
                "window_start": t - self.window_size + 1,
                "window_end": t,
                "window_size": self.window_size,

                # CAUSAL MARKET STATE
                "current_market_state": curr_state or "TRANSITION",
                "current_trend_direction": trend_val,

                # FUTURE MARKET STATE TARGET
                "future_market_state": fut_state or "AMBIGUOUS",
                "future_state_confidence": fut_conf,
                "future_state_horizon": self.market_state_horizon,

                # LEVEL EVENT TARGET
                "level_type": lvl_type,
                "level_price": float(lvl_price),
                "level_distance_at_anchor": float(lvl_dist / row_atr),
                "level_event": lvl_res.event_type if lvl_res else "NO_INTERACTION",
                "break_probability_target": lvl_res.break_probability_target if lvl_res else None,
                "level_bars_to_resolution": lvl_res.bars_to_resolution if lvl_res else 0,
                "level_event_confidence": lvl_res.confidence if lvl_res else 0.0,
                "level_mae": lvl_res.mae if lvl_res else 0.0,
                "level_mfe": lvl_res.mfe if lvl_res else 0.0,

                # STRATEGY OUTCOME TARGET
                "strategy_name": "SMC_Default",
                "signal_type": "Candidate",
                "direction": direction,
                "entry_price": float(row_close),
                "sl_price": float(sl_price),
                "tp_price": float(tp_price),
                "risk_distance": float(abs(row_close - sl_price) / row_atr),
                "reward_distance": float(abs(tp_price - row_close) / row_atr),
                "r_multiple": float(strat_res.r_multiple),
                "strategy_mae": float(strat_res.mae_risk_ratio),
                "strategy_mfe": float(strat_res.mfe_risk_ratio),
                "strategy_bars_to_resolution": int(strat_res.bars_to_resolution),
                "exit_reason": strat_res.exit_reason,
                "strategy_outcome": strat_res.outcome,

                # QUALITY
                "label_status": "VALID" if fut_state is not None else "NO_LABEL",
                "label_confidence": float(fut_conf),
                "ambiguity_reason": fut_info.get("rule_fired", "none"),

                # INPUT FEATURES (from FeaturePipeline)
                **feats
            }

            rows.append(row_data)

        return pd.DataFrame(rows)

    def build_market_state_dataset(self, df: pd.DataFrame, msg: MarketStructureGraph) -> pd.DataFrame:
        """
        Build Dataset A: Market State Dataset (One row = one candle).
        Labels: TREND, RANGE, TRANSITION.
        Stores confidence of each label.
        """
        rows = []
        enabled_features = [f.name for f in self.registry.list_enabled()]

        # Calculate features and labels for each index (skip warmup period)
        warmup = 100
        for idx in range(warmup, len(df)):
            feats = self.pipeline.extract_all(df, msg, idx)

            # Derive label based on objective rules
            label, confidence = self._determine_market_state(df, msg, idx)

            # Construct row
            row_data = {**feats, "label": label, "confidence": confidence, "timestamp": df.iloc[idx].get("Datetime", idx)}
            rows.append(row_data)

        dataset = pd.DataFrame(rows)
        return dataset

    def build_level_break_dataset(
        self,
        df: pd.DataFrame,
        msg: MarketStructureGraph,
        lookahead_bars: int = 20,
        break_threshold_atr: float = 0.5,
        rejection_threshold_atr: float = 1.0,
        monitor: Optional[Any] = None,
        slot_id: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Build Dataset B: Level Break Dataset.
        One row = Price approaching significant supply/demand level.
        Target: 1 = level breaks, 0 = level rejects.
        """
        rows = []
        warmup = 100

        total_len = len(df) - lookahead_bars - warmup
        if total_len <= 0:
            return pd.DataFrame()

        if monitor and monitor.enabled and slot_id:
            monitor.update(slot_id, msg.symbol, "LABELING_LVL", 0, f"Started level break labeling ({total_len} bars)")
        else:
            logger.info(f"[{msg.symbol}] LEVEL BREAK PROCESSING: Starting level break labeling for {total_len} candles...")

        log_interval = max(1, total_len // 10)
        count = 0

        # Precompute active zone indices to avoid dataclass attribute lookups inside the high-frequency loop
        supply_created = [z.created_idx for z in msg.supply_zones]
        supply_broken = [z.broken_idx if (z.broken and z.broken_idx is not None) else 99999999 for z in msg.supply_zones]

        demand_created = [z.created_idx for z in msg.demand_zones]
        demand_broken = [z.broken_idx if (z.broken and z.broken_idx is not None) else 99999999 for z in msg.demand_zones]

        for idx in range(warmup, len(df) - lookahead_bars):
            count += 1
            row_close = df.iloc[idx]["Close"]

            if count % log_interval == 0 or count == total_len:
                pct = int(count / total_len * 100)
                if monitor and monitor.enabled and slot_id:
                    monitor.update(slot_id, msg.symbol, "LABELING_LVL", pct, f"Bars {count}/{total_len}")
                else:
                    logger.info(f"[{msg.symbol}] LEVEL BREAK PROCESSING: {pct}% complete ({count}/{total_len} bars)")
            row_high = df.iloc[idx]["High"]
            row_low = df.iloc[idx]["Low"]
            atr = df.iloc[idx].get("atr_14", 0.0001)

            # Identify nearest active zone using optimized index lists
            active_supplies = [msg.supply_zones[i] for i, (c_idx, b_idx) in enumerate(zip(supply_created, supply_broken)) if c_idx <= idx < b_idx]
            active_demands = [msg.demand_zones[i] for i, (c_idx, b_idx) in enumerate(zip(demand_created, demand_broken)) if c_idx <= idx < b_idx]

            # Check proximity to supply zone (close is within 0.5 ATR below lower bound)
            near_supply = None
            for s in active_supplies:
                if 0 < (s.lower - row_close) <= 0.5 * atr:
                    near_supply = s
                    break

            # Check proximity to demand zone (close is within 0.5 ATR above upper bound)
            near_demand = None
            for d in active_demands:
                if 0 < (row_close - d.upper) <= 0.5 * atr:
                    near_demand = d
                    break

            if not near_supply and not near_demand:
                continue

            # Perform lookahead to label 1 (break) or 0 (rejection)
            target = None
            if near_supply:
                # Supply Break: price exceeds upper boundary
                # Supply Reject: price drops below lower boundary by rejection_threshold_atr * ATR
                for l in range(1, lookahead_bars + 1):
                    future_bar = df.iloc[idx + l]
                    if future_bar["High"] > near_supply.upper:
                        target = 1  # Break
                        break
                    elif future_bar["Low"] < near_supply.lower - rejection_threshold_atr * atr:
                        target = 0  # Rejection
                        break
            elif near_demand:
                # Demand Break: price drops below lower boundary
                # Demand Reject: price rises above upper boundary by rejection_threshold_atr * ATR
                for l in range(1, lookahead_bars + 1):
                    future_bar = df.iloc[idx + l]
                    if future_bar["Low"] < near_demand.lower:
                        target = 1  # Break
                        break
                    elif future_bar["High"] > near_demand.upper + rejection_threshold_atr * atr:
                        target = 0  # Rejection
                        break

            if target is None:
                continue  # Indeterminate within lookahead window

            feats = self.pipeline.extract_all(df, msg, idx)
            row_data = {
                **feats,
                "target": target,
                "zone_type": "Supply" if near_supply else "Demand",
                "timestamp": df.iloc[idx].get("Datetime", idx)
            }
            rows.append(row_data)

        dataset = pd.DataFrame(rows)
        return dataset

    def _determine_market_state(self, df: pd.DataFrame, msg: MarketStructureGraph, idx: int) -> Tuple[str, float]:
        """
        Core objective labeling logic for Market States.
        TREND: Repeated BOS and strong EMA alignment
        RANGE: Prolonged absence of BOS and multiple rejection at zones (or converged EMAs)
        TRANSITION: Everything else
        """
        row = df.iloc[idx]
        atr = row.get("atr_14", 0.0001)

        # Check EMA separation
        fast_ema = row.get("ema_50")
        slow_ema = row.get("ema_600", row.get("ema_800"))

        ema_aligned = False
        ema_separation_atr = 0.0
        if fast_ema is not None and slow_ema is not None:
            ema_separation_atr = abs(fast_ema - slow_ema) / (atr + 1e-9)
            if ema_separation_atr > 1.5:
                ema_aligned = True

        # Check BOS count in the last 100 bars
        bos_count = 0
        for b in msg.bos:
            if idx - 100 <= b.index <= idx:
                bos_count += 1

        # Alternating CHOCH check
        choch_count = 0
        for c in msg.choch:
            if idx - 100 <= c.index <= idx:
                choch_count += 1

        # repeated rejection at zones
        rejections = 0
        active_zones = [z for z in msg.supply_zones + msg.demand_zones if z.created_idx <= idx]
        for z in active_zones:
            if z.touch_count > 1:
                rejections += 1

        # Decision rules
        if ema_aligned and bos_count >= 1:
            confidence = min(1.0, 0.5 + (ema_separation_atr / 10.0) + (bos_count * 0.1))
            return "TREND", float(confidence)

        if (bos_count == 0 and choch_count >= 1) or rejections >= 2 or ema_separation_atr < 0.8:
            confidence = min(1.0, 0.6 + (choch_count * 0.05) + (rejections * 0.05))
            return "RANGE", float(confidence)

        # Default to Transition
        return "TRANSITION", 0.5
