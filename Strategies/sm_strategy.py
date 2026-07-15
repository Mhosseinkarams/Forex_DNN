import os
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional, List, Any, Dict, Tuple

# Optional MT5 import
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

import pandas as pd
import numpy as np

# Framework engines
from Collecting_Data.indicators import IndicatorEngine
from Collecting_Data.position_lifecycle import EXIT_PROFILE_STANDARD, EXIT_PROFILE_SINGLE
from Collecting_Data.utils import safe_file_replace
from Core.trend_context import TrendContext, TrendContextBuilder

# Market structure & supply demand
from Market_Data_Pipeline.structure_graph import MarketStructureGraph, Zone, StructureLevel
from Market_Data_Pipeline.structure_engine import MarketStructureEngine
from Market_Data_Pipeline.supply_demand_engine import SupplyDemandEngine
from Market_Data_Pipeline.state_engine import MarketStateEngine, StateContext
from Trade_Execution.location_engine import TradeLocationEngine

# Runtime ML & confirmation
from ML.feature_pipeline import FeaturePipeline
from ML.ml_decision_engine import MLDecisionEngine
from ML.decision_context import DecisionContext
from Strategies.signal_evaluator import SignalEvaluator, SignalEvaluation
from ML.trade_feature_recorder import TradeFeatureRecorder

# SM additions
from Strategies.refusal_candle_engine import RefusalCandleEngine, RefusalResult

logger = logging.getLogger("SMStrategy")

class SMStrategy:
    """
    Stubborn Man (SM) Strategy

    A sophisticated range trading mean-reversion strategy that trades rejections of
    key supply and demand boundaries. It assumes price is more likely to reject
    important structural zones than immediately break them.

    Thread Safety:
        Uses a re-entrant lock `self._lock` to synchronize all polling thread operations.
    """
    def __init__(
        self,
        data_feed,                    # DataFeed instance
        send_order,                   # SendOrder instance
        trading_journal,              # TradingJournal instance
        drawdown_manager,             # DrawdownManager instance
        symbols: list[str],           # e.g. ["EURUSD_o", "GBPUSD_o"]
        poll_interval_seconds: float = 5.0,
        state_file: str = "sm_strategy_state.json",
        location_engine: Optional[TradeLocationEngine] = None,
        annotator: Optional[Any] = None,
        market_state_model: Optional[Any] = None,
        level_break_model: Optional[Any] = None,
        decision_engine: Optional[MLDecisionEngine] = None,
        feature_pipeline: Optional[FeaturePipeline] = None,
        signal_evaluator: Optional[SignalEvaluator] = None,
        recorder: Optional[TradeFeatureRecorder] = None,
        refusal_engine: Optional[RefusalCandleEngine] = None,

        # SM Strategy Configuration (Configurable Thresholds)
        min_zone_strength: float = 1.0,
        max_zone_age_bars: int = 300,
        min_refusal_score: float = 65.0,
        max_break_probability: float = 0.35, # Expecting level to HOLD, not BREAK
        min_rr: float = 1.2,
        tp_buffer_pct: float = 0.10,         # Buffer before opposite zone (as percentage of zone width/ATR)
        sl_buffer_pct: float = 0.05,         # Buffer beyond zone for SL placement
        shadow_mode: bool = True,
        ml_filtering: bool = False,
        m5_ema_periods: list[int] = [50, 600],
        m15_ema_periods: list[int] = [50, 800]
    ):
        self.data_feed = data_feed
        self.send_order = send_order
        self.trading_journal = trading_journal
        self.drawdown_manager = drawdown_manager
        self.symbols = symbols
        self.poll_interval_seconds = poll_interval_seconds
        self.state_file = state_file
        self.annotator = annotator
        self.market_state_model = market_state_model
        self.level_break_model = level_break_model

        # Config parameters
        self.min_zone_strength = min_zone_strength
        self.max_zone_age_bars = max_zone_age_bars
        self.min_refusal_score = min_refusal_score
        self.max_break_probability = max_break_probability
        self.min_rr = min_rr
        self.tp_buffer_pct = tp_buffer_pct
        self.sl_buffer_pct = sl_buffer_pct
        self.shadow_mode = shadow_mode
        self.ml_filtering = ml_filtering

        # Timeframes & Indicator Engines
        self.m5_ema_periods = m5_ema_periods
        self.m15_ema_periods = m15_ema_periods
        self.engine_m5 = IndicatorEngine(ema_periods=self.m5_ema_periods, slope_period=32)
        self.engine_m15 = IndicatorEngine(ema_periods=self.m15_ema_periods, slope_period=32)

        # Analytical Engines
        self.struct_engine = MarketStructureEngine(lookback=3)
        self.sd_engine = SupplyDemandEngine(atr_period=14, impulse_threshold=2.0)
        self.state_engine = MarketStateEngine()
        self.location_engine = location_engine or TradeLocationEngine()

        # Refusal Candle Engine
        self.refusal_engine = refusal_engine or RefusalCandleEngine()

        # ML Services
        self.feature_pipeline = feature_pipeline or FeaturePipeline()
        self.decision_engine = decision_engine or MLDecisionEngine()
        self.signal_evaluator = signal_evaluator or SignalEvaluator(shadow_mode=self.shadow_mode, ml_filtering=self.ml_filtering)
        self.recorder = recorder or TradeFeatureRecorder()

        # Connect recorder to TradingJournal
        if self.trading_journal is not None:
            self.trading_journal.recorder = self.recorder

        self._setup_runtime_loggers()

        # Shared trackers
        self.last_bar_time = {}  # symbol -> timeframe -> timestamp
        self.signal_history = {s: {"M5": [], "M15": []} for s in symbols}
        self._bar_counters = {s: {"M5": 0, "M15": 0} for s in symbols}

        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

        self._load_state()

    def _setup_runtime_loggers(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logs_dir = os.path.join(base_dir, "Logs")

        def get_or_setup_logger(name, filename_rel):
            filename = os.path.join(logs_dir, filename_rel)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            logger_obj = logging.getLogger(name)
            logger_obj.setLevel(logging.INFO)
            if not logger_obj.handlers:
                handler = logging.FileHandler(filename)
                handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                logger_obj.addHandler(handler)
            logger_obj.propagate = False
            return logger_obj

        self.features_logger = get_or_setup_logger("sm_runtime_features", "sm_runtime_features.log")
        self.decision_logger = get_or_setup_logger("sm_decision_engine", "sm_decision_engine.log")
        self.shadow_logger = get_or_setup_logger("sm_shadow_mode", "sm_shadow_mode.log")
        self.evaluator_logger = get_or_setup_logger("sm_signal_evaluator", "sm_signal_evaluator.log")

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    self.last_bar_time = json.load(f)
                logger.info(f"Loaded SM state from {self.state_file}")
            except Exception as e:
                logger.error(f"Failed to load SM state: {e}")

    def _save_state(self):
        try:
            temp_file = self.state_file + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(self.last_bar_time, f, indent=4)
            safe_file_replace(temp_file, self.state_file)
        except Exception as e:
            logger.error(f"Failed to save SM state: {e}")

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.warning("SMStrategy is already running.")
            return

        with self._lock:
            self.signal_history = {s: {"M5": [], "M15": []} for s in self.symbols}
            self._bar_counters = {s: {"M5": 0, "M15": 0} for s in self.symbols}

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("SMStrategy started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("SMStrategy stopped.")

    def _poll_loop(self):
        while not self._stop_event.is_set():
            try:
                self._poll_cycle()
            except Exception as e:
                logger.error(f"Error in SM poll cycle: {e}", exc_info=True)
            time.sleep(self.poll_interval_seconds)

    def _poll_cycle(self):
        for symbol in self.symbols:
            for timeframe, fast_p, slow_p in [("M5", 50, 600), ("M15", 50, 800)]:
                df_raw = self.data_feed.get_ohlcv(symbol, timeframe)
                if df_raw is None or len(df_raw) < (slow_p + 50):
                    logger.warning(f"Insufficient data for {symbol} {timeframe}")
                    continue

                engine = self.engine_m5 if timeframe == "M5" else self.engine_m15
                df = engine.calculate(df_raw)

                if not self._is_new_bar(symbol, timeframe, df):
                    continue

                with self._lock:
                    self._bar_counters[symbol][timeframe] += 1
                logger.info(f"[SM] New bar detected for {symbol} {timeframe}: {df.iloc[-1]['Datetime']}")

                # Active chart annotation (handled structurally)
                self._evaluate_setup_and_trade(symbol, timeframe, df)
                self._save_state()

    def _is_new_bar(self, symbol, timeframe, df):
        current_bar_time = str(df.iloc[-1]["Datetime"])
        with self._lock:
            if symbol not in self.last_bar_time:
                self.last_bar_time[symbol] = {}
            if timeframe not in self.last_bar_time[symbol]:
                self.last_bar_time[symbol][timeframe] = current_bar_time
                return False
            if self.last_bar_time[symbol][timeframe] != current_bar_time:
                self.last_bar_time[symbol][timeframe] = current_bar_time
                return True
        return False

    def _build_market_structure_graph(self, symbol: str, timeframe: str, df: pd.DataFrame) -> MarketStructureGraph:
        df_struct = self.struct_engine.process(df)
        df_sd = self.sd_engine.process(df_struct)

        last_row = df_sd.iloc[-1]
        dt = pd.to_datetime(last_row["Datetime"]) if "Datetime" in df_sd.columns else datetime.now(timezone.utc)

        swing_highs_list = [s for s in self.struct_engine.swings if s.level_type == 'SwingHigh']
        swing_lows_list = [s for s in self.struct_engine.swings if s.level_type == 'SwingLow']

        graph = MarketStructureGraph(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=dt,
            swing_highs=swing_highs_list,
            swing_lows=swing_lows_list,
            protected_high=self.struct_engine.protected_high,
            protected_low=self.struct_engine.protected_low,
            bos=list(self.struct_engine.bos_list),
            choch=list(self.struct_engine.choch_list),
            supply_zones=[z for z in self.sd_engine.zones if z.type == 'Supply'],
            demand_zones=[z for z in self.sd_engine.zones if z.type == 'Demand'],
            trend_direction="Bull" if last_row.get("trend", 0) == 1 else ("Bear" if last_row.get("trend", 0) == -1 else "Neutral"),
            atr=float(last_row.get("atr_14", 0.0001)),
            volatility=float(last_row.get("atr_14", 0.0001) * 10000.0)
        )
        return graph

    def _evaluate_setup_and_trade(self, symbol: str, timeframe: str, df: pd.DataFrame):
        idx_closed = len(df) - 2 # Analyze the fully closed candle
        bar_closed = df.iloc[idx_closed]
        bar_timestamp = str(bar_closed["Datetime"])

        # Create structural graph point-in-time
        msg = self._build_market_structure_graph(symbol, timeframe, df)

        # Contextual feature extraction
        session_val = "Asian"
        spread_val = float(bar_closed.get("Spread", 0.0))
        dt_val = pd.to_datetime(bar_closed["Datetime"])
        h = dt_val.hour
        if 8 <= h < 13: session_val = "London"
        elif 13 <= h < 17: session_val = "London/NY"
        elif 17 <= h < 22: session_val = "NewYork"

        account_session_ctx = {"session": session_val, "spread": spread_val}
        strategy_ctx = {"strategy": "sm", "signal_type": "reversal"}

        features_fv = self.feature_pipeline.extract_runtime(
            df=df,
            msg=msg,
            idx=idx_closed,
            account_session_context=account_session_ctx,
            strategy_context=strategy_ctx
        )
        self.features_logger.info(f"Features at {bar_timestamp}: {features_fv.features}")

        # Step 1: Detect Market Regime (Only RANGE)
        predicted_state = "UNKNOWN"
        decision_ctx = None
        if self.decision_engine:
            try:
                decision_ctx = self.decision_engine.evaluate(
                    symbol=symbol,
                    timeframe=timeframe,
                    feature_vector=features_fv.features,
                    strategy_name="SMStrategy",
                    timestamp=bar_timestamp
                )
                predicted_state = decision_ctx.predicted_state
            except Exception as ex:
                logger.error(f"Inference evaluation failed: {ex}")

        # Force state evaluation to fallback if decision context is missing
        if predicted_state == "UNKNOWN":
            state_ctx = self.state_engine.evaluate(msg)
            predicted_state = state_ctx.regime # "RANGE", "TREND", "TRANSITION"

        # Step 1: Filter on RANGE regime
        if predicted_state != "RANGE":
            logger.info(f"[SM] Skip {symbol} {timeframe} at {bar_timestamp}: Regime is {predicted_state}, not RANGE.")
            return

        # Step 2 & 3: Identify active ranges and check price interactions
        # Extract active, eligible zones based on thresholds
        eligible_zones: List[Zone] = []
        for zone in msg.supply_zones + msg.demand_zones:
            # Check strength
            if zone.strength_score < self.min_zone_strength:
                continue
            # Check age
            zone_age = idx_closed - zone.created_idx
            if zone_age > self.max_zone_age_bars:
                continue
            # Check invalidation
            if zone.broken or zone.invalidated:
                continue
            eligible_zones.append(zone)

        if not eligible_zones:
            logger.info(f"[SM] No eligible zones found for {symbol} {timeframe}")
            return

        # Test interaction: High or Low of closed bar must overlap/penetrate an eligible zone
        candidate_zones: List[Tuple[Zone, float, int]] = [] # (Zone, penetration, direction)
        high_p = float(bar_closed["High"])
        low_p = float(bar_closed["Low"])

        for zone in eligible_zones:
            if zone.type == "Supply":
                # Sell setup: bar high overlaps or goes above zone lower edge
                if high_p >= zone.lower:
                    penetration = high_p - zone.lower
                    candidate_zones.append((zone, penetration, -1))
            else:
                # Buy setup: bar low overlaps or goes below zone upper edge
                if low_p <= zone.upper:
                    penetration = zone.upper - low_p
                    candidate_zones.append((zone, penetration, 1))

        if not candidate_zones:
            logger.info(f"[SM] Price is away from structural levels for {symbol} {timeframe}")
            return

        # Step 4: RefusalCandleEngine scoring
        # Evaluate each candidate zone rejection, pick the strongest
        best_rejection: Optional[Tuple[Zone, RefusalResult, int]] = None
        for zone, pen, direction in candidate_zones:
            res = self.refusal_engine.evaluate_rejection(df, idx_closed, zone, msg)
            if best_rejection is None or res.score > best_rejection[1].score:
                best_rejection = (zone, res, direction)

        if not best_rejection:
            return

        active_zone, refusal_result, direction = best_rejection

        # Step 5: Trade Confirmation by Refusal Score Threshold
        if refusal_result.score < self.min_refusal_score:
            logger.info(
                f"[SM] Setup rejected for {symbol} {timeframe} | "
                f"Refusal score {refusal_result.score} is below threshold {self.min_refusal_score}"
            )
            # Log rejected trade to journal/recorder
            self._record_and_log_setup(
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                msg=msg,
                refusal_result=refusal_result,
                decision_ctx=decision_ctx,
                features_fv=features_fv,
                bar_timestamp=bar_timestamp,
                accepted=False,
                reason=f"Weak refusal score: {refusal_result.score}"
            )
            return

        # Step 6: Level Break Probability check
        break_prob = decision_ctx.break_probability if (decision_ctx and "LevelBreakProbabilityModel" in getattr(decision_ctx, "model_versions", {})) else 0.0
        if break_prob > self.max_break_probability:
            logger.info(
                f"[SM] Setup rejected for {symbol} {timeframe} | "
                f"Level Break Probability {break_prob:.4f} exceeds threshold {self.max_break_probability}"
            )
            self._record_and_log_setup(
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                msg=msg,
                refusal_result=refusal_result,
                decision_ctx=decision_ctx,
                features_fv=features_fv,
                bar_timestamp=bar_timestamp,
                accepted=False,
                reason=f"High break probability: {break_prob:.4f}"
            )
            return

        # Step 8: TradeLevelEngine SL/TP calculations
        # Set Entry Price
        tick = mt5.symbol_info_tick(symbol) if mt5 else None
        if tick and tick.ask > 0 and tick.bid > 0:
            entry_price = float(tick.ask if direction == 1 else tick.bid)
        else:
            entry_price = float(bar_closed["Close"])

        # SL must be placed BEYOND the active zone
        zone_width = active_zone.upper - active_zone.lower
        sl_offset = zone_width * self.sl_buffer_pct
        if direction == 1:
            sl_price = active_zone.lower - sl_offset
        else:
            sl_price = active_zone.upper + sl_offset

        # TP is structural, placed BEFORE the opposite zone applying a buffer
        opposite_zones = msg.supply_zones if direction == 1 else msg.demand_zones
        valid_opposites = [z for z in opposite_zones if not z.broken and not z.invalidated]

        # Buffer calculation
        atr_val = msg.atr if msg.atr > 0 else 0.0001
        tp_offset = atr_val * self.tp_buffer_pct

        tp_price = 0.0
        if direction == 1:
            # BUY: Target nearest supply lower edge with buffer
            filtered_opps = [z for z in valid_opposites if z.lower > entry_price]
            if filtered_opps:
                # Rank TP candidates (pick the nearest one for safety)
                sorted_opps = sorted(filtered_opps, key=lambda z: z.lower)
                tp_price = sorted_opps[0].lower - tp_offset
            else:
                # Fallback to standard 1.5 R:R
                sl_dist = abs(entry_price - sl_price)
                tp_price = entry_price + (sl_dist * 1.5)
        else:
            # SELL: Target nearest demand upper edge with buffer
            filtered_opps = [z for z in valid_opposites if z.upper < entry_price]
            if filtered_opps:
                sorted_opps = sorted(filtered_opps, key=lambda z: z.upper, reverse=True)
                tp_price = sorted_opps[0].upper + tp_offset
            else:
                sl_dist = abs(sl_price - entry_price)
                tp_price = entry_price - (sl_dist * 1.5)

        # RR validation
        sl_dist = abs(entry_price - sl_price)
        tp_dist = abs(tp_price - entry_price)
        rr_ratio = tp_dist / sl_dist if sl_dist > 0 else 0.0

        if rr_ratio < self.min_rr:
            logger.info(
                f"[SM] Setup rejected for {symbol} {timeframe} | "
                f"RR ratio {rr_ratio:.2f} is below minimum {self.min_rr}"
            )
            self._record_and_log_setup(
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                msg=msg,
                refusal_result=refusal_result,
                decision_ctx=decision_ctx,
                features_fv=features_fv,
                bar_timestamp=bar_timestamp,
                accepted=False,
                reason=f"Poor RR ratio: {rr_ratio:.2f}"
            )
            return

        # Step 9: SignalEvaluator performing final validation
        trading_allowed = self.drawdown_manager.trading_allowed()
        risk_state = {
            "trading_allowed": trading_allowed,
            "drawdown_limit_reached": not trading_allowed
        }

        candidate_dict = {
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,
            "signal_type": "reversal",
            "technical_rules_satisfied": True
        }

        evaluation = self.signal_evaluator.evaluate(
            strategy_name="SMStrategy",
            signal_candidate=candidate_dict,
            feature_vector=features_fv.features,
            decision_context=decision_ctx,
            market_structure=msg,
            supply_demand=active_zone,
            risk_state=risk_state
        )

        # Step 10: Log to Journal & Record
        self._record_and_log_setup(
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            msg=msg,
            refusal_result=refusal_result,
            decision_ctx=decision_ctx,
            features_fv=features_fv,
            bar_timestamp=bar_timestamp,
            accepted=evaluation.accepted,
            reason=", ".join(evaluation.reasons),
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            rr_ratio=rr_ratio
        )

        # Execute order if allowed
        if evaluation.accepted and trading_allowed:
            logger.info(f"[SM] Submitting live mean-reversion setup for {symbol} {timeframe}")
            res = self.send_order.execute(
                symbol=symbol,
                direction=direction,
                entry_price=0.0,
                sl_price=sl_price,
                exit_profile=EXIT_PROFILE_SINGLE, # single exit mean-reversion
                strategy="sm",
                signal_category="reversal",
                signal_id=0, # filled dynamically
                tp_price=tp_price
            )
            logger.info(f"[SM] Order status: {res.get('success')} - {res.get('reason')}")

        # Active Chart Annotation for SM Strategy
        if self.annotator:
            try:
                state_ctx = self.state_engine.evaluate(msg)
                decision_dict = {
                    "direction": direction,
                    "accepted": evaluation.accepted,
                    "reason": f"Refusal Score: {refusal_result.score}",
                    "signal_type": "reversal",
                    "strategy": "sm"
                }

                ml_render_data = {}
                if decision_ctx:
                    ml_render_data = {
                        "trend_prob": decision_ctx.state_probabilities.get("TREND", 0.0),
                        "range_prob": decision_ctx.state_probabilities.get("RANGE", 0.0),
                        "transition_prob": decision_ctx.state_probabilities.get("TRANSITION", 0.0),
                        "break_prob": decision_ctx.break_probability,
                        "reject_prob": decision_ctx.rejection_probability,
                        "confidence": decision_ctx.confidence_score
                    }

                trade_plan = {
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "invalidation_level": sl_price
                }

                self.annotator.render(
                    symbol=symbol,
                    structure_graph=msg,
                    state_context=state_ctx,
                    trade_plan=trade_plan,
                    decision=decision_dict,
                    ml_output=ml_render_data
                )
            except Exception as ex:
                logger.error(f"Failed to render chart annotations: {ex}")

    def _record_and_log_setup(
        self,
        symbol: str,
        timeframe: str,
        direction: int,
        msg: MarketStructureGraph,
        refusal_result: RefusalResult,
        decision_ctx: Optional[DecisionContext],
        features_fv: Any,
        bar_timestamp: str,
        accepted: bool,
        reason: str,
        entry_price: float = 0.0,
        sl_price: float = 0.0,
        tp_price: float = 0.0,
        rr_ratio: float = 0.0
    ):
        """
        Helper method to coordinate logging to TradingJournal, TradeFeatureRecorder, and custom logs.
        """
        # Formulate metrics payload
        extra_fields = {
            "strategy": "sm",
            "refusal_score": refusal_result.score,
            "refusal_confidence": refusal_result.confidence,
            "break_probability": decision_ctx.break_probability if decision_ctx else 0.5,
            "rr_ratio": rr_ratio,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "reason": reason,
            "accepted": accepted
        }

        # Log to Signal Journal (Layer 1)
        signal_id = 0
        if self.trading_journal:
            try:
                signal_id = self.trading_journal.log_signal(
                    signal_type="reversal",
                    symbol=symbol,
                    timeframe=timeframe,
                    direction=direction,
                    entry_price=entry_price,
                    sl_price=sl_price,
                    exit_profile=EXIT_PROFILE_SINGLE,
                    strategy="sm",
                    signal_category="reversal",
                    bar_timestamp=bar_timestamp,
                    extra_fields=extra_fields
                )
            except Exception as j_ex:
                logger.error(f"Failed to log signal to trading journal: {j_ex}")

        # Log to TradeFeatureRecorder (Layer 2)
        if self.recorder:
            try:
                self.recorder.record_candidate(
                    signal_id=signal_id if signal_id else int(time.time()),
                    timestamp=bar_timestamp,
                    strategy="sm",
                    symbol=symbol,
                    timeframe=timeframe,
                    direction="BUY" if direction == 1 else "SELL",
                    features=features_fv.features,
                    decision_context=decision_ctx,
                    accepted=accepted,
                    reason=reason
                )
            except Exception as r_ex:
                logger.error(f"Failed to record features to TradeFeatureRecorder: {r_ex}")

        # Write to specialized runtime loggers
        shadow_log_msg = (
            f"SM_CANDIDATE | Time: {bar_timestamp} | Symbol: {symbol} | Timeframe: {timeframe} | "
            f"Direction: {'BUY' if direction == 1 else 'SELL'} | RefusalScore: {refusal_result.score:.2f} | "
            f"BreakProb: {extra_fields['break_probability']:.4f} | RR: {rr_ratio:.2f} | "
            f"FinalDecision: {'Accepted' if accepted else 'Rejected'} | Reason: {reason}"
        )
        self.shadow_logger.info(shadow_log_msg)
        self.evaluator_logger.info(f"Signal Evaluation for SM: {extra_fields}")
