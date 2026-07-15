import os
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional, List, Any, Dict, Tuple
from dataclasses import asdict

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

# Signal Intelligence Layer dependencies
from Market_Data_Pipeline.strong_candle_engine import StrongCandleEngine, StrongCandle
from Market_Data_Pipeline.refusal_candle_engine import RefusalCandleEngine, RefusalSignal
from Core.signal_candidate import SignalCandidate
from Collecting_Data.signal_recorder import SignalRecorder

logger = logging.getLogger("UniTStrategy")

class UniTStrategy:
    """
    UniT Strategy (Trend-Continuation Pullback Strategy)

    A sophisticated trend continuation strategy that requires strong momentum expansion
    and pullback refusals at key supply/demand areas to trade in the direction of the trend.
    """
    def __init__(
        self,
        data_feed,                    # DataFeed instance
        send_order,                   # SendOrder instance
        trading_journal,              # TradingJournal instance
        drawdown_manager,             # DrawdownManager instance
        symbols: list[str],           # e.g. ["EURUSD_o", "GBPUSD_o"]
        poll_interval_seconds: float = 5.0,
        state_file: str = "unit_strategy_state.json",
        location_engine: Optional[TradeLocationEngine] = None,
        annotator: Optional[Any] = None,
        market_state_model: Optional[Any] = None,
        level_break_model: Optional[Any] = None,
        decision_engine: Optional[MLDecisionEngine] = None,
        feature_pipeline: Optional[FeaturePipeline] = None,
        signal_evaluator: Optional[SignalEvaluator] = None,
        recorder: Optional[TradeFeatureRecorder] = None,

        # Mandatory Engine Dependencies
        strong_candle_engine: Optional[StrongCandleEngine] = None,
        refusal_engine: Optional[RefusalCandleEngine] = None,
        signal_recorder: Optional[SignalRecorder] = None,

        # UniT Configurable Thresholds
        min_trend_slope: float = 0.05,
        min_strong_candle_score: float = 70.0,
        min_pullback_refusal_score: float = 60.0,
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

        # Configs
        self.min_trend_slope = min_trend_slope
        self.min_strong_candle_score = min_strong_candle_score
        self.min_pullback_refusal_score = min_pullback_refusal_score
        self.shadow_mode = shadow_mode
        self.ml_filtering = ml_filtering

        # Engines
        self.m5_ema_periods = m5_ema_periods
        self.m15_ema_periods = m15_ema_periods
        self.engine_m5 = IndicatorEngine(ema_periods=self.m5_ema_periods, slope_period=32)
        self.engine_m15 = IndicatorEngine(ema_periods=self.m15_ema_periods, slope_period=32)

        self.struct_engine = MarketStructureEngine(lookback=3)
        self.sd_engine = SupplyDemandEngine(atr_period=14, impulse_threshold=2.0)
        self.state_engine = MarketStateEngine()
        self.location_engine = location_engine or TradeLocationEngine()

        # Signal Intelligence Layer Engines
        self.strong_candle_engine = strong_candle_engine or StrongCandleEngine()
        self.refusal_engine = refusal_engine or RefusalCandleEngine()
        self.signal_recorder = signal_recorder or SignalRecorder()

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

        self.features_logger = get_or_setup_logger("unit_runtime_features", "unit_runtime_features.log")
        self.decision_logger = get_or_setup_logger("unit_decision_engine", "unit_decision_engine.log")
        self.shadow_logger = get_or_setup_logger("unit_shadow_mode", "unit_shadow_mode.log")
        self.evaluator_logger = get_or_setup_logger("unit_signal_evaluator", "unit_signal_evaluator.log")

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    self.last_bar_time = json.load(f)
                logger.info(f"Loaded UniT state from {self.state_file}")
            except Exception as e:
                logger.error(f"Failed to load UniT state: {e}")

    def _save_state(self):
        try:
            temp_file = self.state_file + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(self.last_bar_time, f, indent=4)
            safe_file_replace(temp_file, self.state_file)
        except Exception as e:
            logger.error(f"Failed to save UniT state: {e}")

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.warning("UniTStrategy is already running.")
            return

        with self._lock:
            self.signal_history = {s: {"M5": [], "M15": []} for s in self.symbols}
            self._bar_counters = {s: {"M5": 0, "M15": 0} for s in self.symbols}

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("UniTStrategy started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("UniTStrategy stopped.")

    def _poll_loop(self):
        while not self._stop_event.is_set():
            try:
                self._poll_cycle()
            except Exception as e:
                logger.error(f"Error in UniT poll cycle: {e}", exc_info=True)
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
                logger.info(f"[UniT] New bar detected for {symbol} {timeframe}: {df.iloc[-1]['Datetime']}")

                self._evaluate_trend_continuation(symbol, timeframe, df)
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

    def _evaluate_trend_continuation(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Optional[SignalCandidate]:
        idx_closed = len(df) - 2
        bar_closed = df.iloc[idx_closed]
        bar_timestamp = str(bar_closed["Datetime"])

        msg = self._build_market_structure_graph(symbol, timeframe, df)

        # Retrieve engines evaluations
        strong_candle = self.strong_candle_engine.evaluate(df, idx_closed, msg)
        refusal_candle = self.refusal_engine.evaluate_rejection(df, idx_closed, None, msg)

        # Context features
        session_val = "Asian"
        spread_val = float(bar_closed.get("Spread", 0.0))
        account_session_ctx = {"session": session_val, "spread": spread_val}
        strategy_ctx = {"strategy": "unit", "signal_type": "pullback"}

        features_fv = self.feature_pipeline.extract_runtime(
            df=df,
            msg=msg,
            idx=idx_closed,
            account_session_context=account_session_ctx,
            strategy_context=strategy_ctx
        )

        # Predict - only proceed if ML predicted state is TREND or fall back to structural trend
        predicted_state = "UNKNOWN"
        decision_ctx = None
        if self.decision_engine:
            try:
                decision_ctx = self.decision_engine.evaluate(
                    symbol=symbol,
                    timeframe=timeframe,
                    feature_vector=features_fv.features,
                    strategy_name="UniTStrategy",
                    timestamp=bar_timestamp
                )
                predicted_state = decision_ctx.predicted_state
            except Exception as e:
                logger.error(f"UniT inference failed: {e}")

        if predicted_state == "UNKNOWN":
            predicted_state = "TREND" if msg.trend_direction in ["Bull", "Bear"] else "RANGE"

        if predicted_state != "TREND":
            logger.info(f"[UniT] Skip {symbol} {timeframe}: Regime is {predicted_state}, not TREND.")
            return None

        # Standard pullback rules:
        # If Bullish trend: we want a bullish strong candle after a pullback refusal inside a Demand zone or EMA area
        # If Bearish trend: we want a bearish strong candle after a pullback refusal inside a Supply zone or EMA area
        direction = 1 if msg.trend_direction == "Bull" else -1

        # Assess candle metrics dynamically from engines
        has_momentum_candle = strong_candle.quality_score >= self.min_strong_candle_score
        has_pullback_refusal = refusal_candle.quality_score >= self.min_pullback_refusal_score

        # Ensure correct direction alignment
        if direction == 1 and not strong_candle.bullish:
            has_momentum_candle = False
        elif direction == -1 and not strong_candle.bearish:
            has_momentum_candle = False

        if not (has_momentum_candle or has_pullback_refusal):
            return None

        # Build trade setup levels
        entry_price = float(bar_closed["Close"])
        atr_val = msg.atr if msg.atr > 0 else 0.0001
        sl_price = entry_price - direction * (atr_val * 2.0)
        tp_price = entry_price + direction * (atr_val * 3.0)
        rr_ratio = 1.5

        trading_allowed = self.drawdown_manager.trading_allowed()
        risk_state = {
            "trading_allowed": trading_allowed,
            "drawdown_limit_reached": not trading_allowed
        }

        candidate_dict = {
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,
            "signal_type": "pullback",
            "technical_rules_satisfied": True
        }

        evaluation = self.signal_evaluator.evaluate(
            strategy_name="UniTStrategy",
            signal_candidate=candidate_dict,
            feature_vector=features_fv.features,
            decision_context=decision_ctx,
            market_structure=msg,
            supply_demand=None,
            risk_state=risk_state
        )

        extra_fields = {
            "strategy": "unit",
            "strong_candle_score": strong_candle.quality_score,
            "refusal_score": refusal_candle.quality_score,
            "accepted": evaluation.accepted
        }

        # Log to journal
        signal_id = 0
        if self.trading_journal:
            try:
                signal_id = self.trading_journal.log_signal(
                    signal_type="pullback",
                    symbol=symbol,
                    timeframe=timeframe,
                    direction=direction,
                    entry_price=entry_price,
                    sl_price=sl_price,
                    exit_profile=EXIT_PROFILE_STANDARD,
                    strategy="unit",
                    signal_category="pullback",
                    bar_timestamp=bar_timestamp,
                    extra_fields=extra_fields
                )
            except Exception as e:
                logger.error(f"UniT failed to log to journal: {e}")

        # Construct SignalCandidate (Part 4)
        candidate = SignalCandidate(
            signal_id=signal_id if signal_id else int(time.time()),
            strategy_name="UniTStrategy",
            strategy_version="1.1.0",
            symbol=symbol,
            timeframe=timeframe,
            timestamp=bar_timestamp,
            direction=direction,
            signal_type="pullback",
            entry_price=entry_price,
            stop_loss=sl_price,
            take_profit=tp_price,
            risk_reward=rr_ratio,
            market_state="TREND",
            trend=msg.trend_direction,
            signal_quality=strong_candle.quality_score,
            confidence=strong_candle.confidence,
            risk_multiplier=0.01,
            strong_candle_info=asdict(strong_candle),
            refusal_info=asdict(refusal_candle),
            market_structure_snapshot={
                "trend_direction": msg.trend_direction,
                "atr": msg.atr
            },
            supply_demand_snapshot={},
            ml_predictions={
                "predicted_state": predicted_state,
                "state_confidence": decision_ctx.state_confidence if decision_ctx else 0.5,
                "break_probability": decision_ctx.break_probability if decision_ctx else 0.5
            },
            reasoning=f"UniT trend continuation pullback signal verified by momentum={strong_candle.quality_score} and refusal={refusal_candle.quality_score}",
            priority="HIGH",
            status="EXECUTED" if evaluation.accepted and trading_allowed else "REJECTED"
        )

        # Log candidate to unified SignalRecorder (Part 5)
        self.signal_recorder.record_candidate(candidate, rejection_reason=", ".join(evaluation.reasons))

        if evaluation.accepted and trading_allowed:
            logger.info(f"[UniT] Submitting live trend continuation order for {symbol}")
            res = self.send_order.execute(
                symbol=symbol,
                direction=direction,
                entry_price=0.0,
                sl_price=sl_price,
                exit_profile=EXIT_PROFILE_STANDARD,
                strategy="unit",
                signal_category="pullback",
                signal_id=candidate.signal_id,
                tp_price=tp_price
            )
            logger.info(f"[UniT] Order status: {res.get('success')} - {res.get('reason')}")

        return candidate
