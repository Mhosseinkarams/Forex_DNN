import os
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional, List, Any
import pandas as pd
import numpy as np

# Optional MT5 import for environments where it's not installed (e.g. Linux CI)
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

# Modules
from Collecting_Data.indicators import IndicatorEngine
from Collecting_Data.position_lifecycle import EXIT_PROFILE_STANDARD, EXIT_PROFILE_SINGLE, EXIT_PROFILE_HIGH_RISK, EXIT_PROFILE_REVERSAL
from Collecting_Data.utils import safe_file_replace
from Core.trend_context import TrendContext, TrendContextBuilder

# Market Data Pipeline & Trade Location Modules
from Market_Data_Pipeline.structure_graph import MarketStructureGraph, StructureLevel, Zone, BOS, CHOCH
from Market_Data_Pipeline.structure_engine import MarketStructureEngine
from Market_Data_Pipeline.supply_demand_engine import SupplyDemandEngine
from Market_Data_Pipeline.state_engine import MarketStateEngine, StateContext
from Trade_Execution.location_engine import TradeLocationEngine

# Runtime ML and Evaluator Integration
from ML.feature_pipeline import FeaturePipeline
from ML.ml_decision_engine import MLDecisionEngine
from Strategies.signal_evaluator import SignalEvaluator, SignalEvaluation
from ML.trade_feature_recorder import TradeFeatureRecorder

logger = logging.getLogger("MMStrategy")

class MMStrategy:
    """
    Purpose:
        Market Maker (MM) Strategy class implementing rule-based entry setups,
        support/resistance-based take profit (TP) and stop loss (SL), and
        flexible risk management.

    Thread Safety:
        The strategy runs on a background polling thread. All accesses and
        mutations to the shared attributes (e.g., `signal_history`,
        `_bar_counters`, and `last_bar_time`) are synchronized using a
        re-entrant lock `self._lock` to avoid race conditions.

    First Bar Behavior:
        On the very first bar of a session or strategy restart, the method
        `_is_new_bar` will initialize the `last_bar_time` tracker and return
        `False`. No signals will be evaluated or submitted for the first bar
        to ensure indicators are synchronized and stable.

    Shadow Mode and ML Filtering:
        - SHADOW_MODE (default True): Candidate signal evaluations and features
          are recorded to runtime logs for shadow verification, but live order
          submission is solely governed by rule-based constraints.
        - ML_FILTERING (default False): When enabled outside of shadow mode,
          active machine learning model predictions will filter/override rule
          based candidates.
    """
    def __init__(
        self,
        data_feed,                    # DataFeed instance
        send_order,                   # SendOrder instance
        trading_journal,              # TradingJournal instance
        drawdown_manager,             # DrawdownManager instance
        symbols: list[str],           # e.g. ["EURUSD_o", "GBPUSD_o"]
        poll_interval_seconds: float = 5.0,
        swing_lookback: int = 10,
        max_sl_pips: int = 25,
        m5_slope_threshold: float = 0.1,
        m15_slope_threshold: float = 0.1,
        price_to_fast_atr_threshold: float = 1.5,
        fast_to_slow_atr_threshold: float = 3.0,
        reversal_ema_sep_threshold: float = 9.0,
        state_file: str = "mm_strategy_state.json",
        location_engine: Optional[TradeLocationEngine] = None,
        annotator: Optional[Any] = None,
        market_state_model: Optional[Any] = None,
        level_break_model: Optional[Any] = None,
        decision_engine: Optional[MLDecisionEngine] = None,
        feature_pipeline: Optional[FeaturePipeline] = None,
        signal_evaluator: Optional[SignalEvaluator] = None,
        recorder: Optional[TradeFeatureRecorder] = None,
        shadow_mode: bool = True,
        ml_filtering: bool = False,
        hr_body_pct: float = 0.70,
        hr_body_vs_avg: float = 1.2,
        std_body_pct: float = 0.60,
        std_body_vs_avg: float = 1.0,
        rev_body_pct: float = 0.80,
        rev_body_vs_avg: float = 1.5,
        m5_ema_periods: list[int] = [50, 600],
        m15_ema_periods: list[int] = [50, 800]
    ):
        self.data_feed = data_feed
        self.send_order = send_order
        self.trading_journal = trading_journal
        self.drawdown_manager = drawdown_manager
        self.symbols = symbols
        self.poll_interval_seconds = poll_interval_seconds
        self.swing_lookback = swing_lookback
        self.max_sl_pips = max_sl_pips
        self.m5_slope_threshold = m5_slope_threshold
        self.m15_slope_threshold = m15_slope_threshold
        self.price_to_fast_atr_threshold = price_to_fast_atr_threshold
        self.fast_to_slow_atr_threshold = fast_to_slow_atr_threshold
        self.reversal_ema_sep_threshold = reversal_ema_sep_threshold
        self.state_file = state_file
        self.annotator = annotator
        self.market_state_model = market_state_model
        self.level_break_model = level_break_model

        # Config Toggles
        self.shadow_mode = shadow_mode
        self.ml_filtering = ml_filtering

        # Strategy Threshold Parameters
        self.hr_body_pct = hr_body_pct
        self.hr_body_vs_avg = hr_body_vs_avg
        self.std_body_pct = std_body_pct
        self.std_body_vs_avg = std_body_vs_avg
        self.rev_body_pct = rev_body_pct
        self.rev_body_vs_avg = rev_body_vs_avg
        self.m5_ema_periods = m5_ema_periods
        self.m15_ema_periods = m15_ema_periods

        # Initialize injected or default ML services
        self.feature_pipeline = feature_pipeline or FeaturePipeline()
        self.decision_engine = decision_engine or MLDecisionEngine()
        self.signal_evaluator = signal_evaluator or SignalEvaluator(shadow_mode=self.shadow_mode, ml_filtering=self.ml_filtering)
        self.recorder = recorder or TradeFeatureRecorder()

        # Connect recorder to the TradingJournal
        if self.trading_journal is not None:
            self.trading_journal.recorder = self.recorder

        # Set up runtime logger handlers
        self._setup_runtime_loggers()

        self.engine_m5 = IndicatorEngine(
            ema_periods=self.m5_ema_periods,
            slope_period=32
        )
        self.engine_m15 = IndicatorEngine(
            ema_periods=self.m15_ema_periods,
            slope_period=32
        )

        # Re-use analytical engines
        self.struct_engine = MarketStructureEngine(lookback=3)
        self.sd_engine = SupplyDemandEngine(atr_period=14, impulse_threshold=2.0)
        self.state_engine = MarketStateEngine()
        self.location_engine = location_engine or TradeLocationEngine()

        self.last_bar_time = {}  # symbol -> timeframe -> timestamp
        self.signal_history = {s: {"M5": [], "M15": []} for s in symbols}
        self._bar_counters = {s: {"M5": 0, "M15": 0} for s in symbols}

        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

        self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    self.last_bar_time = json.load(f)
                logger.info(f"Loaded state from {self.state_file}")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")

    def _save_state(self):
        try:
            temp_file = self.state_file + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(self.last_bar_time, f, indent=4)
            safe_file_replace(temp_file, self.state_file)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def start(self) -> None:
        """
        Purpose:
            Initializes and starts the background strategy thread.
            The thread will continuously poll the market data feed for
            new signal opportunities.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning("MMStrategy is already running.")
            return
        
        # Reset signal history on start
        with self._lock:
            self.signal_history = {s: {"M5": [], "M15": []} for s in self.symbols}
            self._bar_counters = {s: {"M5": 0, "M15": 0} for s in self.symbols}
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("MMStrategy started.")

    def stop(self) -> None:
        """
        Purpose:
            Gracefully terminates the strategy polling thread.
        """
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("MMStrategy stopped.")

    def _poll_loop(self):
        while not self._stop_event.is_set():
            try:
                self._poll_cycle()
            except Exception as e:
                logger.error(f"Error in poll cycle: {e}", exc_info=True)
            
            time.sleep(self.poll_interval_seconds)

    def _poll_cycle(self):
        """
        Purpose:
            The main iterative unit of the strategy. Performs bar-time
            synchronization, indicator calculation, and signal evaluation
            for all configured symbols and timeframes.
        """
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
                
                # New bar detected
                with self._lock:
                    self._bar_counters[symbol][timeframe] += 1
                logger.info(f"New bar detected for {symbol} {timeframe}: {df.iloc[-1]['Datetime']}")
                
                # Active chart annotation
                if self.annotator:
                    try:
                        msg = self._build_market_structure_graph(symbol, timeframe, df)
                        state_ctx = self.state_engine.evaluate(msg)

                        # Generate ml output for the passive annotator if models exist
                        ml_out = {}
                        if self.market_state_model is not None or self.level_break_model is not None:
                            try:
                                idx_closed = -2
                                from ML.feature_pipeline import FeaturePipeline
                                # Try with market_state_model registry or default
                                reg = self.market_state_model.registry if self.market_state_model else self.level_break_model.registry
                                fp_pipeline = FeaturePipeline(reg)
                                feats = fp_pipeline.extract_all(df, msg, idx=idx_closed)
                                features_df = pd.DataFrame([feats])

                                if self.market_state_model is not None:
                                    s_prob = self.market_state_model.predict_proba(features_df)
                                    ml_out["trend_prob"] = s_prob.get("TREND", 0.0)
                                    ml_out["range_prob"] = s_prob.get("RANGE", 0.0)
                                    ml_out["transition_prob"] = s_prob.get("TRANSITION", 0.0)
                                if self.level_break_model is not None:
                                    b_prob = self.level_break_model.predict_proba(features_df)
                                    ml_out["break_prob"] = b_prob.get("BREAK", 0.0)
                                    ml_out["reject_prob"] = b_prob.get("REJECT", 0.0)
                                    ml_out["confidence"] = b_prob.get("confidence", 0.0)
                            except Exception:
                                pass

                        self.annotator.render(
                            symbol=symbol,
                            structure_graph=msg,
                            state_context=state_ctx,
                            ml_output=ml_out
                        )
                    except Exception as ex:
                        logger.error(f"Failed to passively annotate chart: {ex}")

                self._check_and_submit_signal(symbol, timeframe, df, fast_p, slow_p)
                self._save_state()

    def _setup_runtime_loggers(self):
        """Configure specialized handlers for runtime logging."""
        # Use an absolute directory relative to the repository / module location
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

        self.features_logger = get_or_setup_logger("runtime_features", "runtime_features.log")
        self.decision_logger = get_or_setup_logger("decision_engine", "decision_engine.log")
        self.shadow_logger = get_or_setup_logger("shadow_mode", "shadow_mode.log")
        self.evaluator_logger = get_or_setup_logger("signal_evaluator", "signal_evaluator.log")

    def _is_new_bar(self, symbol, timeframe, df):
        current_bar_time = str(df.iloc[-1]["Datetime"])
        with self._lock:
            if symbol not in self.last_bar_time:
                self.last_bar_time[symbol] = {}
            
            if timeframe not in self.last_bar_time[symbol]:
                self.last_bar_time[symbol][timeframe] = current_bar_time
                return False # First time seeing this, don't trigger signal yet
            
            if self.last_bar_time[symbol][timeframe] != current_bar_time:
                self.last_bar_time[symbol][timeframe] = current_bar_time
                return True
        
        return False

    def _build_market_structure_graph(self, symbol: str, timeframe: str, df: pd.DataFrame) -> MarketStructureGraph:
        """
        Build and populate a MarketStructureGraph using the structural engines.
        """
        # Run analytical engines on the dataframe
        df_struct = self.struct_engine.process(df)
        df_sd = self.sd_engine.process(df_struct)

        last_row = df_sd.iloc[-1]
        dt = pd.to_datetime(last_row["Datetime"]) if "Datetime" in df_sd.columns else datetime.now(timezone.utc)

        # Pre-filter swing levels to avoid assigning duplicates initially
        swing_highs_list = [s for s in self.struct_engine.swings if s.level_type == 'SwingHigh']
        swing_lows_list = [s for s in self.struct_engine.swings if s.level_type == 'SwingLow']

        # Create MarketStructureGraph instance
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
            volatility=float(last_row.get("atr_14", 0.0001) * 10000.0) # Relative scaling
        )

        return graph

    def _check_and_submit_signal(self, symbol, timeframe, df, fast_p, slow_p):
        idx_closed = -2
        idx_forming = -1
        
        bar_closed = df.iloc[idx_closed]
        bar_forming = df.iloc[idx_forming]
        
        # Check if candle crossed EMA50 (the entry trigger)
        cross_fast_col = f"cross_ema_{fast_p}"
        cross_fast_val = bar_closed[cross_fast_col]
        if cross_fast_val == 0:
            return

        # Build TrendContext at forming bar (index -1)
        slope_threshold = self.m5_slope_threshold if timeframe == "M5" else self.m15_slope_threshold
        builder = TrendContextBuilder(slope_threshold=slope_threshold)
        trend_context = builder.build(symbol, timeframe, df, idx=idx_forming)

        dist_fast_val = bar_forming[f"dist_ema_{fast_p}"]

        # Priority: HR -> STD -> REV
        
        # 1. High-Risk
        hr_dir = self._evaluate_high_risk(bar_closed, trend_context)
        if hr_dir:
            self._process_signal(symbol, timeframe, "high_risk", hr_dir, df, trend_context)
            return

        # 2. Standard
        std_dir = self._evaluate_standard(bar_closed, trend_context, dist_fast_val)
        if std_dir:
            self._process_signal(symbol, timeframe, "standard", std_dir, df, trend_context)
            return

        # 3. Reversal
        rev_dir = self._evaluate_reversal(bar_closed, trend_context)
        if rev_dir:
            self._process_signal(symbol, timeframe, "reversal", rev_dir, df, trend_context)
            return

    def _evaluate_ema_cross_alignment(self, cross_fast_val, trend_direction, expect_opposite=False):
        """
        Helper method to refactor duplicated EMA-cross checks.
        Ensures cross_fast_val matches/opposes trend_direction correctly.
        """
        if cross_fast_val == 0:
            return None
        if not expect_opposite:
            if cross_fast_val == 1 and trend_direction == "Bull":
                return 1
            elif cross_fast_val == -1 and trend_direction == "Bear":
                return -1
        else:
            if cross_fast_val == 1 and trend_direction == "Bear":
                return 1
            elif cross_fast_val == -1 and trend_direction == "Bull":
                return -1
        return None

    def _evaluate_high_risk(self, bar_closed, trend_context):
        # 1. Previous candle crosses through fast EMA
        cross_fast_val = bar_closed.get("cross_ema_50", 0)
        
        # 2. Cross direction aligns with slow EMA trend (Trend Direction Context)
        direction = self._evaluate_ema_cross_alignment(cross_fast_val, trend_context.trend_direction)
        if direction is None:
            return None
        
        # 3. Previous candle body percentage
        if bar_closed["body_pct"] < self.hr_body_pct:
            return None
        
        # 4. Previous candle size vs average
        if bar_closed["body_vs_avg"] < self.hr_body_vs_avg:
            return None
        
        # 5. Slow EMA slope
        slope_threshold = self.m5_slope_threshold if trend_context.timeframe == "M5" else self.m15_slope_threshold
        if trend_context.ema_slope < slope_threshold:
            return None
        
        return direction

    def _evaluate_standard(self, bar_closed, trend_context, dist_fast_val=None):
        cross_fast_val = bar_closed.get("cross_ema_50", 0)

        # 1. EMA alignment (Trend Context)
        direction = self._evaluate_ema_cross_alignment(cross_fast_val, trend_context.trend_direction)
        if direction is None:
            return None

        # 2. Candle crossing EMA50 in trend direction (Entry Trigger)
        if cross_fast_val != direction:
            return None

        # 3. Price proximity to fast EMA (ATR-dynamic)
        if dist_fast_val is None:
            dist_fast_val = bar_closed.get("dist_ema_50", 0.0)
        if abs(dist_fast_val) >= self.price_to_fast_atr_threshold:
            return None
        
        # 4. EMA separation (ATR-dynamic)
        if trend_context.ema_distance_atr >= self.fast_to_slow_atr_threshold:
            return None
        
        # 5. Previous candle body percentage
        if bar_closed["body_pct"] < self.std_body_pct:
            return None
        
        # 6. Previous candle size vs average
        if bar_closed["body_vs_avg"] <= self.std_body_vs_avg:
            return None
        
        # 7. Slow EMA slope
        slope_threshold = self.m5_slope_threshold if trend_context.timeframe == "M5" else self.m15_slope_threshold
        if trend_context.ema_slope < slope_threshold:
            return None
        
        # 8. Direction match (candle confirms EMA direction)
        if bar_closed["candle_direction"] != direction:
            return None
        
        return direction

    def _evaluate_reversal(self, bar_closed, trend_context):
        # 1. EMA separation is large
        if trend_context.ema_distance_atr < self.reversal_ema_sep_threshold:
            return None
        
        # 2. Previous candle crosses through fast EMA (Entry Trigger)
        cross_fast_val = bar_closed.get("cross_ema_50", 0)
        
        # 3. Cross direction is OPPOSITE to slow EMA trend (Trend Direction Context)
        direction = self._evaluate_ema_cross_alignment(cross_fast_val, trend_context.trend_direction, expect_opposite=True)
        if direction is None:
            return None
            
        # 4. Previous candle body percentage
        if bar_closed["body_pct"] < self.rev_body_pct:
            return None
        
        # 5. Previous candle size vs average
        if bar_closed["body_vs_avg"] < self.rev_body_vs_avg:
            return None
        
        return direction

    def _process_signal(self, symbol, timeframe, signal_type, direction, df, trend_context):
        # bar_timestamp is the Datetime of the signal bar (bar[-2])
        bar_timestamp = str(df.iloc[-2]["Datetime"])
        
        # Default trade parameters
        if signal_type == "standard":
            exit_profile = EXIT_PROFILE_STANDARD
            risk_pct_default = 0.01
        elif signal_type == "high_risk":
            exit_profile = EXIT_PROFILE_HIGH_RISK
            risk_pct_default = 0.005
        else: # reversal
            exit_profile = EXIT_PROFILE_REVERSAL
            risk_pct_default = 0.003

        # SL Calculation using shared engines and TradeLocationEngine
        sl_price = self._calculate_sl(symbol, direction, df)
        
        # Live ask/bid for entry_price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None or tick.ask <= 0 or tick.bid <= 0:
            logger.error(f"Failed to fetch valid tick for {symbol}: {tick}")
            return
        entry_price = float(tick.ask if direction == 1 else tick.bid)

        # --- RUNTIME FEATURE PIPELINE AND INFERENCE FLOW ---
        msg = self._build_market_structure_graph(symbol, timeframe, df)

        session_val = str(msg.session) if hasattr(msg, "session") else "Asian"
        spread_val = float(df.iloc[-2].get("Spread", 0.0))
        account_session_ctx = {
            "session": session_val,
            "spread": spread_val
        }
        strategy_ctx = {
            "signal_direction": direction,
            "signal_type": signal_type
        }

        # 1. Extract exactly the same features used during training
        features_fv = self.feature_pipeline.extract_runtime(
            df=df,
            msg=msg,
            idx=-2,
            account_session_context=account_session_ctx,
            strategy_context=strategy_ctx
        )

        # Log runtime feature vector
        self.features_logger.info(f"Features for candidate signal {bar_timestamp}: {features_fv.features}")

        # 2. Query MLDecisionEngine
        decision_ctx = None
        if self.decision_engine:
            try:
                decision_ctx = self.decision_engine.evaluate(
                    symbol=symbol,
                    timeframe=timeframe,
                    feature_vector=features_fv.features,
                    strategy_name="MMStrategy",
                    timestamp=bar_timestamp
                )
                self.decision_logger.info(
                    f"Decision Engine output: State={decision_ctx.predicted_state}, "
                    f"BreakProb={decision_ctx.break_probability:.4f}, "
                    f"QualityScore={decision_ctx.trade_quality_score:.4f}"
                )
            except Exception as e:
                logger.error(f"MLDecisionEngine inference failure: {e}", exc_info=True)

        # 3. Retrieve risk/drawdown state
        trading_allowed = self.drawdown_manager.trading_allowed()
        risk_state = {
            "trading_allowed": trading_allowed,
            "drawdown_limit_reached": not trading_allowed
        }

        # 4. Invoke unified SignalEvaluator
        candidate_dict = {
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,
            "signal_type": signal_type,
            "technical_rules_satisfied": True
        }

        evaluation = self.signal_evaluator.evaluate(
            strategy_name="MMStrategy",
            signal_candidate=candidate_dict,
            feature_vector=features_fv.features,
            decision_context=decision_ctx,
            market_structure=msg,
            supply_demand=None,
            risk_state=risk_state
        )

        # Log to signal evaluator
        self.evaluator_logger.info(
            f"Evaluation result: Accepted={evaluation.accepted}, Priority={evaluation.priority}, "
            f"Reasons={evaluation.reasons}, Warnings={evaluation.warnings}"
        )

        # Log candidate details in Shadow Mode format
        pred_state = decision_ctx.predicted_state if decision_ctx else "TRANSITION"
        state_conf = decision_ctx.state_confidence if decision_ctx else 0.5
        break_prob = decision_ctx.break_probability if decision_ctx else 0.5
        trade_qual = decision_ctx.trade_quality_score if decision_ctx else 0.5
        policy_rec = str(decision_ctx.policy_recommendation) if decision_ctx else "None"

        shadow_log_msg = (
            f"SHADOW_MODE_CANDIDATE | Time: {bar_timestamp} | Symbol: {symbol} | Timeframe: {timeframe} | "
            f"Direction: {'BUY' if direction == 1 else 'SELL'} | MarketStatePred: {pred_state} ({state_conf:.4f}) | "
            f"BreakProb: {break_prob:.4f} | TradeQuality: {trade_qual:.4f} | "
            f"PolicyRec: {policy_rec} | FinalDecision: {'Accepted' if evaluation.accepted else 'Rejected'}"
        )
        self.shadow_logger.info(shadow_log_msg)

        # Active chart annotation for signal levels
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

        if self.annotator:
            try:
                state_ctx = self.state_engine.evaluate(msg)

                decision_dict = {
                    "direction": direction,
                    "accepted": evaluation.accepted,
                    "reason": signal_type,
                    "signal_type": signal_type,
                    "strategy": "mm"
                }

                self.annotator.render(
                    symbol=symbol,
                    structure_graph=msg,
                    state_context=state_ctx,
                    trade_plan={"entry_price": entry_price, "sl_price": sl_price, "tp_price": entry_price + direction * (abs(entry_price - sl_price) * 2.0)},
                    decision=decision_dict,
                    ml_output=ml_render_data
                )
            except Exception as ex:
                logger.error(f"Failed to passively annotate signal: {ex}")
        
        # Distance/technical metrics for ML
        extra_fields = self._get_signal_distances(symbol, timeframe, signal_type, direction)
        
        # Add indicator and trend context values to extra_fields
        idx_closed = -2
        idx_forming = -1
        atr_val = float(df.iloc[idx_forming]["atr_14"])
        
        extra_fields.update({
            "trend_direction": trend_context.trend_direction,
            "trend_strength": trend_context.trend_strength,
            "is_strong_trend": trend_context.is_strong_trend,
            "is_weak_trend": trend_context.is_weak_trend,
            "bars_since_cross": trend_context.bars_since_cross,
            "bars_since_trend_change": trend_context.bars_since_trend_change,
            "ema_fast": trend_context.ema_fast,
            "ema_slow": trend_context.ema_slow,
            "ema_slope": trend_context.ema_slope,
            "ema_distance": trend_context.ema_distance,
            "ema_separation_atr": trend_context.ema_distance_atr,
            "atr": atr_val,
            "body_pct": float(df.iloc[idx_closed]["body_pct"]),
            "body_vs_avg": float(df.iloc[idx_closed]["body_vs_avg"]),
            "risk_pct_default": risk_pct_default,
        })

        if not trading_allowed:
            extra_fields["blocked_by_drawdown"] = True
            logger.info(f"Signal {signal_type} {direction} for {symbol} BLOCKED by drawdown")
        
        # Print a snapshot of the context to live logs
        ema_slope_dir_text = "Positive" if trend_context.trend_direction == "Bull" else "Negative"
        snapshot = (
            f"\n--- Trend Context ---\n"
            f"Direction : {trend_context.trend_direction}\n"
            f"Strength : {trend_context.trend_strength}\n"
            f"EMA Distance : {trend_context.ema_distance_atr:.1f} ATR\n"
            f"EMA Slope : {ema_slope_dir_text}\n"
            f"Bars Since Cross : {trend_context.bars_since_cross}\n"
            f"Bars Since Trend Change : {trend_context.bars_since_trend_change}\n"
            f"---------------------"
        )
        logger.info(snapshot)

        # Log to journal (Layer 1 Event)
        signal_id = self.trading_journal.log_signal(
            signal_type=signal_type,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            entry_price=entry_price,
            sl_price=sl_price,
            exit_profile=exit_profile,
            strategy="mm",
            signal_category=signal_type,
            bar_timestamp=bar_timestamp,
            extra_fields=extra_fields
        )
        
        # Log to TradeFeatureRecorder if registered
        if self.recorder:
            try:
                self.recorder.record_candidate(
                    signal_id=signal_id,
                    timestamp=bar_timestamp,
                    strategy="mm",
                    symbol=symbol,
                    timeframe=timeframe,
                    direction="BUY" if direction == 1 else "SELL",
                    features=features_fv.features,
                    decision_context=decision_ctx,
                    accepted=evaluation.accepted,
                    reason=", ".join(evaluation.reasons)
                )
            except Exception as rec_ex:
                logger.error(f"TradeFeatureRecorder record_candidate failed: {rec_ex}")

        # Update history
        self._update_signal_history(symbol, timeframe, signal_type, direction)
        
        # Continue using existing MM rules (ML-Filtering has no trade execution influence in Shadow Mode)
        # We obey technical_rules and trading_allowed
        if trading_allowed:
            logger.info(f"Submitting {signal_type} {direction} for {symbol} {timeframe}")
            res = self.send_order.execute(
                symbol=symbol,
                direction=direction,
                entry_price=0.0, # Market order
                sl_price=sl_price,
                exit_profile=exit_profile,
                strategy="mm",
                signal_category=signal_type,
                signal_id=signal_id
            )
            logger.info(f"Order result for {symbol}: {res.get('success')} - {res.get('reason')}")

    def _calculate_sl(self, symbol, direction, df):
        # 1. Fetch current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            # Fallback to last close
            entry_price = float(df.iloc[-1]["Close"])
        else:
            entry_price = float(tick.ask if direction == 1 else tick.bid)

        # 2. Build the current bar's MarketStructureGraph
        msg = self._build_market_structure_graph(symbol, "M5", df)

        # 3. Resolve base SL price from TradeLocationEngine using structural coordinates
        levels = self.location_engine.get_trade_levels(msg, direction, entry_price)
        sl_price = levels["sl_price"]

        # 4. Apply pip distance and broker constraints exactly as required by MMStrategy
        info = mt5.symbol_info(symbol)
        if info is None:
            return float(sl_price)
        
        pip_size = info.point * 10
        max_sl_dist = self.max_sl_pips * pip_size
        
        current_dist = abs(entry_price - sl_price)
        if current_dist > max_sl_dist:
            sl_price = entry_price - direction * max_sl_dist
            logger.warning(f"SL capped for {symbol} at {self.max_sl_pips} pips")

        # Minimum SL distance
        stops_level_price = info.trade_stops_level * info.point
        if abs(entry_price - sl_price) < stops_level_price:
            sl_price = entry_price - direction * (stops_level_price + info.point)
            
        return float(sl_price)

    def _get_signal_distances(self, symbol, timeframe, signal_type, direction):
        with self._lock:
            history = self.signal_history.get(symbol, {}).get(timeframe, [])
            count = self._bar_counters[symbol][timeframe]
        
        def find_last(type_filter=None, dir_filter=None):
            for sig in reversed(history):
                if type_filter and sig["type"] != type_filter: continue
                if dir_filter and sig["direction"] != dir_filter: continue
                return sig
            return None

        last_std = find_last(type_filter="standard", dir_filter=direction)
        last_hr = find_last(type_filter="high_risk", dir_filter=direction)
        last_any = find_last(dir_filter=direction)
        
        dist_std = count - last_std["count"] if last_std else -1
        dist_hr = count - last_hr["count"] if last_hr else -1
        dist_any = count - last_any["count"] if last_any else -1

        return {
            f"bars_since_last_standard_{direction}": dist_std,
            f"bars_since_last_high_risk_{direction}": dist_hr,
            "bars_since_last_any_signal": dist_any
        }

    def _update_signal_history(self, symbol, timeframe, signal_type, direction):
        with self._lock:
            count = self._bar_counters[symbol][timeframe]
            self.signal_history[symbol][timeframe].append({
                "type": signal_type,
                "direction": direction,
                "count": count
            })

if __name__ == "__main__":
    import unittest
    from unittest.mock import MagicMock, patch
    import sys

    # Mock MT5 for testing
    mock_mt5 = MagicMock()
    sys.modules["MetaTrader5"] = mock_mt5
    import MetaTrader5 as mt5

    class TestMMStrategy(unittest.TestCase):
        def setUp(self):
            self.data_feed = MagicMock()
            self.send_order = MagicMock()
            self.trading_journal = MagicMock()
            self.drawdown_manager = MagicMock()
            self.symbols = ["EURUSD_o"]
            self.state_file = "test_mm_state.json"
            
            self.strategy = MMStrategy(
                self.data_feed,
                self.send_order,
                self.trading_journal,
                self.drawdown_manager,
                self.symbols,
                state_file=self.state_file
            )
            # Initialize internal structures normally done in start()
            self.strategy.signal_history = {s: {"M5": [], "M15": []} for s in self.symbols}
            self.strategy._bar_counters = {s: {"M5": 0, "M15": 0} for s in self.symbols}
            
            # Default MT5 mocks
            mt5.symbol_info_tick.return_value = MagicMock(ask=1.1000, bid=1.0990)
            mt5.symbol_info.return_value = MagicMock(point=0.00001, trade_stops_level=0)
            self.drawdown_manager.trading_allowed.return_value = True

        def tearDown(self):
            if os.path.exists(self.state_file):
                os.remove(self.state_file)

        def make_df(self, n_bars=850, ema_fast_above_slow=True, bullish_candles=True):
            prices = 1.1000 + np.linspace(0, 0.01 if ema_fast_above_slow else -0.01, n_bars)
            df = pd.DataFrame({
                "Datetime": pd.date_range("2024-01-01", periods=n_bars, freq="5min"),
                "Open": prices,
                "High": prices + 0.0005,
                "Low": prices - 0.0005,
                "Close": prices,
                "TickVolume": 100,
                "Spread": 1
            })
            if bullish_candles:
                df["Open"] = df["Close"] - 0.0007
                df["High"] = df["Close"] + 0.0001
                df["Low"] = df["Open"] - 0.0001
            else:
                df["Open"] = df["Close"] + 0.0007
                df["High"] = df["Open"] + 0.0001
                df["Low"] = df["Close"] - 0.0001
            return df

        def test_standard_buy_signal(self):
            # Test Case 1 & 2: Standard BUY/SELL
            df_raw = self.make_df(ema_fast_above_slow=True, bullish_candles=True)
            self.strategy._is_new_bar = MagicMock(return_value=True)
            self.send_order.execute.return_value = {"success": True}
            
            df = self.strategy.engine_m5.calculate(df_raw)
            df.loc[df.index[-2], "cross_ema_50"] = 1
            with patch.object(self.strategy, '_evaluate_standard', return_value=1):
                self.strategy._check_and_submit_signal("EURUSD_o", "M5", df, 50, 600)
                
            self.trading_journal.log_signal.assert_called_once()
            self.assertEqual(self.trading_journal.log_signal.call_args[1]["signal_type"], "standard")
            self.assertEqual(self.trading_journal.log_signal.call_args[1]["direction"], 1)
            self.send_order.execute.assert_called_once()
            self.assertEqual(self.send_order.execute.call_args[1]["exit_profile"], EXIT_PROFILE_STANDARD)

        def test_high_risk_buy_signal(self):
            # Test Case 3: High-Risk BUY
            df_raw = self.make_df()
            self.strategy._is_new_bar = MagicMock(return_value=True)
            
            df = self.strategy.engine_m5.calculate(df_raw)
            df.loc[df.index[-2], "cross_ema_50"] = 1
            with patch.object(self.strategy, '_evaluate_high_risk', return_value=1):
                self.strategy._check_and_submit_signal("EURUSD_o", "M5", df, 50, 600)
                
            self.assertEqual(self.trading_journal.log_signal.call_args[1]["signal_type"], "high_risk")
            self.assertEqual(self.send_order.execute.call_args[1]["exit_profile"], EXIT_PROFILE_HIGH_RISK)

        def test_reversal_sell_signal(self):
            # Test Case 4: Reversal SELL
            df_raw = self.make_df()
            self.strategy._is_new_bar = MagicMock(return_value=True)
            
            df = self.strategy.engine_m5.calculate(df_raw)
            df.loc[df.index[-2], "cross_ema_50"] = -1
            with patch.object(self.strategy, '_evaluate_reversal', return_value=-1):
                self.strategy._check_and_submit_signal("EURUSD_o", "M5", df, 50, 600)
                
            self.assertEqual(self.trading_journal.log_signal.call_args[1]["signal_type"], "reversal")
            self.assertEqual(self.trading_journal.log_signal.call_args[1]["direction"], -1)

        def test_signal_priority(self):
            # Test Case 5: Signal priority
            df_raw = self.make_df()
            self.strategy._is_new_bar = MagicMock(return_value=True)
            
            df = self.strategy.engine_m5.calculate(df_raw)
            df.loc[df.index[-2], "cross_ema_50"] = 1
            with patch.object(self.strategy, '_evaluate_high_risk', return_value=1), \
                 patch.object(self.strategy, '_evaluate_standard', return_value=1), \
                 patch.object(self.strategy, '_evaluate_reversal', return_value=1):
                self.strategy._check_and_submit_signal("EURUSD_o", "M5", df, 50, 600)
                
            self.trading_journal.log_signal.assert_called_once()
            self.assertEqual(self.trading_journal.log_signal.call_args[1]["signal_type"], "high_risk")

        def test_no_signal_ema_alignment(self):
            # Test Case 6: No signal when EMA alignment fails (candle direction mismatch)
            df_raw = self.make_df(ema_fast_above_slow=True, bullish_candles=False) # Bearish candle in Bullish trend
            self.strategy._is_new_bar = MagicMock(return_value=True)
            
            df = self.strategy.engine_m5.calculate(df_raw)
            df.loc[df.index[-2], "cross_ema_50"] = 1
            self.strategy._check_and_submit_signal("EURUSD_o", "M5", df, 50, 600)
            self.trading_journal.log_signal.assert_not_called()

        def test_no_signal_repeated_bar(self):
            # Test Case 7: No signal on repeated bar
            df_raw = self.make_df()
            self.data_feed.get_ohlcv.return_value = df_raw
            
            self.strategy._poll_cycle() # First call initializes
            self.strategy._poll_cycle() # Second call same bar
            self.trading_journal.log_signal.assert_not_called()

        def test_drawdown_blocked(self):
            # Test Case 8: Drawdown blocked
            df_raw = self.make_df()
            self.strategy._is_new_bar = MagicMock(return_value=True)
            self.drawdown_manager.trading_allowed.return_value = False
            
            df = self.strategy.engine_m5.calculate(df_raw)
            df.loc[df.index[-2], "cross_ema_50"] = 1
            with patch.object(self.strategy, '_evaluate_standard', return_value=1):
                self.strategy._check_and_submit_signal("EURUSD_o", "M5", df, 50, 600)
                
            self.trading_journal.log_signal.assert_called_once()
            self.assertTrue(self.trading_journal.log_signal.call_args[1]["extra_fields"]["blocked_by_drawdown"])
            self.send_order.execute.assert_not_called()

        def test_sl_capped(self):
            # Test Case 9: SL capped at 25 pips
            df_raw = self.make_df()
            df_raw.loc[df_raw.index[-10:-1], "Low"] = 1.0000 
            sl = self.strategy._calculate_sl("EURUSD_o", 1, df_raw)
            self.assertAlmostEqual(sl, 1.0975) # 1.1000 - 25 * 0.0001

        def test_m15_slope_800(self):
            # Test Case 10: M15 uses ema_slope_800
            df_raw = self.make_df()
            df = self.strategy.engine_m15.calculate(df_raw)
            self.assertIn("ema_slope_800", df.columns)

        def test_signal_distance_tracking(self):
            # Test Case 11: Signal distance tracking
            self.strategy._bar_counters["EURUSD_o"]["M5"] = 100
            self.strategy.signal_history["EURUSD_o"]["M5"] = [{"type": "standard", "direction": 1, "count": 90}]
            dists = self.strategy._get_signal_distances("EURUSD_o", "M5", "standard", 1)
            self.assertEqual(dists["bars_since_last_standard_1"], 10)

        def test_extra_fields_logged(self):
            # Test Case 12: extra_fields logged correctly
            df_raw = self.make_df()
            self.strategy._is_new_bar = MagicMock(return_value=True)

            df = self.strategy.engine_m5.calculate(df_raw)
            df.loc[df.index[-2], "cross_ema_50"] = 1
            with patch.object(self.strategy, '_evaluate_standard', return_value=1):
                self.strategy._check_and_submit_signal("EURUSD_o", "M5", df, 50, 600)
            
            fields = self.trading_journal.log_signal.call_args[1]["extra_fields"]
            required = ["ema_fast", "ema_slow", "atr", "body_pct", "body_vs_avg", "ema_slope", 
                        "ema_separation_atr", "risk_pct_default", "bars_since_last_standard_1", 
                        "bars_since_last_high_risk_1", "bars_since_last_any_signal"]
            for r in required:
                self.assertIn(r, fields)

    unittest.main()
