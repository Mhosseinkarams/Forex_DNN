import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Optional MT5 import
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from Market_Data_Pipeline.structure_graph import MarketStructureGraph
from Market_Data_Pipeline.state_engine import StateContext
from Visualization.debug_config import DebugConfig
from Visualization.render_types import DrawInstruction
from Visualization.draw_instruction_writer import DrawInstructionWriter
from Market_Data_Pipeline.strong_candle_engine import StrongCandleEngine
from Market_Data_Pipeline.refusal_candle_engine import RefusalCandleEngine

logger = logging.getLogger("VisualizationEngine")

class ChartAnnotationEngine:
    """
    Purpose:
        The passive ChartAnnotationEngine builds structural, state, levels, signal,
        and ML visualization drawing instructions as DrawInstruction objects,
        which are then written to independent CSV files per symbol via DrawInstructionWriter.
        It also supports Matplotlib overlay for passive interactive debugging in research notebooks.
    """
    def __init__(self, config: Optional[DebugConfig] = None):
        self.config = config or DebugConfig()

    def _format_timestamp(self, ts) -> str:
        if ts is None or (isinstance(ts, float) and np.isnan(ts)):
            return ""
        if hasattr(ts, "strftime"):
            try:
                return ts.strftime("%Y.%m.%d %H:%M:%S")
            except Exception:
                pass
        try:
            pd_ts = pd.Timestamp(ts)
            return pd_ts.strftime("%Y.%m.%d %H:%M:%S")
        except Exception:
            return str(ts)

    def render(
        self,
        symbol: str,
        structure_graph: MarketStructureGraph,
        state_context: Optional[StateContext] = None,
        trade_plan: Optional[Dict[str, Any]] = None,
        decision: Optional[Dict[str, Any]] = None,
        ml_output: Optional[Dict[str, Any]] = None
    ):
        """
        Main API called by the trading engine/strategy.
        Determines which CSV files need to be written or updated based on DebugConfig layers.
        """
        # Determine target directory
        target_dir = None
        if mt5 is not None:
            try:
                info = mt5.terminal_info()
                if info and hasattr(info, "data_path") and info.data_path:
                    target_dir = os.path.join(info.data_path, "MQL5", "Files")
            except Exception as e:
                logger.debug(f"Could not fetch MT5 terminal info: {e}")

        if not target_dir:
            # Fallback for offline/backtest mode
            target_dir = os.path.join("output", "Files")

        os.makedirs(target_dir, exist_ok=True)

        # 1. Swings and Structure Breaks (BOS, CHOCH) -> EURUSD_structure.csv
        if self.config.is_enabled("swings") or self.config.is_enabled("structure"):
            struct_insts = []

            # Swings
            if self.config.is_enabled("swings"):
                for i, sh in enumerate(structure_graph.swing_highs):
                    time_str = self._format_timestamp(sh.timestamp) if sh.timestamp else ""
                    color = "Orange" if getattr(sh, "structure_type", "Major") == "Minor" else ("Magenta" if getattr(sh, "structure_type", "Major") == "Internal" else "Red")
                    struct_insts.append(DrawInstruction(
                        type_name="SWING",
                        name=f"FXDNN_SWING_H_{i}",
                        time1=time_str,
                        price1=f"{sh.price:.5f}",
                        color=color,
                        style="ArrowDown",
                        text=f"H{sh.index}:{sh.price:.5f}({getattr(sh, 'structure_type', 'Major')})"
                    ))
                for i, sl in enumerate(structure_graph.swing_lows):
                    time_str = self._format_timestamp(sl.timestamp) if sl.timestamp else ""
                    color = "LightGreen" if getattr(sl, "structure_type", "Major") == "Minor" else ("Cyan" if getattr(sl, "structure_type", "Major") == "Internal" else "Green")
                    struct_insts.append(DrawInstruction(
                        type_name="SWING",
                        name=f"FXDNN_SWING_L_{i}",
                        time1=time_str,
                        price1=f"{sl.price:.5f}",
                        color=color,
                        style="ArrowUp",
                        text=f"L{sl.index}:{sl.price:.5f}({getattr(sl, 'structure_type', 'Major')})"
                    ))

                if structure_graph.protected_high:
                    ph = structure_graph.protected_high
                    time_str = self._format_timestamp(ph.timestamp) if ph.timestamp else ""
                    struct_insts.append(DrawInstruction(
                        type_name="LEVEL",
                        name="FXDNN_PROTECTED_HIGH",
                        time1=time_str,
                        price1=f"{ph.price:.5f}",
                        color="DarkRed",
                        style="Solid",
                        text="Protected High"
                    ))
                if structure_graph.protected_low:
                    pl = structure_graph.protected_low
                    time_str = self._format_timestamp(pl.timestamp) if pl.timestamp else ""
                    struct_insts.append(DrawInstruction(
                        type_name="LEVEL",
                        name="FXDNN_PROTECTED_LOW",
                        time1=time_str,
                        price1=f"{pl.price:.5f}",
                        color="DarkGreen",
                        style="Solid",
                        text="Protected Low"
                    ))

            # Structure Breaks
            if self.config.is_enabled("structure"):
                for i, b in enumerate(structure_graph.bos):
                    time_str = self._format_timestamp(b.timestamp) if b.timestamp else ""
                    color = "Blue" if b.direction == 1 else "Magenta"
                    dir_text = "Bullish" if b.direction == 1 else "Bearish"
                    struct_insts.append(DrawInstruction(
                        type_name="BOS",
                        name=f"FXDNN_BOS_{i}",
                        time1=time_str,
                        price1=f"{b.broken_level:.5f}",
                        color=color,
                        style="Dash",
                        text=f"BOS ({dir_text})"
                    ))
                for i, c in enumerate(structure_graph.choch):
                    time_str = self._format_timestamp(c.timestamp) if c.timestamp else ""
                    color = "Cyan" if c.new_trend == 1 else "Orange"
                    dir_text = "Bullish" if c.new_trend == 1 else "Bearish"
                    struct_insts.append(DrawInstruction(
                        type_name="CHOCH",
                        name=f"FXDNN_CHOCH_{i}",
                        time1=time_str,
                        price1=f"{c.price:.5f}",
                        color=color,
                        style="DashDot",
                        text=f"CHOCH ({dir_text})"
                    ))

            DrawInstructionWriter.write_instructions(
                os.path.join(target_dir, f"{symbol}_structure.csv"),
                struct_insts
            )

        # 2. Supply & Demand Zones (Range Boundaries) -> EURUSD_zones.csv
        if self.config.is_enabled("zones"):
            zone_insts = []
            for i, z in enumerate(structure_graph.supply_zones):
                time_start = self._format_timestamp(z.created_time) if z.created_time else ""
                broken_status = "Broken" if z.broken else "Active"
                zone_insts.append(DrawInstruction(
                    type_name="ZONE",
                    name=f"FXDNN_ZONE_SUPPLY_{i}",
                    time1=time_start,
                    price1=f"{z.lower:.5f}",
                    price2=f"{z.upper:.5f}",
                    color="LightCoral",
                    style="Supply",
                    text=f"Supply Zone | Str: {z.strength_score:.1f} | {broken_status}"
                ))
            for i, z in enumerate(structure_graph.demand_zones):
                time_start = self._format_timestamp(z.created_time) if z.created_time else ""
                broken_status = "Broken" if z.broken else "Active"
                zone_insts.append(DrawInstruction(
                    type_name="ZONE",
                    name=f"FXDNN_ZONE_DEMAND_{i}",
                    time1=time_start,
                    price1=f"{z.lower:.5f}",
                    price2=f"{z.upper:.5f}",
                    color="LightBlue",
                    style="Demand",
                    text=f"Demand Zone | Str: {z.strength_score:.1f} | {broken_status}"
                ))

            DrawInstructionWriter.write_instructions(
                os.path.join(target_dir, f"{symbol}_zones.csv"),
                zone_insts
            )

        # 3. Trade Levels (SL, TP, Entry, Invalidation) -> EURUSD_levels.csv
        if self.config.is_enabled("levels") and trade_plan:
            levels_insts = []
            entry = trade_plan.get("entry_price")
            sl = trade_plan.get("sl_price")
            tp = trade_plan.get("tp_price")
            invalidation = trade_plan.get("invalidation_level")

            if entry:
                levels_insts.append(DrawInstruction(
                    type_name="LEVEL", name="FXDNN_LEVEL_ENTRY", price1=f"{entry:.5f}", color="Blue", style="Solid", text="Entry"
                ))
            if sl:
                levels_insts.append(DrawInstruction(
                    type_name="LEVEL", name="FXDNN_LEVEL_SL", price1=f"{sl:.5f}", color="Red", style="Dash", text="StopLoss"
                ))
            if tp:
                levels_insts.append(DrawInstruction(
                    type_name="LEVEL", name="FXDNN_LEVEL_TP", price1=f"{tp:.5f}", color="Green", style="Dash", text="TakeProfit"
                ))
            if invalidation:
                levels_insts.append(DrawInstruction(
                    type_name="LEVEL", name="FXDNN_LEVEL_INVALIDATION", price1=f"{invalidation:.5f}", color="Gray", style="Dot", text="Invalidation"
                ))

            DrawInstructionWriter.write_instructions(
                os.path.join(target_dir, f"{symbol}_levels.csv"),
                levels_insts
            )

        # 4. Signals (Accepted & Rejected) -> EURUSD_signals.csv
        if self.config.is_enabled("signals") and decision:
            sig_insts = []
            direction = decision.get("direction", 0)
            accepted = decision.get("accepted", True)
            reason = decision.get("reason", "")
            sig_type = decision.get("signal_type", "Standard")
            strategy = decision.get("strategy", "MM")

            if direction != 0:
                color = "Green" if accepted else "Gray"
                style = "ArrowUp" if direction == 1 else "ArrowDown"
                status_text = "Accepted" if accepted else "Rejected"
                label = f"{strategy.upper()} {sig_type.upper()} {status_text} | {reason}"

                sig_insts.append(DrawInstruction(
                    type_name="SIGNAL",
                    name="FXDNN_SIGNAL_LATEST",
                    color=color,
                    style=style,
                    text=label
                ))

            DrawInstructionWriter.write_instructions(
                os.path.join(target_dir, f"{symbol}_signals.csv"),
                sig_insts
            )

        # 5. Market State -> EURUSD_state.csv
        if self.config.is_enabled("structure") and state_context:
            state_insts = [
                DrawInstruction(
                    type_name="PANEL",
                    name="FXDNN_PANEL_STATE",
                    text=(
                        f"Trend:{state_context.trend_direction};"
                        f"Confidence:{state_context.confidence_score:.2f};"
                        f"Volatility:{state_context.volatility_regime};"
                        f"Market:{state_context.regime}"
                    )
                )
            ]
            DrawInstructionWriter.write_instructions(
                os.path.join(target_dir, f"{symbol}_state.csv"),
                state_insts
            )

        # 6. ML Output -> EURUSD_ml.csv
        if self.config.is_enabled("ml") and ml_output:
            ml_insts = [
                DrawInstruction(
                    type_name="PANEL",
                    name="FXDNN_PANEL_ML",
                    text=(
                        f"TrendProb:{ml_output.get('trend_prob', 0.0):.2f};"
                        f"RangeProb:{ml_output.get('range_prob', 0.0):.2f};"
                        f"TransitionProb:{ml_output.get('transition_prob', 0.0):.2f};"
                        f"BreakProb:{ml_output.get('break_prob', 0.0):.2f};"
                        f"RejectProb:{ml_output.get('reject_prob', 0.0):.2f};"
                        f"Confidence:{ml_output.get('confidence', 0.0):.2f}"
                    )
                )
            ]
            DrawInstructionWriter.write_instructions(
                os.path.join(target_dir, f"{symbol}_ml.csv"),
                ml_insts
            )

        # 7. Strong & Refusal Candles -> EURUSD_candles.csv
        if self.config.is_enabled("candles") or True:
            candle_insts = []
            if decision:
                if "strong_candle_info" in decision:
                    sc_info = decision["strong_candle_info"]
                    cls_val = sc_info.get("classification", "UNKNOWN")
                    sc_val = sc_info.get("quality_score", 0)
                    cnf_val = sc_info.get("confidence", 0.0)
                    candle_insts.append(DrawInstruction(
                        type_name="CANDLE",
                        name="FXDNN_CANDLE_STRONG",
                        color="Green" if sc_info.get("bullish") else "Red",
                        style="ArrowUp" if sc_info.get("bullish") else "ArrowDown",
                        text=f"Strong Candle: {cls_val} | Score: {sc_val} | Conf: {cnf_val:.2f}"
                    ))
                if "refusal_info" in decision:
                    rc_info = decision["refusal_info"]
                    cls_val = rc_info.get("classification", "UNKNOWN")
                    sc_val = rc_info.get("quality_score", 0)
                    cnf_val = rc_info.get("confidence", 0.0)
                    candle_insts.append(DrawInstruction(
                        type_name="CANDLE",
                        name="FXDNN_CANDLE_REFUSAL",
                        color="Blue" if rc_info.get("bullish") else "Orange",
                        style="ArrowUp" if rc_info.get("bullish") else "ArrowDown",
                        text=f"Refusal Rejection: {cls_val} | Score: {sc_val} | Conf: {cnf_val:.2f}"
                    ))

            DrawInstructionWriter.write_instructions(
                os.path.join(target_dir, f"{symbol}_candles.csv"),
                candle_insts
            )

    def annotate_mt5(
        self,
        symbol: str,
        msg: MarketStructureGraph,
        state_ctx: Optional[StateContext] = None,
        trade_levels: Optional[Dict[str, Any]] = None,
        signal_info: Optional[Dict[str, Any]] = None
    ):
        """
        Legacy shim to maintain backwards compatibility with existing strategy/execution code.
        Directly redirects to self.render to keep things highly aligned.
        """
        decision_dict = None
        if signal_info:
            decision_dict = {
                "direction": signal_info.get("direction", 0),
                "accepted": signal_info.get("accepted", True),
                "reason": signal_info.get("reason", "Legacy Signal"),
                "signal_type": "Legacy",
                "strategy": "MM"
            }

        self.render(
            symbol=symbol,
            structure_graph=msg,
            state_context=state_ctx,
            trade_plan=trade_levels,
            decision=decision_dict
        )

    def annotate_matplotlib(
        self,
        ax,
        msg: MarketStructureGraph,
        state_ctx: Optional[StateContext] = None,
        trade_levels: Optional[Dict[str, Any]] = None,
        signal_info: Optional[Dict[str, Any]] = None,
        ml_info: Optional[Dict[str, Any]] = None,
        df: Optional[pd.DataFrame] = None
    ):
        """
        Draw structural overlays directly on a Matplotlib axis.
        This provides offline interactive debugging / backtest visualization validation.
        """
        import matplotlib.patches as patches

        # Dynamically determine the visible window limits on the Matplotlib axes
        # to prevent automatic horizontal axes expansion and vertical scaling squishing.
        x_min, x_max = ax.get_xlim()
        if (x_min == 0.0 and x_max == 1.0) or (abs(x_max - x_min) < 2.0):
            start_i = 0
            end_i = len(df) if df is not None else 99999999
            is_zoomed = False
        else:
            start_i = max(0, int(np.floor(x_min)) - 5)
            end_i = int(np.ceil(x_max)) + 5
            is_zoomed = True

        # Title/State text overlay
        if state_ctx and self.config.is_enabled("structure"):
            text_str = (
                f"Regime: {state_ctx.regime} (Conf: {state_ctx.confidence_score:.2f})\n"
                f"Trend: {state_ctx.trend_direction} | Vol: {state_ctx.volatility_regime}"
            )
            ax.text(
                0.02, 0.95, text_str,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10
            )

        # 1. Swings (dots/markers)
        if self.config.is_enabled("swings"):
            for swing in msg.swing_highs:
                if not is_zoomed or (start_i <= swing.index <= end_i):
                    ax.plot(swing.index, swing.price, 'v', color='red', markersize=6, alpha=0.7)
                    ax.text(swing.index, swing.price + (msg.atr * 0.1), f"H:{swing.index}", color='red', fontsize=6)
            for swing in msg.swing_lows:
                if not is_zoomed or (start_i <= swing.index <= end_i):
                    ax.plot(swing.index, swing.price, '^', color='green', markersize=6, alpha=0.7)
                    ax.text(swing.index, swing.price - (msg.atr * 0.1), f"L:{swing.index}", color='green', fontsize=6)

            if msg.protected_high and (not is_zoomed or (start_i <= msg.protected_high.index <= end_i)):
                ax.plot(msg.protected_high.index, msg.protected_high.price, 'o', color='darkred', markersize=8, label='Protected High')
            if msg.protected_low and (not is_zoomed or (start_i <= msg.protected_low.index <= end_i)):
                ax.plot(msg.protected_low.index, msg.protected_low.price, 'o', color='darkgreen', markersize=8, label='Protected Low')

        # 2. Structure Breaks (BOS / CHOCH lines)
        if self.config.is_enabled("structure"):
            for b in msg.bos:
                if not is_zoomed or (start_i <= b.index <= end_i):
                    color = 'blue' if b.direction == 1 else 'magenta'
                    label_text = "BOS (Bull)" if b.direction == 1 else "BOS (Bear)"
                    if is_zoomed:
                        ax.hlines(y=b.broken_level, xmin=max(start_i, b.index), xmax=end_i, colors=color, linestyles='--', alpha=0.5)
                    else:
                        ax.axhline(y=b.broken_level, color=color, linestyle='--', alpha=0.5)
                    ax.text(b.index, b.broken_level, label_text, color=color, fontsize=8, alpha=0.8)

            for c in msg.choch:
                if not is_zoomed or (start_i <= c.index <= end_i):
                    color = 'cyan' if c.new_trend == 1 else 'orange'
                    label_text = "CHOCH (Bull)" if c.new_trend == 1 else "CHOCH (Bear)"
                    ax.plot(c.index, c.price, 'x', color=color, markersize=10, markeredgewidth=2)
                    ax.text(c.index, c.price, label_text, color=color, fontsize=8, alpha=0.8)

        # 3. Supply & Demand Zones (rectangles)
        if self.config.is_enabled("zones"):
            for z in msg.supply_zones:
                if z.broken:
                    continue
                if not is_zoomed or (z.created_idx <= end_i):
                    rect_width = (end_i - z.created_idx) if is_zoomed else 100
                    rect = patches.Rectangle(
                        (z.created_idx, z.lower),
                        rect_width,
                        z.upper - z.lower,
                        linewidth=1, edgecolor='red', facecolor='red', alpha=0.1
                    )
                    ax.add_patch(rect)
                    ax.text(z.created_idx, z.upper, f"Supply (Str: {z.strength_score:.1f})", color='red', fontsize=7, alpha=0.7)

            for z in msg.demand_zones:
                if z.broken:
                    continue
                if not is_zoomed or (z.created_idx <= end_i):
                    rect_width = (end_i - z.created_idx) if is_zoomed else 100
                    rect = patches.Rectangle(
                        (z.created_idx, z.lower),
                        rect_width,
                        z.upper - z.lower,
                        linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.1
                    )
                    ax.add_patch(rect)
                    ax.text(z.created_idx, z.lower, f"Demand (Str: {z.strength_score:.1f})", color='blue', fontsize=7, alpha=0.7)

        # 4. Trade Levels (Candidate, Stop Loss, Take Profit)
        if self.config.is_enabled("levels") and trade_levels:
            entry = trade_levels.get("entry_price")
            sl = trade_levels.get("sl_price")
            tp = trade_levels.get("tp_price")
            invalidation = trade_levels.get("invalidation_level")

            if entry:
                ax.axhline(y=entry, color='blue', linestyle='-', linewidth=1.5, label='Entry')
            if sl:
                ax.axhline(y=sl, color='red', linestyle='-', linewidth=1.5, label='SL')
            if tp:
                ax.axhline(y=tp, color='green', linestyle='-', linewidth=1.5, label='TP')
            if invalidation:
                ax.axhline(y=invalidation, color='gray', linestyle=':', linewidth=1.0, label='Invalidation')

        # 5. Signals (accepted or rejected arrows)
        if self.config.is_enabled("signals") and signal_info:
            direction = signal_info.get("direction", 0)
            accepted = signal_info.get("accepted", True)
            reason = signal_info.get("reason", "")
            index = signal_info.get("index", len(msg.swing_highs)) # Fallback index

            if direction == 1:
                color = 'green' if accepted else 'gray'
                ax.annotate(
                    f"BUY{' (Rejected)' if not accepted else ''}\n{reason}",
                    xy=(index, msg.atr),
                    xytext=(index, msg.atr - msg.atr * 2.0),
                    arrowprops=dict(facecolor=color, shrink=0.05, width=1.5, headwidth=6)
                )
            elif direction == -1:
                color = 'red' if accepted else 'gray'
                ax.annotate(
                    f"SELL{' (Rejected)' if not accepted else ''}\n{reason}",
                    xy=(index, msg.atr),
                    xytext=(index, msg.atr + msg.atr * 2.0),
                    arrowprops=dict(facecolor=color, shrink=0.05, width=1.5, headwidth=6)
                )

        # Highlight Strong & Refusal Candles on the Matplotlib axis if DataFrame is provided
        if df is not None:
            strong_engine = StrongCandleEngine()
            refusal_engine = RefusalCandleEngine()

            # Evaluate and draw visible bars for display
            eval_start = max(0, start_i)
            eval_end = min(len(df), end_i)

            for i in range(eval_start, eval_end):
                sc = strong_engine.evaluate(df, i, msg)
                if sc.classification in ["VERY_STRONG", "STRONG", "EXPANSION", "CLIMAX", "EXHAUSTION"]:
                    col = "green" if sc.bullish else "red"
                    marker = "P" if sc.classification == "VERY_STRONG" else "*"
                    ax.plot(i, df.iloc[i]["Close"], marker=marker, color=col, markersize=8, alpha=0.8)
                    ax.text(i, df.iloc[i]["High"] + (msg.atr * 0.1), f"{sc.classification}({sc.quality_score})", color=col, fontsize=6, rotation=45)

                rc = refusal_engine.evaluate_rejection(df, i, None, msg)
                if rc.classification in ["PERFECT", "HIGH", "MEDIUM"]:
                    col = "blue" if rc.bullish else "orange"
                    ax.plot(i, df.iloc[i]["Close"], marker="d", color=col, markersize=10, alpha=0.9)
                    ax.text(i, df.iloc[i]["Low"] - (msg.atr * 0.25), f"REF:{rc.classification}({rc.quality_score})", color=col, fontsize=7)

        # 6. ML probability overlays
        if self.config.is_enabled("ml") and ml_info:
            ml_text = (
                f"Market State State Probabilities:\n"
                f"  Trend: {ml_info.get('trend_prob', 0.0)*100:.1f}%\n"
                f"  Range: {ml_info.get('range_prob', 0.0)*100:.1f}%\n"
                f"  Transition: {ml_info.get('transition_prob', 0.0)*100:.1f}%\n"
                f"Level Break Probabilities:\n"
                f"  Break: {ml_info.get('break_prob', 0.0)*100:.1f}%\n"
                f"  Reject: {ml_info.get('reject_prob', 0.0)*100:.1f}%\n"
                f"Confidence: {ml_info.get('confidence', 0.0)*100:.1f}%"
            )
            ax.text(
                0.75, 0.05, ml_text,
                transform=ax.transAxes,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                fontsize=8
            )
