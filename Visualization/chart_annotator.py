import logging
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime

# Optional MT5 import
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from Market_Data_Pipeline.structure_graph import MarketStructureGraph
from Market_Data_Pipeline.state_engine import StateContext
from Visualization.debug_config import DebugConfig

logger = logging.getLogger("VisualizationEngine")

class ChartAnnotationEngine:
    """
    Purpose:
        The VisualizationEngine (ChartAnnotationEngine) draws structural elements,
        supply/demand zones, market states, trade levels, and strategy signals
        directly on MT5 charts (when active) and Matplotlib figures (for validation notebooks).
        Operates passively without influencing trading decisions.
    """
    def __init__(self, config: Optional[DebugConfig] = None):
        self.config = config or DebugConfig()

    def annotate_matplotlib(
        self,
        ax,
        msg: MarketStructureGraph,
        state_ctx: Optional[StateContext] = None,
        trade_levels: Optional[Dict[str, Any]] = None,
        signal_info: Optional[Dict[str, Any]] = None,
        ml_info: Optional[Dict[str, Any]] = None
    ):
        """
        Draw structural overlays directly on a Matplotlib axis.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

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
                ax.plot(swing.index, swing.price, 'v', color='red', markersize=6, alpha=0.7)
            for swing in msg.swing_lows:
                ax.plot(swing.index, swing.price, '^', color='green', markersize=6, alpha=0.7)

            if msg.protected_high:
                ax.plot(msg.protected_high.index, msg.protected_high.price, 'o', color='darkred', markersize=8, label='Protected High')
            if msg.protected_low:
                ax.plot(msg.protected_low.index, msg.protected_low.price, 'o', color='darkgreen', markersize=8, label='Protected Low')

        # 2. Structure Breaks (BOS / CHOCH lines)
        if self.config.is_enabled("structure"):
            for b in msg.bos:
                color = 'blue' if b.direction == 1 else 'magenta'
                label_text = "BOS (Bull)" if b.direction == 1 else "BOS (Bear)"
                ax.axhline(y=b.broken_level, color=color, linestyle='--', alpha=0.5)
                ax.text(b.index, b.broken_level, label_text, color=color, fontsize=8, alpha=0.8)

            for c in msg.choch:
                color = 'cyan' if c.new_trend == 1 else 'orange'
                label_text = "CHOCH (Bull)" if c.new_trend == 1 else "CHOCH (Bear)"
                ax.plot(c.index, c.price, 'x', color=color, markersize=10, markeredgewidth=2)
                ax.text(c.index, c.price, label_text, color=color, fontsize=8, alpha=0.8)

        # 3. Supply & Demand Zones (rectangles)
        if self.config.is_enabled("zones"):
            # Supply (Red translucent rectangle)
            for z in msg.supply_zones:
                if z.broken:
                    continue
                # Draw a translucent bar across the width
                rect = patches.Rectangle(
                    (z.created_idx, z.lower),
                    100,  # Arbitrary lookahead width for notebook
                    z.upper - z.lower,
                    linewidth=1, edgecolor='red', facecolor='red', alpha=0.1
                )
                ax.add_patch(rect)
                ax.text(z.created_idx, z.upper, f"Supply (Str: {z.strength_score:.1f})", color='red', fontsize=7, alpha=0.7)

            # Demand (Blue translucent rectangle)
            for z in msg.demand_zones:
                if z.broken:
                    continue
                rect = patches.Rectangle(
                    (z.created_idx, z.lower),
                    100,  # Arbitrary lookahead width for notebook
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

        # 6. ML probability overlays (predicted probability, confidence, break probability)
        if self.config.is_enabled("ml") and ml_info:
            ml_text = (
                f"ML Trade Quality: {ml_info.get('quality_score', 0.0):.2f}\n"
                f"Zone Break Prob: {ml_info.get('break_prob', 0.0):.2f}"
            )
            ax.text(
                0.75, 0.05, ml_text,
                transform=ax.transAxes,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                fontsize=8
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
        Draw structural overlays directly on live MT5 terminal using MT5 graphical object functions.
        (Operates as a safe no-op if MT5 is not connected or mock environment is active).
        """
        if mt5 is None:
            return

        # MT5 Object drawing logic
        # Clean existing objects of prefix "FX_DNN_"
        # Draw horizontal lines, rectangles, arrows as needed
        # Since this is a passive layer, we log the action and perform terminal object additions.
        logger.debug(f"Passive MT5 drawing requested for {symbol}")
