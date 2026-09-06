import os
import logging
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from Configs.path_manager import PathManager
from Market_Data_Pipeline.structure_graph import MarketStructureGraph, Zone

logger = logging.getLogger("LabelInspector")


class LabelInspector:
    """
    Trader Validation Workbench Tool.
    Visualizes an anchor candle sample t along with:
      - Historical input window [t - window_size + 1 ... t]
      - Future horizon [t + 1 ... t + future_horizon]
      - SMC Swings, BOS, CHOCH, and active Supply/Demand zones
      - Tested level and level event outcome
      - Strategy trade entry, SL, TP, and trade outcome
    Generates Matplotlib plots for visual trader inspection.
    """

    def __init__(self, window_size: int = 35, future_horizon: int = 20):
        self.window_size = window_size
        self.future_horizon = future_horizon

    def render_sample(
        self,
        df: pd.DataFrame,
        msg: MarketStructureGraph,
        anchor_idx: int,
        output_path: Optional[str] = None,
        zone: Optional[Zone] = None,
        trade_setup: Optional[Dict[str, Any]] = None
    ) -> plt.Figure:
        """
        Renders a comprehensive diagnostic chart for the sample anchored at index anchor_idx.
        """
        start_idx = max(0, anchor_idx - self.window_size + 1)
        end_idx = min(len(df) - 1, anchor_idx + self.future_horizon)

        df_slice = df.iloc[start_idx:end_idx + 1].copy()
        slice_indices = np.arange(len(df_slice))

        fig, ax = plt.subplots(figsize=(14, 7), dpi=120)

        # Plot Candlesticks
        for i, (_, row) in enumerate(df_slice.iterrows()):
            open_p, high_p, low_p, close_p = row["Open"], row["High"], row["Low"], row["Close"]
            color = "green" if close_p >= open_p else "red"

            # Wick
            ax.plot([i, i], [low_p, high_p], color=color, linewidth=1.0)
            # Body
            body_bottom = min(open_p, close_p)
            body_height = max(abs(close_p - open_p), 1e-5)
            rect = patches.Rectangle((i - 0.3, body_bottom), 0.6, body_height, color=color, alpha=0.8)
            ax.add_patch(rect)

        # Map original DataFrame index to slice index
        anchor_rel_idx = anchor_idx - start_idx

        # Vertical divider line at Anchor Candle t
        ax.axvline(x=anchor_rel_idx, color="purple", linestyle="--", linewidth=1.5, label=f"Anchor Candle t={anchor_idx}")

        # Highlight Input Window and Future Horizon
        ax.axvspan(0, anchor_rel_idx, color="blue", alpha=0.08, label="Input Window (Data <= t)")
        ax.axvspan(anchor_rel_idx, len(df_slice) - 1, color="gold", alpha=0.08, label="Future Horizon (Data > t)")

        # Render Supply/Demand Zones active at anchor_idx
        if msg:
            active_supplies = msg.get_active_supplies(anchor_idx)
            active_demands = msg.get_active_demands(anchor_idx)

            for sz in active_supplies[:2]:
                ax.axhspan(sz.lower, sz.upper, color="red", alpha=0.25, label=f"Supply Zone [{sz.lower:.5f}-{sz.upper:.5f}]")

            for dz in active_demands[:2]:
                ax.axhspan(dz.lower, dz.upper, color="green", alpha=0.25, label=f"Demand Zone [{dz.lower:.5f}-{dz.upper:.5f}]")

        # Highlight tested level/zone if passed
        if zone:
            ax.axhspan(zone.lower, zone.upper, color="orange", alpha=0.4, label=f"Tested Zone ({zone.type})")

        # Highlight Strategy SL/TP if passed
        if trade_setup:
            entry = trade_setup.get("entry_price")
            sl = trade_setup.get("sl_price")
            tp = trade_setup.get("tp_price")
            outcome = trade_setup.get("outcome", "UNKNOWN")

            if entry:
                ax.axhline(y=entry, color="blue", linestyle=":", linewidth=1.2, label=f"Entry: {entry:.5f}")
            if sl:
                ax.axhline(y=sl, color="darkred", linestyle="-.", linewidth=1.2, label=f"SL: {sl:.5f}")
            if tp:
                ax.axhline(y=tp, color="darkgreen", linestyle="-.", linewidth=1.2, label=f"TP: {tp:.5f}")

            ax.text(0.02, 0.95, f"Strategy Outcome: {outcome}", transform=ax.transAxes,
                    fontsize=12, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))

        ax.set_title(f"Label Inspection Workbench | Symbol: {msg.symbol if msg else 'N/A'} | Anchor Index: {anchor_idx}", fontsize=14)
        ax.set_xlabel("Bars (Relative Slice Index)", fontsize=10)
        ax.set_ylabel("Price", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.4)

        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            fig.savefig(output_path, dpi=120)
            logger.info(f"Saved label inspection plot to {output_path}")

        return fig
