import pytest
import os
import pandas as pd
import numpy as np
from Visualization.label_inspector import LabelInspector
from Market_Data_Pipeline.structure_graph import MarketStructureGraph, Zone
from Configs.path_manager import PathManager


def test_label_inspector_render():
    inspector = LabelInspector(window_size=20, future_horizon=10)

    n_bars = 50
    df = pd.DataFrame({
        "Open": np.linspace(1.1000, 1.1050, n_bars),
        "High": np.linspace(1.1010, 1.1060, n_bars),
        "Low": np.linspace(1.0990, 1.1040, n_bars),
        "Close": np.linspace(1.1005, 1.1055, n_bars),
        "atr_14": [0.0010] * n_bars
    })

    msg = MarketStructureGraph(symbol="EURUSD", timeframe="M5")
    zone = Zone(upper=1.1030, lower=1.1010, type="Supply", created_idx=0)
    msg.supply_zones.append(zone)

    trade_setup = {
        "entry_price": 1.1020,
        "sl_price": 1.1000,
        "tp_price": 1.1050,
        "outcome": "WIN"
    }

    out_png = PathManager.get_relative_path("temporary", "test_inspection.png")
    fig = inspector.render_sample(
        df=df,
        msg=msg,
        anchor_idx=25,
        output_path=out_png,
        zone=zone,
        trade_setup=trade_setup
    )

    assert os.path.exists(out_png)
