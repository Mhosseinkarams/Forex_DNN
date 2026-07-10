# Visualization Subsystem V2 Redesign

## Purpose
The Visualization Subsystem is a core passive debugging and validation engine. It enables developers and researchers to inspect every trade decision, structural element, and future ML model output without affecting execution speed or trading logic.

It is completely passive, state-decoupled, and operates asynchronously.

---

## High-Level Architecture
Instead of using slow, fragile, and hard-to-maintain JSON files, the V2 Redesign employs highly structured, independent CSV layer files written in the MT5 sandbox or local output directories.

```
       [Trading Engine / MMStrategy]
                     │ (Calls passive .render())
                     ▼
         [ChartAnnotationEngine]
                     │ (Constructs list of DrawInstructions)
                     ▼
          [DrawInstructionWriter]
                     │ (Exports atomic, independent CSVs)
                     ▼
          [MQL5 Renderer Indicator]
                     │ (Reads CSV rows & polls periodically)
                     ▼
            [Chart Objects Draw]
```

---

## File Format Spec (CSV)
Each visual instruction category resides in its own, independent file:
- `EURUSD_structure.csv` (Swings, Protected High/Low, BOS, CHOCH)
- `EURUSD_levels.csv` (SL, TP, Entry, Invalidation)
- `EURUSD_zones.csv` (Supply, Demand)
- `EURUSD_signals.csv` (Accepted, Rejected triggers)
- `EURUSD_state.csv` (Market Regimes, Trend direction)
- `EURUSD_ml.csv` (Break Probabilities, Quality Scores, Confidences)

### CSV Header
`TYPE,NAME,TIME1,TIME2,PRICE1,PRICE2,COLOR,STYLE,TEXT`

### Example Rows
- `LEVEL,FXDNN_LEVEL_ENTRY,,,1.12345,,Blue,Solid,Entry`
- `LEVEL,FXDNN_LEVEL_SL,,,1.12100,,Red,Dash,StopLoss`
- `ZONE,FXDNN_ZONE_DEMAND_0,2026.01.01 12:00:00,,1.11500,1.11650,LightBlue,Demand,Demand Zone | Str: 2.5 | Active`
- `PANEL,FXDNN_PANEL_STATE,,,,,,Trend:Bull;Confidence:0.92;Volatility:High;Market:Trending`

---

## MQL5 Renderer Indicator (`FX_DNN_Chart_Renderer.mq5`)
The native indicator runs in MT5, polling only the changed layers once per second (`InpPollIntervalMs`).
- It parses CSV lines quickly and creates high-performance MT5 graphical objects (`OBJ_HLINE`, `OBJ_RECTANGLE`, `OBJ_ARROW`, `OBJ_TREND`, `OBJ_LABEL`).
- It caches and tracks active object names to prevent recreating objects, avoiding performance hiccups.

---

## Jupyter Notebook Integration (`visualization_examples.ipynb`)
Research workflows leverage the exact same visualization logic using Matplotlib. Passing the same `MarketStructureGraph` produces matching overlays on historical backtest charts.

---

## How to Configure Debug Layers (`debug_config.json`)
You can easily toggle each visual layer dynamically by editing `debug_config.json` without modifying any Python code:
```json
{
    "layers": {
        "swings": true,
        "structure": true,
        "zones": true,
        "levels": true,
        "signals": true,
        "ml": true
    }
}
```

---

## Future ML Module Integration
As future predictive models are trained, simply pass an `ml_output` dictionary to `render()` containing:
- `quality_score`
- `break_prob`
- `range_prob`
- `trend_prob`
- `confidence`

This will automatically render the interactive ML information panel on top of the MT5 chart without any strategy changes.
