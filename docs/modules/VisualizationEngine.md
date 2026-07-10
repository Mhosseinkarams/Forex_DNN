# VisualizationEngine (ChartAnnotationEngine)

## Purpose
The `VisualizationEngine` (`ChartAnnotationEngine`) provides interactive, passive visual validation. It renders the computed internal state of the `MarketStructureGraph` directly on MT5 charts and Jupyter notebook figures.

## Layer Controls
The engine supports independently toggling visual categories:
- `swings`: High/low pivots and protected limits.
- `structure`: CHOCH and BOS indicators.
- `zones`: Translucent boxes for Supply (red) and Demand (blue).
- `levels`: Selected SL and TP coordinates.
- `signals`: Visual arrow annotations for entries and gray arrows for filtered/rejected candidates.

## Usage
Configure display properties dynamically in `Visualization/debug_config.py`. Enable layers via interactive menus or notebook plot overlays:
```python
annotator = ChartAnnotationEngine()
annotator.annotate_matplotlib(ax, graph, state_ctx)
```
