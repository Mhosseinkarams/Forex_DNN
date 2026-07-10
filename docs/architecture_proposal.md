# FOREX_DNN ARCHITECTURAL REFACTORING PROPOSAL
**From Strategy-Driven to Market-Driven Architecture**

---

## 1. High-Level Architecture Diagram

The architectural refactoring transitions the Forex_DNN system from a strategy-driven model—where each strategy acts as a silo computing its own trend, structure, indicators, and trade location levels—into a centralized, deterministic, **market-driven pipeline**.

In the new architecture, raw market data is consumed once, enriched through standard indicators and specialized deterministic structural engines, and organized into a unified, shared spatial representation: the `MarketStructureGraph`.

The strategy layer becomes lightweight and strategy-agnostic, asking only one question: **"Should I trade?"**, relying on the centralized `TradeLocationEngine` to supply structural stop-loss, take-profit, and entry boundaries.

```
       [ Raw Market Data / Broker / Data Feed ]
                        │
                        ▼
             [ IndicatorEngine (EMA/ATR) ]
                        │
                        ▼
          [ MarketStructureEngine (Swings/BOS/CHOCH) ]
          [ SupplyDemandEngine (Zones/Mitigation)    ]
                        │
                        ▼
            [ MarketStructureGraph ]  ◄─────────── [ VisualizationEngine ]
            ┌────────────────────────────────┐       (ChartAnnotationEngine
            │ - Swing Highs / Swing Lows     │        drawn directly in MT5/
            │ - BOS / CHOCH                  │        Matplotlib notebooks)
            │ - Supply / Demand Zones        │
            │ - Liquidity Pools              │
            └───────────────┬────────────────┘
                            │
                            ├────────────────────────┐
                            ▼                        ▼
                  [ MarketStateEngine ]      [ FeaturePipeline ]
                  (Trend/Volatility Regimes) (ML-ready structural features)
                            │                        │
                            ▼                        ▼
                    [ StrategyEngine ]      [ Future ML Models ]
                    (MM / UniT / SM)       (Classifier/TradeQuality/etc.)
                            │                        │
                            └──────────┬─────────────┘
                                       ▼
                            [ TradeLocationEngine ]
                            (Structural SL/TP selection)
                                       │
                                       ▼
                            [ Risk / Sizing Engine ]
                            (DrawdownManager / Sizer)
                                       │
                                       ▼
                            [ Execution Engine ]
                            (SendOrder / PositionManager)
                                       │
                                       ▼
                           [ TradingJournal / Auditor ]
```

---

## 2. Folder Structure

The folder hierarchy is reorganized to cleanly decouple data processing, market engines, trade management, strategy execution, and visualization, keeping everything modular and avoiding any nested dependency violations.

```
Forex_DNN/
│
├── Collecting_Data/              # Raw data ingest and logging
│   ├── auth.py                   # MT5 Authentication
│   ├── data_feed.py              # Live / Historical Feed abstract/concrete classes
│   ├── indicators.py             # Optimized IndicatorEngine
│   ├── position_lifecycle.py     # PositionLifecycle & OutcomeInfo
│   ├── trading_journal.py        # Centralized TradingJournal
│   └── utils.py                  # Core utility helpers & safe_file_replace
│
├── Market_Data_Pipeline/         # Core market intelligence layer (NEW)
│   ├── __init__.py
│   ├── structure_graph.py        # MarketStructureGraph class & models (StructureLevel, etc.)
│   ├── structure_engine.py       # MarketStructureEngine
│   ├── supply_demand_engine.py   # SupplyDemandEngine
│   ├── state_engine.py           # MarketStateEngine (Trend/Range/Volatility)
│   └── feature_pipeline.py       # FeaturePipeline for ML features
│
├── Strategies/                   # Strategy definitions
│   ├── __init__.py
│   ├── base_strategy.py          # Abstract base strategy class (NEW)
│   ├── mm_strategy.py            # Refactored MMStrategy (lightweight, uses MSG)
│   ├── unit_strategy.py          # Future UniT Strategy
│   └── sm_strategy.py            # Future SM Strategy
│
├── Trade_Execution/              # Risk and order execution
│   ├── __init__.py
│   ├── location_engine.py        # TradeLocationEngine (NEW)
│   ├── drawdown.py               # DrawdownManager
│   ├── risk_sizing.py            # PositionSizer
│   ├── send_order.py             # Order Sender
│   ├── exit_manager.py           # ExitManager
│   ├── position_manager.py       # PositionManager
│   └── position_tracker.py       # PositionTracker
│
├── Visualization/                # Visualization Engine (NEW)
│   ├── __init__.py
│   ├── chart_annotator.py        # ChartAnnotationEngine for MT5 / Matplotlib overlays
│   └── debug_config.py           # Layer configuration and interactive debugging settings
│
├── docs/                         # Architecture, developer guides, module guides
│   ├── architecture.md
│   ├── market_structure.md
│   └── supply_demand.md
│
├── examples/                     # Jupyter Research Notebooks
│   ├── research_workbench.ipynb  # Main research workbench
│   ├── backtest_eurusd.ipynb     # EURUSD Backtest Notebook
│   ├── val_structure.ipynb       # Validation Notebook: MarketStructureEngine
│   ├── val_supply_demand.ipynb   # Validation Notebook: SupplyDemandEngine
│   └── val_market_state.ipynb    # Validation Notebook: MarketStateEngine
│
├── simulation/                   # Historical Backtesting & Simulation Framework
│   ├── backtest_report.py
│   ├── historical_data_feed.py
│   ├── simulation_account.py
│   ├── simulation_broker.py
│   ├── simulation_clock.py
│   ├── simulation_environment.py
│   ├── simulation_order_engine.py
│   ├── simulation_runner.py
│   └── statistics_engine.py
│
├── integration_validation.py     # E2E system health suite
├── live_validation.py            # Live MT5 connection validation
├── main.py                       # Unified system entry point
└── requirements.txt              # Standard system dependencies
```

---

## 3. Module Dependency Graph

Dependencies strictly move in **one direction** (top-down / analytical-to-executional) to prevent circular imports. Execution engines and strategies never depend on each other, nor do engines directly depend on UI / visualization layers.

```
       [ Collecting_Data.data_feed ]
                    │
                    ▼
       [ Collecting_Data.indicators ]
                    │
                    ▼
     [ Market_Data_Pipeline.structure_graph ]
        │                               │
        ├───────────────────────────────┤
        ▼                               ▼
 [ structure_engine ]         [ supply_demand_engine ]
        │                               │
        └───────────────┬───────────────┘
                        ▼
         [ MarketStructureGraph (Instance) ]  ◄──── [ Visualization.chart_annotator ]
           │                           │
           ▼                           ▼
 [ state_engine ]             [ feature_pipeline ] ────► [ ML Models ]
           │                           │
           └────────────┬──────────────┘
                        ▼
               [ Strategies (Engine) ]
                        │
                        ▼
         [ Trade_Execution.location_engine ]
                        │
                        ▼
         [ Trade_Execution.risk_sizing ]
                        │
                        ▼
         [ Trade_Execution.send_order ]
                        │
                        ▼
         [ Trade_Execution.position_tracker ]
```

---

## 4. Engine Interaction Diagram

The dynamic interaction flow during a single bar processing event highlights how the raw candle tick is systematically transformed into a trading decision.

```
[DataFeed]  [Indicators]  [StructureEngine]  [SupplyDemand]  [MSGraph]  [StateEngine]  [Strategy]  [LocationEngine] [SendOrder]
    │             │               │                 │            │            │             │              │             │
    │──get_ohlcv─►│               │                 │            │            │             │              │             │
    │◄───df──────│               │                 │            │            │             │              │             │
    │             │──calculate───►│                 │            │            │             │              │             │
    │             │◄───df_ind─────│                 │            │            │             │              │             │
    │             │               │────process─────►│            │            │             │              │             │
    │             │               │◄──df_struct─────│            │            │             │              │             │
    │             │               │                 │──process──►│            │             │              │             │
    │             │               │                 │◄──df_sd────│            │             │              │             │
    │             │               │                 │            │──update───►│             │              │             │
    │             │               │                 │            │            │──calculate─►│              │             │
    │             │               │                 │            │            │◄──state─────│              │             │
    │             │               │                 │            │            │             │──evaluate───►│             │
    │             │               │                 │            │            │             │◄──should_tr──│             │
    │             │               │                 │            │            │             │              │             │
    │             │               │                 │            │            │             │(If Yes)      │             │
    │             │               │                 │            │            │             │──get_levels─►│             │
    │             │               │                 │            │            │             │◄──sl_tp_dict─│             │
    │             │               │                 │            │            │             │              │───execute──►│
```

---

## 5. Strategy Lifecycle

In the refactored architecture, strategies inherit from a standard `BaseStrategy` that defines a strict interface:

1. **Initialization**: Strategies are initialized with references to `DataFeed`, `MarketStructureGraph`, `TradeLocationEngine`, `DrawdownManager`, and `SendOrder`.
2. **Subscription**: Strategies subscribe to specific symbols and timeframes.
3. **Polling / Tick Event**:
   - The strategy receives a callback or polls a symbol/timeframe.
   - It fetches the updated `MarketStructureGraph` instance representing the current bar state.
   - It invokes its private `_evaluate_rules(graph, state)` method. This method uses only properties from `MarketStructureGraph` (e.g., `graph.trend`, `graph.last_bos_direction`, `graph.nearest_demand_distance`) and `MarketStateEngine` to determine bias.
4. **Signal Generation**:
   - If bias matches entry criteria, it requests stop-loss (SL) and take-profit (TP) coordinates from `TradeLocationEngine` by passing the `MarketStructureGraph`.
   - The strategy calls `SendOrder.execute(...)` with the resolved trade parameters.
5. **Shutdown**: Gracefully detaches and serializes strategy state.

---

## 6. Trade Lifecycle

The trade lifecycle is highly standardized and centrally audited, driven by explicit Exit Profiles:

```
[ Signal Triggered ]
         │
         ▼
[ TradeLocationEngine ] ──► Computes SL and TP based strictly on structural high/low/zones
         │
         ▼
[ DrawdownManager ] ──► Confirms trading is allowed & checks maximum allowable risk budget
         │
         ▼
[ PositionSizer ] ──► Sizer calculates safe lot sizes:
         │            Lot = Risk_Dollars / ((SL_Distance / Tick_Size) * Tick_Value)
         ▼
[ SendOrder ] ──► Dispatches virtual or live MT5 trade request
         │
         ├──► Success: Registered in [ PositionTracker ] and [ ExitManager ]
         │             Logged as `order_open` in [ TradingJournal ]
         │
         └──► Failure: Recorded as `order_failure` in [ TradingJournal ]
         │
         ▼
[ ExitManager ] ──► Actively monitors SL/TP and performs multi-stage or full close
         │
         ▼
[ PositionClosed ] ──► Generates complete [ PositionLifecycle ] summary
                       Logged as `position_closed` and `lifecycle` in [ TradingJournal ]
                       Archived and audited by [ TradeAuditor ]
```

---

## 7. Data Lifecycle

The system enforces deterministic processing by ensuring all components utilize the exact same historical data feed bar by bar, eliminating look-ahead bias and discrepancies between modules:

1. **Ingest**: `HistoricalDataFeed` loads OHLCV CSV data (backtesting) or `MT5DataFeed` requests live rates.
2. **Standardization**: Data is structured into a pandas DataFrame containing standard timestamp alignment.
3. **Indication**: Columns for fast and slow EMAs and ATR are appended via `IndicatorEngine`.
4. **Graph Generation**: Structural points (Swings, BOS, CHOCH, Supply/Demand zones) are mapped to a point-in-time array and populated inside the current `MarketStructureGraph`.
5. **Feature Engineering**: Feature pipeline converts current `MarketStructureGraph` coordinates into normalized numeric rows.
6. **State Contextualization**: The `MarketStateEngine` overlays trend and volatility metrics, completing the data frame before it is dispatched to strategies or ML models.

---

## 8. ML Lifecycle

Machine Learning is cleanly decoupled from deterministic market structure calculations. It acts as an optimization layer operating **after** the physical structures are known:

```
┌────────────────────────────────────────────────────────┐
│               Deterministic Base Pipeline              │
│  [Raw Data] ──► [Engines] ──► [MarketStructureGraph]   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
                    [FeaturePipeline]
             Extracts distances, age, and
           structural strengths as ML inputs
                            │
                            ▼
               ┌─────────────────────────┐
               │    Pre-trained Models   │
               └────────────┬────────────┘
                            │
       ┌────────────────────┼────────────────────┐
       ▼                    ▼                    ▼
[MarketState]        [LevelBreakProb]      [TradeQuality]
Classifies trend     Probability of a      Scores strategy
vs ranging state     zone failing          setup quality
```

---

## 9. MarketStructureGraph Object Model

The `MarketStructureGraph` is the centralized, type-safe data structure holding all structural objects detected by the core analytical engines.

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

@dataclass
class StructureLevel:
    price: float
    index: int
    timestamp: datetime
    strength: int
    level_type: str  # 'SwingHigh', 'SwingLow', 'ProtectedHigh', 'ProtectedLow'

@dataclass
class LiquidityPool:
    upper: float
    lower: float
    index: int
    timestamp: datetime
    pool_type: str  # 'EqualHighs', 'EqualLows', 'SwingSweep'
    is_swept: bool = False
    swept_idx: Optional[int] = None

@dataclass
class MarketStructureGraph:
    symbol: str
    timeframe: str
    timestamp: datetime

    # Swings and Structural Points
    swing_highs: List[StructureLevel] = field(default_factory=list)
    swing_lows: List[StructureLevel] = field(default_factory=list)
    protected_high: Optional[StructureLevel] = None
    protected_low: Optional[StructureLevel] = None

    # Breaks
    bos: List[BOS] = field(default_factory=list)
    choch: List[CHOCH] = field(default_factory=list)

    # Zones
    supply_zones: List[Zone] = field(default_factory=list)
    demand_zones: List[Zone] = field(default_factory=list)

    # Liquidity
    liquidity_pools: List[LiquidityPool] = field(default_factory=list)

    # Indicators / Context
    trend_direction: str = "Neutral"  # "Bull", "Bear", "Neutral"
    ema_relationship: str = "Flat"    # "BullishSeparated", "BearishSeparated", "Converged"
    ema_distance_atr: float = 0.0
    atr: float = 0.0
    volatility: float = 0.0
    range_width_pips: float = 0.0
    session: str = "Unknown"          # "London", "NewYork", "Asian"

    def get_nearest_demand(self, price: float) -> Optional[Zone]:
        active_demands = [z for z in self.demand_zones if not z.broken and z.upper < price]
        return max(active_demands, key=lambda z: z.upper) if active_demands else None

    def get_nearest_supply(self, price: float) -> Optional[Zone]:
        active_supplies = [z for z in self.supply_zones if not z.broken and z.lower > price]
        return min(active_supplies, key=lambda z: z.lower) if active_supplies else None
```

---

## 10. Class Diagrams

### Analytical Components Class Diagram
```
┌───────────────────────────────────────────┐
│           MarketStructureEngine           │
├───────────────────────────────────────────┤
│ - lookback: int                           │
│ - swings: List[SwingPoint]                │
├───────────────────────────────────────────┤
│ + process(df: DataFrame) -> DataFrame     │
│ + detect_swings(df: DataFrame)            │
└─────────────────────┬─────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────┐
│            SupplyDemandEngine             │
├───────────────────────────────────────────┤
│ - impulse_threshold: float                │
│ - zones: List[Zone]                       │
├───────────────────────────────────────────┤
│ + process(df: DataFrame) -> DataFrame     │
└─────────────────────┬─────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────┐
│            MarketStructureGraph           │
├───────────────────────────────────────────┤
│ + symbol: str                             │
│ + timeframe: str                          │
│ + supply_zones: List[Zone]                │
│ + demand_zones: List[Zone]                │
│ + swing_highs: List[StructureLevel]       │
└─────────────────────┬─────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────┐
│             MarketStateEngine             │
├───────────────────────────────────────────┤
│ - confidence_threshold: float             │
├───────────────────────────────────────────┤
│ + evaluate(msg: MSGraph) -> StateContext  │
└───────────────────────────────────────────┘
```

---

## 11. Migration Plan from the Current Architecture

The migration is executed systematically, ensuring that at no point is any live trading or backtesting system completely non-functional.

1. **Step 1: Code Isolation & Placement**
   - Create the directory structures `Market_Data_Pipeline/`, `Trade_Execution/`, and `Visualization/`.
   - Copy existing `market_structure.py` and `supply_demand.py` into `Market_Data_Pipeline/` without modifying their existing behavior. This maintains immediate backward-compatibility.

2. **Step 2: Define Interfaces**
   - Define the `MarketStructureGraph` models and the `BaseStrategy` interface.
   - Implement the `TradeLocationEngine` to extract and supply levels.

3. **Step 3: Refactor MMStrategy**
   - Adapt `MMStrategy` to inherit from `BaseStrategy`.
   - Instead of running standard EMA Indicator calculations inside `MMStrategy`, redirect the strategy's data pipeline to fetch and process updates from `MarketStructureGraph`.
   - Set up the strategy to delegate all SL and TP logic to the `TradeLocationEngine`.

4. **Step 4: Adaptation of Backtesting and Simulation Loops**
   - Update `SimulationRunner` and backtesting scripts to initialize the centralized `Market_Data_Pipeline` components and supply the resulting `MarketStructureGraph` to running strategies.

5. **Step 5: Run Integration and Validation Suites**
   - Verify that all outputs (pips, PnL, lot sizing, and entry counts) in backtests remain perfectly aligned with previous baselines.

---

## 12. Repository Cleanup Plan

To maintain production excellence, all dead, unused, and legacy experimental files must be cleaned up:

- **Archiving Old Code**: Move legacy preprocessing files (`preproc_multi_inout.py`, `preproc_single_inout.py`, `preproc_pivot.py`) into a newly created `archive/` folder if still historically valuable, or remove them entirely to keep the directory clean.
- **Deduplicating Modules**: Move the risk sizing, order sending, drawdown management, exit tracking, and auditing modules from `PositionManager/` to `Trade_Execution/` to unify trade management naming standards.
- **Cleaning Local Roots**: Relocate or remove top-level compiled binary models or loose diagnostic images (`model.png`, `.h5` files) into dedicated `/models/` or `/output/` directories.

---

## 13. Documentation Cleanup Plan

- **README.md Update**: Rewrite the Core Architecture and Module Overview sections to document the market-driven flow and the centerpiece `MarketStructureGraph`.
- **New Module Guides**:
  - Write `docs/modules/MarketStructureGraph.md` describing its structure and properties.
  - Write `docs/modules/TradeLocationEngine.md` documenting structural stop-loss and take-profit resolution.
  - Write `docs/modules/VisualizationEngine.md` containing drawings, interactive layers, and debugging setup.
- **Flow Diagrams**: Update all SVG/ASCII architectural diagrams in `docs/architecture.md` to reflect the decoupled engines.

---

## 14. Files that Can Be Deleted or Archived

The following files are obsolete and should be moved to `archive/` or deleted to avoid maintenance overhead:

| File / Path | Action | Reason |
| :--- | :--- | :--- |
| `Collecting_Data/preproc_multi_inout.py` | Archive | Old pre-processing layout |
| `Collecting_Data/preproc_single_inout.py` | Archive | Old pre-processing layout |
| `Collecting_Data/preproc_pivot.py` | Archive | Replaced by deterministic structure engine |
| `Collecting_Data/Price.py` | Delete | Redundant data downloader |
| `Collecting_Data/mt5data.py` | Delete | Redundant MT5 retrieval script |
| `model.png` | Move | Belongs in research output |
| `*.h5` | Move | Compiled models belong in `DNN/models/` |

---

## 15. Files that Must Be Preserved

These files represent the core stable business logic of the system and must not lose any tested functionality:

- **`Collecting_Data/trading_journal.py`**: Journal schema and central logging must remain identical to maintain chronological safety.
- **`Collecting_Data/position_lifecycle.py`**: Centralized Exit Profile attributes (`standard`, `single`, `high_risk`, `reversal`) and PnL reporting structures.
- **`PositionManager/exit_manager.py` & `position_tracker.py`**: State restoration, tick-based formulas, JPY/Gold/Index calculations, and MT5-aligned event hooks.
- **`simulation/` (All files)**: Virtual clock, chronological Timeline runner, virtual broker mock, and account balances.
- **`integration_validation.py` & `live_validation.py`**: The comprehensive system health check suite.

---

## 16. Step-by-Step Implementation Roadmap

A detailed 4-phase plan to execute the refactoring with zero risk.

### Phase 1: Directory Setup & Structural Core (Step 1-3)
1. Set up directories: `Market_Data_Pipeline/`, `Trade_Execution/`, and `Visualization/`.
2. Construct the `MarketStructureGraph` dataclass with all internal fields (zones, swings, structural levels, EMA separation, ATR, London/NY sessions).
3. Implement `MarketStructureEngine` and `SupplyDemandEngine` as pure processors that take indicator dataframes and produce a populated `MarketStructureGraph` object for any given bar index.

### Phase 2: Location Engine, base Strategy, and State Engine (Step 4-7)
4. Implement `TradeLocationEngine`:
   - Inputs: `MarketStructureGraph`, trade direction, and signal category.
   - Outputs: Entry, SL, TP, and Risk/Reward parameters. Resolves SL by locating the nearest protected structural swing point or supply/demand zone.
5. Create `MarketStateEngine` to analyze the graph and output trend regime (Trending, Ranging, Transition, Expansion, Compression) with confidence metrics.
6. Design the `BaseStrategy` base class specifying standard initialization and tick execution interfaces.
7. Implement `FeaturePipeline` to read `MarketStructureGraph` coordinates and format them into an array of clean ML-ready indicators (e.g., normalized distances).

### Phase 3: Strategy Migration & Trade Execution Decoupling (Step 8-11)
8. Refactor `MMStrategy` to inherit from `BaseStrategy`. Remove internal swing-point calculation, standard indicators calculation loops, and direct SL resolution. Wire the strategy to query the shared `MarketStructureGraph` and `TradeLocationEngine`.
9. Relocate risk sizing, drawdown checks, order dispatches, and trackers from `PositionManager/` to `Trade_Execution/` with backward-compatibility imports.
10. Update backtesting engines (`SimulationRunner`) to construct the central MSG and supply it sequentially to strategy classes.
11. Build the validation notebooks (`examples/val_structure.ipynb`, `examples/val_supply_demand.ipynb`) validating structure detection.

### Phase 4: Visualization Engine & Final Verification (Step 12-15)
12. Create the `VisualizationEngine` (`ChartAnnotationEngine`):
    - Subscribes to outputs from the analytics pipeline.
    - Implements layers that can be toggled on/off.
    - Implements **Debug Mode**: Draws BOS labels, zone rectangles (Supply: Red, Demand: Blue), liquidity sweeps, and trade levels on Matplotlib charts and MT5 charts.
13. Execute Repository Cleanup: Archive legacy files, deduplicate scripts.
14. Rewrite Documentation Guides: Update developer guides, module Markdown files, and the main project README.
15. Run full system validation suites (`integration_validation.py`, `live_validation.py`) to confirm zero regressions.

---

## Visualization Engine Architecture Detail

The `VisualizationEngine` (or `ChartAnnotationEngine`) is a core diagnostic tool designed to operate passively alongside the trading system. It reads the computed state of the `MarketStructureGraph` and overlays visual markers onto active trading frames without altering execution pathways.

### Drawing Interfaces
The engine provides two concrete renderers:
1. **`MatplotlibAnnotator`**: Used in validation and research notebooks to draw clear historical overlays (candles, colored boxes for zones, and arrow annotations for entries).
2. **`MT5Annotator`**: Uses MetaTrader 5's graphical object functions (`ObjectCreate`, `ObjectSetInteger`) to draw rectangles, labels, and trendlines in real-time.

### Interactive Debugging Layers
The annotation system contains separate, independent drawing layers:
- `Layer.SWINGS`: Points at high/low pivots.
- `Layer.STRUCTURE`: Horizontal lines showing CHOCH, BOS, and protected levels.
- `Layer.ZONES`: Colored translucent rectangles representing Supply & Demand zones with freshness scores.
- `Layer.LEVELS`: Entry, SL, TP, and invalidation boundaries.
- `Layer.SIGNALS`: Arrows highlighting accepted signals or gray markers representing rejected signals with rejection labels.

A standard configuration file `debug_config.json` allows developers to turn each layer on or off dynamically during backtests or live terminal debugging sessions.
