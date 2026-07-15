# Architecture Guide (Market-Driven Refactored)

This document provides a deep dive into the internal architecture of the Forex Trading Framework, explaining how modules interact, the data flow pipelines, and the lifecycle of signals and positions.

## High-Level Architecture

The framework is built using a decoupled, event-driven architecture. It transitions from a strategy-driven model to a centralized, deterministic, **market-driven pipeline**.

In this new pipeline, raw market data is consumed once, enriched through standard indicators and specialized deterministic structural engines, and organized into a unified, shared spatial representation: the `MarketStructureGraph`.

### Component Layers

1.  **Environment Layer**: The bottom layer provides a unified interface (`SimulationEnvironment`) to either the live MetaTrader 5 API or the internal backtesting broker.
2.  **Data & Pipeline Layer**: Handles raw data retrieval (`DataFeed`), indicator calculation (`IndicatorEngine`), and structural intelligence (`MarketStructureEngine`, `SupplyDemandEngine`, `MarketStateEngine`, `FeaturePipeline`).
3.  **Central Market Representation**: Holds the populated `MarketStructureGraph` instances.
4.  **Centralized ML Inference & Decision Layer (Module 16)**: Houses `MLDecisionEngine`, `ModelRegistry`, `BaseCalibrator` variants, and `BasePolicy` implementations which aggregate multi-model predictions into an immutable, thread-safe `DecisionContext`.
5.  **Strategy & Trade Location Layer**: Lightweight trend-following (MM) and range mean-reversion (SM) strategies query the `MarketStructureGraph`, consume `DecisionContext` predictions, leverage `RefusalCandleEngine` (for rejections), and use `TradeLocationEngine` to resolve structural boundaries.
6.  **Execution & Management Layer**: Contains the core execution engines, including `PositionTracker`, `DrawdownManager`, `ExitManager`, `SendOrder`, and `PositionSizer`.

## Module Responsibilities

### Data Acquisition & Processing
- **MT5DataFeed**: Handles the lifecycle of the MT5 connection. It monitors feed health and provides OHLCV data.
- **IndicatorEngine**: A stateless calculator that takes raw OHLCV DataFrames and appends technical indicators and metadata (EMA slopes, candle body ratios).

### Market Intelligence (New)
- **MarketStructureEngine**: Detects swing highs/lows, BOS, CHOCH, and protected levels.
- **SupplyDemandEngine**: Tracks institutional supply and demand zones, zone touch counts, mitigations, and freshness.
- **MarketStructureGraph**: Central, shared data container representing the point-in-time structural graph of the market.
- **MarketStateEngine**: Classifies current market state regimes (Trending, Ranging, Transition, Expansion, Compression).
- **FeaturePipeline**: Formats graph coordinates into ML-ready numerical vectors.

### Centralized ML Inference & Decision Layer (New)
- **MLDecisionEngine**: Aggregates models, validates feature vectors via `FeatureRegistry`, runs calibrated inference, executes policy recommendations, and builds the immutable `DecisionContext`.
- **ModelRegistry**: Lazy-loads, caches, and tracks registered model assets (such as `MarketStateClassifier`, `LevelBreakProbabilityModel`, and `TradeQualityModel`), gracefully ignoring missing optional models.
- **Confidence Calibration**: Platt scaling, Isotonic regression, and Identity calibration layers decouple raw probabilities from production confidence outputs.
- **Policy Layer**: Sizing, risk-scaling, and breakout/rejection-based target setting recommended by `RuleBasedPolicy`.

### Trade Location & Sizing
- **TradeLocationEngine**: Computes candidate entries, stop-loss, take-profit, and invalidation levels based strictly on structural information.
- **PositionSizer**: Calculates the exact lot size based on account balance, risk percentage, and stop-loss distance.
- **SendOrder**: The gatekeeper for new positions. It validates entries against drawdown limits, symbol conflicts, and risk caps.

## Object Relationships

- `MMStrategy` depends on `DataFeed`, `MarketStructureGraph`, `TradeLocationEngine`, and `SendOrder`.
- `SMStrategy` depends on `DataFeed`, `MarketStructureGraph`, `RefusalCandleEngine`, `TradeLocationEngine`, and `SendOrder`.
- `TradeLocationEngine` depends on `MarketStructureGraph`.
- `SendOrder` depends on `PositionManager`, `PositionTracker`, `DrawdownManager`, `PositionSizer`, and `ExitManager`.
- `ExitManager` depends on `PositionTracker` and `PositionManager`.

## Runtime ML Integration Layer (Milestone 5)

Milestone 5 implements the runtime ML integration framework, guaranteeing that live inference and offline training operate under a single, unified source of truth without feature drift, look-ahead bias, or circular strategy dependencies.

### 1. Unified Execution Flow

The framework follows a strict, unidirectional hierarchical pipeline:

```
Indicators (IndicatorEngine)
       │
       ▼
MarketStructureGraph (MarketStructureEngine + SupplyDemandEngine)
       │
       ▼
FeaturePipeline (Extracts, standardizes, and validates feature vectors)
       │
       ▼
MLDecisionEngine (Loads models, runs calibrated inference, recommends policy)
       │
       ▼
SignalEvaluator (Evaluates MM Strategy technical rules and ML Diagnostics)
       │
       ▼
MMStrategy (Owns final decision, manages trades, runs in Shadow Mode)
       │
       ▼
TradeFeatureRecorder (Logs candidates and trade outcomes for future retraining)
```

The strategy always owns the final decision; ML never directly executes, manages, or closes positions on its own.

## Stubborn Man (SM) Strategy & Refusal Candle Engine

### 1. Strategy Trading Philosophy
While the MM Strategy trades trend continuation and EMA breakouts, the Stubborn Man (SM) Strategy specializes in mean reversion and range boundaries. SM assumes that prices are more likely to reject key institutional supply and demand zones than immediately break them.

### 2. Complete Execution Pipeline
The complete, end-to-end range trading pipeline flow is as follows:
```
MarketStructureEngine (Swings, Shifts)
       │
       ▼
SupplyDemandEngine (Active Range Boundaries)
       │
       ▼
MarketStateClassifier (Validates regime is RANGE; TREND or TRANSITION exits immediately)
       │
       ▼
TradeLevelEngine (Resolves structural SL beyond zone & TP before opposite zone with buffer)
       │
       ▼
RefusalCandleEngine (Evaluates multi-factor score for structural level rejection)
       │
       ▼
LevelBreakProbabilityModel (Validates that Level Break Probability <= max_break_probability)
       │
       ▼
MLDecisionEngine (Aggregates inferences under Shadow Mode)
       │
       ▼
SignalEvaluator (Performs final risk and validation check)
       │
       ▼
SMStrategy (Submits order, logs to journal, and writes Daily Retraining files)
```

### 3. Refusal Candle Engine Scoring
The `RefusalCandleEngine` (`Strategies/refusal_candle_engine.py`) uses a highly parameterized, score-based mechanism evaluating multi-factor indicators:
- **Wick/Body Ratio**: Size of the rejection wick compared to the candle body size.
- **Wick Percentage**: Rejection wick length relative to total candle range.
- **Close Position**: Closeness of the bar close to the bottom (for supply) or top (for demand).
- **Zone Penetration Depth**: Depth of the rejection bar's extremity into the S/D zone.
- **Close Outside Zone**: Confirms that price was successfully rejected outside the zone.
- **Previous Candle direction & Momentum**: Rejection is strongest if incoming momentum was strong, signaling an abrupt block.
- **Volume Spike**: Rejections backed by high tick volume represent strong institutional participation.

### 2. Runtime Feature Pipeline (Module 17)
The `FeaturePipeline` under `ML/feature_pipeline.py` consumes current OHLC dataframes, indicator dataframes, the active `MarketStructureGraph`, and execution contexts. It extracts and standardizes features dynamically according to the active `FeatureRegistry` order.
- **Strict Validation**: The pipeline proactively detects missing values, NaNs, infinites, and type mismatches. Any discrepancy triggers a log failure and falls back to the registry-defined default value.
- **Runtime Performance**: Reusable calculations are cached to avoid redundant computation of indicator and mathematical operations.

### 3. Shadow Mode MM Strategy Integration (Module 18)
In Version 1, ML does not influence or filter trading execution. The strategy operates in **Shadow Mode** (`SHADOW_MODE = True`, `ML_FILTERING = False`), where the complete ML pipeline is evaluated on every candidate signal, but trading continues to obey technical rules.
- **Specialized Loggers**: Every candidate signal writes detailed diagnostics across specialized logger files:
  - `Logs/runtime_features.log`
  - `Logs/decision_engine.log`
  - `Logs/shadow_mode.log`
  - `Logs/signal_evaluator.log`

### 4. Trade Feature Recorder (Module 19)
The `TradeFeatureRecorder` under `ML/trade_feature_recorder.py` logs candidates and final outcomes in thread-safe, daily rolling files (supporting CSV and Parquet).
- **Tabular RETRAINING Compatibility**: When a candidate is evaluated, a flat record is written. When a trade is eventually completed (logged via `TradingJournal.log_lifecycle`), the recorder loads the daily file, updates the row matching the master `signal_id` with exit prices, duration, profit, R-multiple, drawdowns, and result labels, and overwrites it.

### 5. Unified Signal Evaluator (Module 20)
The `SignalEvaluator` under `Strategies/signal_evaluator.py` provides a single, unified, strategy-agnostic interface to decide if a signal is accepted, rejected, its confidence, and priorities, separating strategies completely from ML internal states.
