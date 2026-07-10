# Architecture Guide (Market-Driven Refactored)

This document provides a deep dive into the internal architecture of the Forex Trading Framework, explaining how modules interact, the data flow pipelines, and the lifecycle of signals and positions.

## High-Level Architecture

The framework is built using a decoupled, event-driven architecture. It transitions from a strategy-driven model to a centralized, deterministic, **market-driven pipeline**.

In this new pipeline, raw market data is consumed once, enriched through standard indicators and specialized deterministic structural engines, and organized into a unified, shared spatial representation: the `MarketStructureGraph`.

### Component Layers

1.  **Environment Layer**: The bottom layer provides a unified interface (`SimulationEnvironment`) to either the live MetaTrader 5 API or the internal backtesting broker.
2.  **Data & Pipeline Layer**: Handles raw data retrieval (`DataFeed`), indicator calculation (`IndicatorEngine`), and structural intelligence (`MarketStructureEngine`, `SupplyDemandEngine`, `MarketStateEngine`, `FeaturePipeline`).
3.  **Central Market Representation**: Holds the populated `MarketStructureGraph` instances.
4.  **Strategy & Trade Location Layer**: Lightweight strategies query the `MarketStructureGraph` and leverage `TradeLocationEngine` to resolve structural boundaries.
5.  **Execution & Management Layer**: Contains the core execution engines, including `PositionTracker`, `DrawdownManager`, `ExitManager`, `SendOrder`, and `PositionSizer`.

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

### Trade Location & Sizing
- **TradeLocationEngine**: Computes candidate entries, stop-loss, take-profit, and invalidation levels based strictly on structural information.
- **PositionSizer**: Calculates the exact lot size based on account balance, risk percentage, and stop-loss distance.
- **SendOrder**: The gatekeeper for new positions. It validates entries against drawdown limits, symbol conflicts, and risk caps.

## Object Relationships

- `MMStrategy` depends on `DataFeed`, `MarketStructureGraph`, `TradeLocationEngine`, and `SendOrder`.
- `TradeLocationEngine` depends on `MarketStructureGraph`.
- `SendOrder` depends on `PositionManager`, `PositionTracker`, `DrawdownManager`, `PositionSizer`, and `ExitManager`.
- `ExitManager` depends on `PositionTracker` and `PositionManager`.
