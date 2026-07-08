# Architecture Guide

This document provides a deep dive into the internal architecture of the Forex Trading Framework, explaining how modules interact, the data flow pipelines, and the lifecycle of signals and positions.

## High-Level Architecture

The framework is built using a decoupled, event-driven architecture. It follows the principle of "Separation of Concerns," where data acquisition, strategy logic, risk management, and execution are handled by independent modules.

### Component Layers

1.  **Environment Layer**: The bottom layer provides a unified interface (`SimulationEnvironment`) to either the live MetaTrader 5 API or the internal backtesting broker.
2.  **Data & Services Layer**: Provides core services like data retrieval (`DataFeed`), indicator calculation (`IndicatorEngine`), and state persistence.
3.  **Core Framework Layer**: Contains the "brain" of the execution engine, including `PositionTracker`, `DrawdownManager`, `ExitManager`, and `SendOrder`.
4.  **Strategy Layer**: Where the user implements signal logic (e.g., `MMStrategy`).

## Module Responsibilities

### Data Acquisition & Processing
- **MT5DataFeed**: Handles the lifecycle of the MT5 connection. It monitors feed health (latency/freshness) and provides OHLCV data.
- **IndicatorEngine**: A stateless calculator that takes raw OHLCV DataFrames and appends technical indicators and metadata (EMA slopes, candle body ratios).

### Execution Orchestration
- **SendOrder**: The gatekeeper for new positions. It validates entries against drawdown limits, symbol conflicts, and risk caps before calling the broker.
- **PositionSizer**: Calculates the exact lot size based on account balance, risk percentage, and stop-loss distance.
- **PositionManager**: The environment-agnostic bridge to the broker. It translates framework requests (Open/Close/Modify) into `MqlTradeRequest` objects.

### Lifecycle Management
- **PositionTracker**: Maintains the "Ground Truth" of open positions. It polls the broker and calculates real-time risk/reward metrics.
- **DrawdownManager**: Enforces daily and total loss ceilings. It snapshots the balance at the start of the day and monitors current equity.
- **ExitManager**: Manages the "Active Life" of a trade. It implements staged take-profits (TP1, TP2) and moves stop-losses to breakeven or trailing.

### Logging & Analysis
- **TradingJournal**: A multi-layered logger. Layer 1 records chronological events; Layer 2 generates the final `PositionLifecycle` summary.
- **TradeAuditor**: A forensic tool used to reconstruct a trade's history for debugging or performance review.

## Execution Pipelines

### Live Pipeline (Real-Time)
1.  **Poll**: `MMStrategy` polls `DataFeed` for new closed bars.
2.  **Detect**: `IndicatorEngine` calculates features; `MMStrategy` evaluates signal rules.
3.  **Validate**: `SendOrder` checks `DrawdownManager` and `PositionTracker` for conflicts.
4.  **Execute**: `PositionManager` sends the order to MT5.
5.  **Track**: `ExitManager` and `PositionTracker` take over to manage the trade until closure.

### Historical Pipeline (Backtesting)
1.  **Simulate**: `SimulationRunner` drives a virtual clock.
2.  **Iterate**: For every bar, it updates `SimulationBroker` prices.
3.  **Sync**: Core modules (`ExitManager`, `Tracker`) poll the virtual broker exactly as they would in live trading.
4.  **Finalize**: Upon completion, a full `StatisticsEngine` report is generated.

## Position Lifecycle

The framework views a trade as a four-stage process:
1.  **Signal**: Discovery of a technical setup.
2.  **Execution**: Successful order entry and lot allocation.
3.  **Management**: Staged TP hits, SL modifications, and trailing.
4.  **Outcome**: Final closure and forensic reconstruction of performance.

## Object Relationships

- `SendOrder` depends on `PositionManager`, `PositionTracker`, `DrawdownManager`, `PositionSizer`, and `ExitManager`.
- `ExitManager` depends on `PositionTracker` and `PositionManager`.
- `MMStrategy` depends on `DataFeed`, `SendOrder`, `TradingJournal`, and `DrawdownManager`.

## Extension Points

The framework is designed to be extended without modifying core logic:
- **New Strategies**: Implement a class that consumes `DataFeed` and calls `SendOrder.execute()`.
- **New Indicators**: Add logic to `IndicatorEngine.calculate()`.
- **New Exit Profiles**: Define new profiles in `position_lifecycle.py` and implement the logic in `ExitManager`.
- **New Risk Models**: Create a class following the `PositionSizer` interface.
