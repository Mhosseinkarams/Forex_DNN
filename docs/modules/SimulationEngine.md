# SimulationEngine Module

## Purpose
The `SimulationEngine` (located in the `simulation/` directory) provides the infrastructure for high-fidelity backtesting. It allows the framework to execute strategies against historical data with realistic fills, account math, and staged trade management.

## Components

### `SimulationEnvironment (env)`
A singleton that proxies the MetaTrader 5 API. In backtest mode, it redirects calls like `order_send` and `positions_get` to the `SimulationBroker`.

### `SimulationBroker`
The virtual broker. It maintains a list of virtual positions and deals. It also monitors for SL/TP hits whenever the market price is updated.

### `SimulationClock`
Controls the virtual time. It ensures that the `get_now()` calls in the framework return the correct historical timestamp.

### `SimulationRunner`
The main loop that drives the backtest. It iterates through the historical data, updates prices, and triggers the strategy and management cycles.

## Public API (SimulationRunner)

### `SimulationRunner(symbols, timeframes, data_files, initial_balance, ...)`
- **data_files** (dict): Mapping of `(symbol, timeframe)` to CSV paths.

### `run()`
Executes the simulation from the first available bar to the last.

### `generate_report()`
Triggers the `StatisticsEngine` and saves the final performance summary.

## Interaction with Other Modules
- **MMStrategy**: Runs unchanged in the simulation, receiving data from `HistoricalDataFeed`.
- **ExitManager**: Reconciles against the `SimulationBroker` exactly as it would with MT5.
- **PositionManager**: Sends virtual orders to the `SimulationBroker`.

## Best Practices
- **Fidelity**: Always include both M5 and M15 data if the strategy uses multi-timeframe indicators, even if only one timeframe is used for signals.
- **Spread**: The simulation uses the `Spread` column from the CSV to calculate bid/ask prices realistically.
