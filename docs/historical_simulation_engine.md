# Historical Simulation Engine (Module 11)

## Architecture

The Historical Simulation Engine is designed to allow the existing Forex Trading Framework to operate on historical data using nearly the same execution path as live trading.

### Components

- **SimulationClock**: Maintains simulated time and allows advancing it.
- **HistoricalDataFeed**: Serves historical OHLCV data from CSV files.
- **SimulationAccount**: Tracks simulated balance, equity, margin, and drawdown.
- **SimulationBroker**: Replaces MT5 execution for backtesting, managing positions and deals.
- **SimulationEnvironment**: Provides a unified interface for time, account info, and broker data, abstracting the differences between live and backtest modes.
- **SimulationOrderEngine**: A drop-in replacement for `SendOrder` that interacts with the simulated environment.
- **SimulationRunner**: Drives the simulation loop (Advance Candle -> Poll Strategy -> Update Components).
- **StatisticsEngine**: Calculates performance metrics from completed `PositionLifecycle` objects.
- **BacktestReport**: Generates formatted reports.

## Execution Flow

1. **Initialization**: Load historical data, set up simulation modules, and initialize the trading strategy.
2. **Simulation Loop**:
   - Advance `SimulationClock`.
   - Advance `HistoricalDataFeed`.
   - Update `DrawdownManager`.
   - Poll `MMStrategy` for new signals.
   - Poll `PositionTracker` to update open positions.
   - Poll `ExitManager` to manage exits and partial closes.
3. **Finalization**:
   - Process all completed `PositionLifecycle` objects.
   - Calculate statistics.
   - Generate a backtest report.

## Live vs Backtest Comparison

| Feature | Live Mode | Backtest Mode |
|---------|-----------|---------------|
| **Data Feed** | `MT5DataFeed` (MT5 API) | `HistoricalDataFeed` (CSV) |
| **Clock** | `datetime.now()` | `SimulationClock` |
| **Execution** | `PositionManager` (MT5 API) | `SimulationBroker` |
| **Order Engine** | `SendOrder` | `SimulationOrderEngine` |
| **Account Info** | `mt5.account_info()` | `SimulationAccount` |

## Simulation Assumptions (Phase 1)

- **Immediate Fill**: Orders are filled immediately at the requested price.
- **No Slippage/Spread**: Simulations assume zero slippage and zero spread.
- **No Commissions**: No broker commissions are applied.
- **M1 Ticks**: For backtests, M1 close prices are used to simulate tick arrival.

## Future Extensions

- **Realistic Fill Models**: Support for slippage, spread simulation, and commission models.
- **Tick-Level Reconstruction**: Higher fidelity simulation using tick data.
- **Genetic Optimization**: Automated strategy parameter tuning.
- **Monte Carlo Analysis**: Statistical evaluation of strategy robustness.
