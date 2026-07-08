# Backtesting Guide

This guide provides a complete tutorial on using the high-fidelity Historical Simulation Engine to validate trading strategies.

## Purpose

The backtesting engine is designed to mirror live trading as closely as possible. It uses the same core modules (`ExitManager`, `PositionTracker`, `SendOrder`) as live trading, but replaces the MT5 connection with a virtual broker and a simulated clock.

## Architecture

The simulation infrastructure consists of:
- **SimulationClock**: Manages virtual time and allows stepping through history.
- **HistoricalDataFeed**: Loads OHLCV data from CSV files and provides it to strategies.
- **SimulationBroker**: Mocks the MetaTrader 5 API, including `order_send`, `positions_get`, and `history_deals_get`.
- **SimulationAccount**: Tracks virtual balance, equity, and margin.
- **SimulationRunner**: The main loop that orchestrates the simulation.

## Simulation Flow

1.  **Initialize**: Load data files and instantiate core modules.
2.  **Environment Setup**: Set `env` to backtest mode, linking it to the virtual broker.
3.  **Main Loop**:
    - Advance the clock.
    - Update virtual market prices.
    - Poll `PositionTracker` and `ExitManager`.
    - Evaluate `MMStrategy` logic.
4.  **Finalize**: Reconstruct all trades via `TradeAuditor` and generate statistics.

## Required Data Format

Historical data should be in CSV format with the following columns:
`Datetime`, `Open`, `High`, `Low`, `Close`, `TickVolume`, `Spread`

- **Datetime**: ISO8601 format (e.g., `2023-10-27 10:00:00`).
- **Spread**: Integer in points.

## How to Start a Simulation

### Basic Example (Single Symbol)

```python
from simulation.simulation_runner import SimulationRunner

# Define data paths
data_files = {
    ("EURUSD_o", "M5"): "Data/EURUSD_M5.csv",
    ("EURUSD_o", "M15"): "Data/EURUSD_M15.csv"
}

# Initialize runner
runner = SimulationRunner(
    symbol="EURUSD_o",
    timeframes=["M5", "M15"],
    data_files=data_files,
    initial_balance=10000.0,
    journal_root="Backtest_Results"
)

# Run simulation
runner.run()
```

### Changing Timeframes

To change the base timeframe of the simulation, ensure your CSV data matches the requested timeframe and update the `timeframes` list in `SimulationRunner`.

### Multiple Symbols

Currently, `SimulationRunner` is optimized for single-symbol backtesting. To run multiple symbols, you can instantiate multiple runners or extend the `SimulationRunner` loop to iterate over multiple symbols per tick.

## Configuration

Backtest behavior can be tuned in `SimulationRunner`:
- **Initial Balance**: Set the starting virtual capital.
- **Leverage**: Set the account leverage (e.g., 100, 500).
- **Journal Root**: Directory where backtest results and logs will be saved.

## Logging & Reports

- **Events**: Detailed logs of every simulation action are saved to `Backtest_Results/backtest/events/`.
- **Reports**: A summary report `backtest_report.txt` is generated in the journal root.
- **Statistics**: Includes Win Rate, Profit Factor, Net Profit, Expectancy, and Max Drawdown.

## Inspecting Completed Trades

After a simulation, use the `TradeAuditor` to inspect specific trades:

```bash
python trade_auditor.py --mode backtest --latest
```

This will show a list of trades from the `Backtest_Results` directory and allow you to view a full forensic reconstruction of any position.

## Debugging Failed Simulations

- **Missing Data**: Ensure all timeframes required by the strategy (e.g., M5 and M15) are provided in `data_files`.
- **Warmup Period**: EMA 600/800 requires significant historical data before the first signal can be generated. Ensure your CSV files have at least 1000 bars of "pre-roll" data.
- **Log Files**: Check `Logs/SimulationRunner.log` for execution errors.

## Complete Workflow Example

1.  **Export Data**: Download EURUSD M5 and M15 data from MT5 to `Data/`.
2.  **Run Simulation**: Execute your runner script.
3.  **Review Report**: Open `Backtest_Results/backtest_report.txt`.
4.  **Forensic Audit**: Select a losing trade and run `trade_auditor.py` to see the exit stages and indicator values at the time of entry.
5.  **Refine**: Adjust strategy parameters in `MMStrategy` and re-run.
