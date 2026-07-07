# StatisticsEngine Module

## Purpose
The `StatisticsEngine` is responsible for calculating quantitative performance metrics from a set of completed trades. It provides the mathematical basis for evaluating strategy profitability and risk.

## Responsibilities
- **Data Transformation**: Converting a list of `PositionLifecycle` objects into a Pandas DataFrame.
- **Metric Calculation**: Computing Win Rate, Profit Factor, Expectancy, and Average Win/Loss.
- **Streak Analysis**: Identifying maximum consecutive wins and losses.
- **Reporting**: Formatting metrics into a structured dictionary and summary report.

## Public API

### `StatisticsEngine(lifecycles: list[PositionLifecycle])`
**Constructor**
Initializes with a list of audited trade summaries.

### `calculate_metrics() -> dict`
Returns a dictionary containing:
- `total_trades`, `win_rate`.
- `net_profit`, `gross_profit`, `gross_loss`.
- `profit_factor`, `expectancy`.
- `avg_win`, `avg_loss`, `avg_duration_seconds`.
- `max_consecutive_wins`, `max_consecutive_losses`.

### `BacktestReport.generate_summary(...) -> str`
**Static Method**
Generates a formatted text-based report from the metrics dictionary.

## Interaction with Other Modules
- **SimulationRunner**: Calls the engine at the end of a backtest.
- **PositionLifecycle**: Provides the raw data for calculations.

## Example Output

```text
================================================
                BACKTEST REPORT
================================================
Total Trades:   50
Wins:           30
Losses:         20
Win Rate:       60.00%
Net Profit:     $1250.00
Profit Factor:  1.75
Expectancy:     $25.00
================================================
```

## Best Practices
- **Sample Size**: Ensure the backtest contains at least 30-50 trades before drawing conclusions from the statistics.
- **Expectancy**: Prioritize Expectancy over Win Rate. A high Win Rate with a negative Expectancy indicates poor risk/reward management.
