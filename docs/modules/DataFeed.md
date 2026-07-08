# DataFeed Module

## Purpose
The `DataFeed` module provides a unified interface for retrieving market data in both live trading and historical backtesting environments. It abstracts the complexities of the MetaTrader 5 API and handles connection health monitoring.

## Responsibilities
- **Connection Management**: Handling `initialize` and `shutdown` of the MT5 terminal.
- **Health Monitoring**: Measuring API latency and data freshness to detect "degraded" market conditions.
- **Data Retrieval**: Providing OHLCV data with standardized schemas.
- **Resampling**: Converting base M1 data into higher timeframes (M5, M15, etc.).

## Public API

### `MT5DataFeed(login, password, server)`
**Constructor**
- **login** (int): MT5 account ID.
- **password** (str): MT5 account password.
- **server** (str): Broker server name.
- **Note**: Automatically loads credentials from `credentials.json` if not provided.

### `connect() -> bool`
Initializes the MT5 connection. Returns `True` if successful.

### `get_ohlcv(symbol, timeframe_str, count) -> pd.DataFrame`
Retrieves the latest `count` candles for a given symbol and timeframe string (e.g., "M5").

### `check_health(symbol) -> FeedHealth`
Measures API latency and tick age. Returns `HEALTHY`, `DEGRADED`, or `DISCONNECTED`.

## Usage Example

```python
from Collecting_Data.data_feed import MT5DataFeed

feed = MT5DataFeed()
if feed.connect():
    df = feed.get_ohlcv("EURUSD_o", "M5", count=100)
    print(df.head())
```

## Interaction with Other Modules
- **MMStrategy**: Consumes OHLCV data to evaluate signal logic.
- **DrawdownManager**: Uses the feed to get the current server time for day-rollover detection.
- **Historical Simulation**: Replaced by `HistoricalDataFeed` in backtest mode.

## Best Practices
- Always check `feed.connect()` before attempting data retrieval.
- Use `wait_for_healthy()` in live trading loops to pause during high-latency periods.
