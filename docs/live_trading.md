# Live Trading Guide

This guide covers the setup, execution, and monitoring of the framework in a production environment.

## Environment Setup

### 1. Requirements
- Windows OS (Required for the `MetaTrader5` Python library).
- MetaTrader 5 Terminal installed and logged into a broker account.
- Stable internet connection.

### 2. MT5 Configuration
- Go to **Tools > Options > Expert Advisors**.
- Check **"Allow Algorithmic Trading"**.
- Check **"Allow WebRequest for listed URL"** (Optional, if using external APIs).

### 3. Framework Configuration
Create a `credentials.json` file in the project root:
```json
{
  "mt5": {
    "login": 12345678,
    "password": "your_password",
    "server": "Broker-Server"
  }
}
```

## Running the System

To start the trading system, run the `main.py` script:
```bash
python main.py
```

### Expected Startup Logs
The console should show the following sequence:
1.  MT5 Connection Success.
2.  Account Balance retrieval.
3.  Module Initialization (Tracker, Drawdown, ExitManager, etc.).
4.  Background Thread Start (Tracker, ExitManager, Strategy).
5.  Active Monitoring message.

## Monitoring

### Logs
All logs are saved in the `Logs/` directory.
- `Main.log`: High-level system status.
- `PositionManager.log`: Order execution details.
- `ExitManager.log`: TP/SL modification events.

### Journals
Real-time events are logged in `Journals/live/events/`.
- Open `*_events.csv` to see signals as they are detected.
- Completed trades are summarized in `Journals/live/positions/`.

### Dashboard (Console)
The `main.py` loop provides periodic updates on:
- Current account balance and equity.
- Open risk and remaining drawdown limits.
- Number of active positions.

## Position Management

The framework manages positions automatically:
- **Partial Closes**: Executed by `ExitManager` when TP levels are hit.
- **Breakeven**: Stop-loss moved to entry price after TP1.
- **Manual Intervention**: If you manually close a position in MT5, `ExitManager` will detect the disappearance, determine the reason from broker deals, and finalize the journal entry automatically.

## Shutdown & Restart

### Graceful Shutdown
Press `Ctrl+C` in the terminal. The system will:
1.  Stop the strategy polling loop.
2.  Stop background management threads.
3.  Save the current state to `State/*.json`.
4.  Disconnect from MT5.

### Recovery
Upon restart, the framework:
1.  Loads state files from `State/`.
2.  Queries the broker for all open positions.
3.  Re-registers positions with `ExitManager` to resume staged management.
4.  Calculates new drawdown limits based on the current balance.

## TradeAuditor

To investigate a trade that occurred during the live session:
```bash
python trade_auditor.py --mode live --latest
```

## Common Problems & Solutions

### "MT5 Init Failed"
- Ensure the MT5 terminal is open and logged in.
- Verify that `credentials.json` contains the correct login/password/server.
- Check if your broker requires a specific server suffix (e.g., `_o` for LiteFinance).

### "Trade Blocked by Drawdown"
- The daily loss limit has been reached. The system will resume trading at the next daily rollover (00:00 server time).
- You can manually reset this by deleting `State/drawdown_state.json` (Not recommended in production).

### "Market Closed"
- Signals detected during the weekend or bank holidays will be logged but `SendOrder` will fail with an MT5 error. The framework handles this gracefully and waits for the next bar.

## Production Checklist
1.  [ ] MT5 is open and "Algo Trading" is enabled (Green icon).
2.  [ ] `credentials.json` is configured correctly.
3.  [ ] `State/` directory is writable.
4.  [ ] System clock is synchronized.
5.  [ ] Log rotation is configured (if running for long periods).
