# Configuration Guide

This document provides a comprehensive reference for all framework configuration parameters.

## Strategy Parameters (`MMStrategy`)

| Parameter | Purpose | Default | Recommended |
| :--- | :--- | :--- | :--- |
| `symbols` | List of symbols to trade (e.g., `["EURUSD_o"]`). | Required | - |
| `poll_interval_seconds` | How often to check for new bars. | `5.0` | `1.0` - `5.0` |
| `swing_lookback` | Bars to look back for SL swing high/low. | `10` | `10` - `20` |
| `max_sl_pips` | Maximum allowed SL distance in pips. | `25` | `20` - `30` |
| `m5_slope_threshold` | Minimum EMA 600 slope for M5 trend. | `0.1` | `0.1` |
| `m15_slope_threshold` | Minimum EMA 800 slope for M15 trend. | `0.1` | `0.1` |
| `reversal_ema_sep_threshold` | Min ATR-dist between EMAs for Reversal. | `9.0` | `8.0` - `12.0` |

## Risk & Drawdown (`DrawdownManager`)

| Parameter | Purpose | Default | Recommended |
| :--- | :--- | :--- | :--- |
| `daily_limit_pct` | Max daily loss before blocking trading. | `0.03` (3%) | `0.02` - `0.05` |
| `total_limit_pct` | Max total loss from initial balance. | `0.10` (10%) | `0.10` - `0.20` |

### Per-Signal Risk Caps (`SendOrder`)
The system applies hard-coded risk caps based on signal category:
- **Standard**: 1.0%
- **High-Risk**: 0.5%
- **Reversal**: 0.3%

## Exit Profiles (`PositionLifecycle`)

| Profile | Behavior | Logic |
| :--- | :--- | :--- |
| `EXIT_PROFILE_STANDARD` | Multi-staged exit. | TP1 (1R) -> Close 50% & SL to BE. TP2 (2R) -> Close remaining. |
| `EXIT_PROFILE_SINGLE` | Simple exit. | TP1 (1R) -> Close 100%. |

## Data Feed (`MT5DataFeed`)

| Constant | Purpose | Value |
| :--- | :--- | :--- |
| `LATENCY_WARN_MS` | Log warning if API takes longer than this. | `50` |
| `LATENCY_PAUSE_MS` | Mark feed DEGRADED if latency exceeds this. | `200` |
| `DATA_STALE_S` | Mark feed DEGRADED if last tick is older than this. | `10` |

## Polling Intervals

| Module | Purpose | Frequency |
| :--- | :--- | :--- |
| `PositionTracker` | Real-time risk monitoring. | `5.0s` |
| `ExitManager` | TP/SL hit evaluation. | `1.0s` |
| `MMStrategy` | Signal detection. | `5.0s` |

## Magic Numbers

It is critical to use unique magic numbers for different systems:
- `MAGIC_UNITY`: `100001` (Reserved for future unified model).
- `MAGIC_MM`: `100002` (Current MM Strategy).

## File Paths

- `journal_root`: Defaults to `Journals/`.
- `state_dir`: Defaults to `State/`.
- `log_dir`: Defaults to `Logs/`.

## Broker Specifics (LiteFinance)

- **Symbol Suffix**: All symbols must end in `_o` (e.g., `GBPUSD_o`).
- **Volume Step**: Typically `0.01`.
- **Stops Level**: Distance from price where SL/TP cannot be placed. The framework automatically adjusts SL if it's too close to this level.
