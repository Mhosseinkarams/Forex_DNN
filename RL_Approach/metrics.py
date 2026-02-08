import numpy as np
import pandas as pd

def calculate_metrics(equity_curve, returns):
    """
    Calculates key financial metrics from equity curve and step-wise returns.

    Args:
        equity_curve: List or array of net worth over time.
        returns: List or array of step-wise percentage returns.

    Returns:
        Dictionary of financial metrics.
    """
    equity_curve = np.array(equity_curve)
    returns = np.array(returns)

    if len(equity_curve) < 2:
        return {}

    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

    # Annualization factor (assuming hourly data, 24/7 or 24/5)
    # Forex usually 24/5 ~ 6240 hours/year
    periods_per_year = 24 * 260

    # Sharpe Ratio
    std_returns = np.std(returns)
    if std_returns != 0:
        sharpe_ratio = (np.mean(returns) / std_returns) * np.sqrt(periods_per_year)
    else:
        sharpe_ratio = 0

    # Sortino Ratio
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        std_downside = np.std(downside_returns)
        if std_downside != 0:
            sortino_ratio = (np.mean(returns) / std_downside) * np.sqrt(periods_per_year)
        else:
            sortino_ratio = 0
    else:
        sortino_ratio = np.inf if np.mean(returns) > 0 else 0

    # Maximum Drawdown
    peak = np.maximum.accumulate(equity_curve)
    # Avoid division by zero if equity is 0
    peak[peak == 0] = 1e-9
    drawdowns = (peak - equity_curve) / peak
    max_drawdown = np.max(drawdowns)

    # Profit Factor
    profits = returns[returns > 0]
    losses = np.abs(returns[returns < 0])
    gross_profit = np.sum(profits)
    gross_loss = np.sum(losses)
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf

    # Win Rate and Expectancy
    win_rate = len(profits) / len(returns) if len(returns) > 0 else 0
    avg_win = np.mean(profits) if len(profits) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # Calmar Ratio
    calmar_ratio = total_return / max_drawdown if max_drawdown != 0 else np.inf

    return {
        "Total Return": total_return,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown": max_drawdown,
        "Profit Factor": profit_factor,
        "Expectancy": expectancy,
        "Calmar Ratio": calmar_ratio,
        "Win Rate": win_rate
    }

def print_metrics(metrics):
    print("\n--- Financial Metrics ---")
    for k, v in metrics.items():
        if "Ratio" in k or "Factor" in k or "Expectancy" in k:
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v*100:.2f}%")
    print("-------------------------\n")
