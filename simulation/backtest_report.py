from typing import Dict, Any

class BacktestReport:
    @staticmethod
    def generate_summary(strategy_name: str, symbol: str, timeframe: str, metrics: Dict[str, Any]) -> str:
        if not metrics:
            return "NO TRADES EXECUTED"

        report = [
            "================================================",
            "                BACKTEST REPORT                 ",
            "================================================",
            f"Strategy:       {strategy_name}",
            f"Symbol:         {symbol}",
            f"Timeframe:      {timeframe}",
            "------------------------------------------------",
            "Signals & Trades",
            "------------------------------------------------",
            f"Total Signals:  {metrics.get('total_signals', 'N/A')}",
            f"Failed Signals: {metrics.get('failed_count', 'N/A')}",
            f"Total Trades:   {metrics['total_trades']}",
            f"Wins:           {metrics['win_count']}",
            f"Losses:         {metrics['loss_count']}",
            f"BreakEven:      {metrics['be_count']}",
            f"Win Rate:       {metrics['win_rate']:.2f}%",
            "------------------------------------------------",
            "Performance",
            "------------------------------------------------",
            f"Net Profit:     ${metrics['net_profit']:.2f}",
            f"Profit Factor:  {metrics['profit_factor']:.2f}",
            f"Expectancy:     ${metrics['expectancy']:.2f}",
            f"Avg Win:        ${metrics['avg_win']:.2f}",
            f"Avg Loss:       ${metrics['avg_loss']:.2f}",
            "------------------------------------------------",
            "Risk",
            "------------------------------------------------",
            f"Consecutive Wins:   {metrics['max_consecutive_wins']}",
            f"Consecutive Losses: {metrics['max_consecutive_losses']}",
            "================================================"
        ]
        return "\n".join(report)
