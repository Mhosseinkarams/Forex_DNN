from typing import Dict, Any

class BacktestReport:
    """
    Purpose:
        A utility class for generating human-readable performance summaries
        from backtest metrics.
    """
    @staticmethod
    def generate_summary(strategy_name: str, symbol: str, timeframe: str, metrics: Dict[str, Any]) -> str:
        """
        Purpose:
            Formats a metrics dictionary into a structured text report.

        Arguments:
            strategy_name (str): The name of the tested strategy.
            symbol (str): The symbol traded.
            timeframe (str): The timeframe of the simulation.
            metrics (dict): Dictionary from StatisticsEngine.calculate_metrics().

        Returns:
            str: A multi-line formatted string summary.
        """
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
            "Trades",
            "------------------------------------------------",
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
