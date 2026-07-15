import pandas as pd
import numpy as np
from typing import List, Dict, Any
from Collecting_Data.position_lifecycle import PositionLifecycle

class StatisticsEngine:
    def __init__(self, lifecycles: List[PositionLifecycle]):
        self.lifecycles = lifecycles
        self.df = self._to_dataframe()

    def _to_dataframe(self) -> pd.DataFrame:
        if not self.lifecycles:
            return pd.DataFrame()
        rows = [lc.to_csv_row() for lc in self.lifecycles]
        return pd.DataFrame(rows)

    def calculate_metrics(self) -> Dict[str, Any]:
        if self.df.empty:
            return {}

        total_signals = len(self.df)
        executed_trades = self.df[self.df['execution_ticket'] > 0]
        failed_signals = self.df[self.df['outcome_result'] == 'FAILED']
        open_positions = self.df[self.df['outcome_result'] == 'open']

        closed_executed_trades = executed_trades[
            (executed_trades['outcome_result'] == 'WIN') |
            (executed_trades['outcome_result'] == 'LOSS') |
            (executed_trades['outcome_result'] == 'BREAKEVEN')
        ]

        wins = closed_executed_trades[closed_executed_trades['outcome_result'] == 'WIN']
        losses = closed_executed_trades[closed_executed_trades['outcome_result'] == 'LOSS']
        breakevens = closed_executed_trades[closed_executed_trades['outcome_result'] == 'BREAKEVEN']

        win_count = len(wins)
        loss_count = len(losses)
        be_count = len(breakevens)
        failed_count = len(failed_signals)
        open_count = len(open_positions)
        executed_count = len(executed_trades)
        closed_count = len(closed_executed_trades)

        win_rate = (win_count / closed_count * 100) if closed_count > 0 else 0

        gross_profit = wins['outcome_realized_profit'].sum()
        gross_loss = abs(losses['outcome_realized_profit'].sum())
        net_profit = gross_profit - gross_loss

        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        expectancy = (net_profit / closed_count) if closed_count > 0 else 0

        avg_win = wins['outcome_realized_profit'].mean() if win_count > 0 else 0
        avg_loss = losses['outcome_realized_profit'].mean() if loss_count > 0 else 0

        avg_duration = closed_executed_trades['outcome_duration'].mean() if closed_count > 0 else 0

        results = closed_executed_trades['outcome_result'].tolist()
        max_cons_wins = 0
        max_cons_losses = 0
        current_wins = 0
        current_losses = 0

        for res in results:
            if res == 'WIN':
                current_wins += 1
                current_losses = 0
                max_cons_wins = max(max_cons_wins, current_wins)
            elif res == 'LOSS':
                current_losses += 1
                current_wins = 0
                max_cons_losses = max(max_cons_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0

        return {
            "total_signals": total_signals,
            "executed_trades_count": executed_count,
            "failed_count": failed_count,
            "open_count": open_count,
            "closed_trades_count": closed_count,
            "win_count": win_count,
            "loss_count": loss_count,
            "be_count": be_count,
            "win_rate": win_rate,
            "net_profit": net_profit,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_duration_seconds": avg_duration,
            "max_consecutive_wins": max_cons_wins,
            "max_consecutive_losses": max_cons_losses
        }
