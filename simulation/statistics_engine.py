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
        executed_df = self.df[self.df['execution_ticket'] > 0]
        rejected_count = total_signals - len(executed_df)

        total_trades = len(executed_df)
        wins = executed_df[executed_df['outcome_result'] == 'WIN']
        losses = executed_df[executed_df['outcome_result'] == 'LOSS']
        breakevens = executed_df[executed_df['outcome_result'] == 'BREAKEVEN']
        open_trades = executed_df[executed_df['outcome_result'] == 'open']

        win_count = len(wins)
        loss_count = len(losses)
        be_count = len(breakevens)
        open_count = len(open_trades)

        closed_trades_count = win_count + loss_count + be_count
        win_rate = (win_count / closed_trades_count * 100) if closed_trades_count > 0 else 0

        gross_profit = wins['outcome_realized_profit'].sum()
        gross_loss = abs(losses['outcome_realized_profit'].sum())
        net_profit = gross_profit - gross_loss

        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        expectancy = (net_profit / closed_trades_count) if closed_trades_count > 0 else 0

        avg_win = wins['outcome_realized_profit'].mean() if win_count > 0 else 0
        avg_loss = losses['outcome_realized_profit'].mean() if loss_count > 0 else 0

        # Duration is only meaningful for closed trades
        avg_duration = executed_df[executed_df['outcome_result'] != 'open']['outcome_duration'].mean() if closed_trades_count > 0 else 0

        results = executed_df['outcome_result'].tolist()
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
            elif res == 'BREAKEVEN':
                current_wins = 0
                current_losses = 0
            # 'open' results don't break streaks yet as they are not outcomes

        return {
            "total_signals": total_signals,
            "total_trades": total_trades,
            "rejected_count": rejected_count,
            "win_count": win_count,
            "loss_count": loss_count,
            "be_count": be_count,
            "open_count": open_count,
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
