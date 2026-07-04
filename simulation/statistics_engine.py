import pandas as pd
import numpy as np
from typing import List
from Collecting_Data.position_lifecycle import PositionLifecycle

class StatisticsEngine:
    @staticmethod
    def calculate_metrics(lifecycles: List[PositionLifecycle], initial_balance: float):
        if not lifecycles:
            return {}

        df_list = [l.to_csv_row() for l in lifecycles]
        df = pd.DataFrame(df_list)

        # Helper to ensure numeric
        def to_num(col): return pd.to_numeric(df[col], errors='coerce').fillna(0)

        profits = to_num('outcome_realized_profit')
        wins = profits[profits > 0]
        losses = profits[profits < 0]
        be = profits[profits == 0]

        total_trades = len(df)
        win_count = len(wins)
        loss_count = len(losses)
        be_count = len(be)

        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())
        net_profit = gross_profit - gross_loss

        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        avg_win = wins.mean() if win_count > 0 else 0
        avg_loss = losses.mean() if loss_count > 0 else 0
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

        # Drawdown from equity curve
        # Note: This is closed-trade drawdown.
        balance_curve = [initial_balance]
        current = initial_balance
        for p in profits:
            current += p
            balance_curve.append(current)

        balance_curve = pd.Series(balance_curve)
        peak = balance_curve.cummax()
        drawdown = (peak - balance_curve) / peak * 100
        max_drawdown = drawdown.max()

        # Consecutive wins/losses
        results = (profits > 0).astype(int).tolist()
        max_cons_wins = 0
        max_cons_losses = 0
        curr_wins = 0
        curr_losses = 0

        for r in results:
            if r == 1:
                curr_wins += 1
                curr_losses = 0
                max_cons_wins = max(max_cons_wins, curr_wins)
            elif r == 0 and profits.iloc[results.index(r)] < 0: # Loss
                curr_losses += 1
                curr_wins = 0
                max_cons_losses = max(max_cons_losses, curr_losses)
            else: # Breakeven
                curr_wins = 0
                curr_losses = 0

        # R-Multiple
        r_multiples = to_num('outcome_r_multiple')
        avg_r = r_multiples.mean()

        # Duration
        durations = to_num('outcome_duration')
        avg_duration = durations.mean()

        return {
            "total_trades": total_trades,
            "wins": win_count,
            "losses": loss_count,
            "breakeven": be_count,
            "win_rate": win_rate,
            "net_profit": net_profit,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "max_drawdown_pct": max_drawdown,
            "max_cons_wins": max_cons_wins,
            "max_cons_losses": max_cons_losses,
            "avg_r_multiple": avg_r,
            "avg_duration_seconds": avg_duration,
            "final_balance": balance_curve.iloc[-1]
        }

    @staticmethod
    def calculate_metrics_from_df(df: pd.DataFrame, initial_balance: float):
        if df.empty:
            return {}

        # Similar logic to calculate_metrics but directly from flattened DF
        def to_num(col): return pd.to_numeric(df[col], errors='coerce').fillna(0)

        profits = to_num('outcome_realized_profit')
        wins = profits[profits > 0]
        losses = profits[profits < 0]
        be = profits[profits == 0]

        total_trades = len(df)
        win_count = len(wins)
        loss_count = len(losses)
        be_count = len(be)

        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())
        net_profit = gross_profit - gross_loss

        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        avg_win = wins.mean() if win_count > 0 else 0
        avg_loss = losses.mean() if loss_count > 0 else 0
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

        balance_curve = [initial_balance]
        current = initial_balance
        for p in profits:
            current += p
            balance_curve.append(current)

        balance_curve = pd.Series(balance_curve)
        peak = balance_curve.cummax()
        drawdown = (peak - balance_curve) / peak * 100
        max_drawdown = drawdown.max()

        max_cons_wins = 0
        max_cons_losses = 0
        curr_wins = 0
        curr_losses = 0

        for p in profits:
            if p > 0:
                curr_wins += 1
                curr_losses = 0
                max_cons_wins = max(max_cons_wins, curr_wins)
            elif p < 0:
                curr_losses += 1
                curr_wins = 0
                max_cons_losses = max(max_cons_losses, curr_losses)
            else:
                curr_wins = 0
                curr_losses = 0

        avg_r = to_num('outcome_r_multiple').mean()
        avg_duration = to_num('outcome_duration').mean()

        return {
            "total_trades": total_trades,
            "wins": win_count,
            "losses": loss_count,
            "breakeven": be_count,
            "win_rate": win_rate,
            "net_profit": net_profit,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "max_drawdown_pct": max_drawdown,
            "max_cons_wins": max_cons_wins,
            "max_cons_losses": max_cons_losses,
            "avg_r_multiple": avg_r,
            "avg_duration_seconds": avg_duration,
            "final_balance": balance_curve.iloc[-1]
        }
