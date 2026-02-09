import gym
from gym import spaces
import numpy as np
import pandas as pd

class ForexEnv(gym.Env):
    """
    Custom Environment for Forex Trading with realistic market frictions.
    Incorporates spreads, commissions, slippage, and advanced reward shaping.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size=24, initial_balance=1000,
                 spread=0.0002, commission=0.00005, slippage=0.0001,
                 reward_shaping=True):
        super(ForexEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance

        # Market frictions
        self.spread = spread
        self.commission = commission
        self.slippage = slippage
        self.reward_shaping = reward_shaping

        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: Price window + account info
        # We exclude labels from features
        exclude_cols = ['DTYYYYMMDD', '<Time>', 'Datetime', 'Classification', 'Binary_Label', 'Multi_Label', 'Pivot_Label', 'Price_Change', 'Peak', 'Trough']
        self.features = [c for c in df.columns if c not in exclude_cols]
        self.num_features = len(self.features)

        # Shape: (window_size, num_features) + 4 account features (balance, shares, unrealized_pnl, equity)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, self.num_features + 4), dtype=np.float32
        )

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.current_step = self.window_size
        self.returns = []
        self.equity_curve = [self.initial_balance]

        return self._next_observation()

    def _next_observation(self):
        # Extract the window of features
        frame = self.df.loc[self.current_step - self.window_size: self.current_step - 1, self.features].values

        # Account info normalization (relative to initial balance)
        current_price = self.df.loc[self.current_step - 1, "Close"]
        unrealized_pnl = (current_price - self.cost_basis) * self.shares_held if self.shares_held > 0 else 0

        # Normalize shares_held as a fraction of current net worth
        position_ratio = (self.shares_held * current_price) / self.net_worth if self.net_worth > 0 else 0

        account_info = np.column_stack((
            np.full(self.window_size, self.balance / self.initial_balance),
            np.full(self.window_size, position_ratio),
            np.full(self.window_size, unrealized_pnl / self.initial_balance),
            np.full(self.window_size, self.net_worth / self.initial_balance)
        ))

        obs = np.column_stack((frame, account_info))
        return obs.astype(np.float32)

    def step(self, action):
        old_net_worth = self.net_worth

        # Execute trade
        self._take_action(action)

        self.current_step += 1

        # Check if done
        done = self.current_step > len(self.df) - 1

        # Update net worth based on new price
        if not done:
            current_price = self.df.loc[self.current_step, "Close"]
            self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        # Reward calculation
        step_return = (self.net_worth - old_net_worth) / old_net_worth
        self.returns.append(step_return)

        reward = step_return

        if self.reward_shaping:
            # Drawdown penalty
            drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
            reward -= 0.1 * (drawdown ** 2)

            # Penalize excessive inactivity if balance is just sitting there
            # (Optional: depends on strategy)

        self.equity_curve.append(self.net_worth)

        obs = self._next_observation()
        info = {
            'net_worth': self.net_worth,
            'max_net_worth': self.max_net_worth,
            'step_return': step_return
        }

        return obs, reward, done, info

    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, "Close"]

        if action == 1: # Buy
            if self.balance > 0:
                # Add spread and slippage to buy price
                entry_price = current_price + self.spread + self.slippage
                # Commission as a percentage of trade value
                trade_value = self.balance
                self.balance -= trade_value * self.commission

                self.shares_held = self.balance / entry_price
                self.balance = 0
                self.cost_basis = entry_price

        elif action == 2: # Sell
            if self.shares_held > 0:
                # Subtract spread and slippage from sell price
                exit_price = current_price - self.spread - self.slippage

                trade_value = self.shares_held * exit_price
                self.balance = trade_value * (1 - self.commission)
                self.shares_held = 0
                self.cost_basis = 0

    def render(self, mode='human', close=False):
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step} | Net Worth: {self.net_worth:.2f} | Profit: {profit:.2f}')
