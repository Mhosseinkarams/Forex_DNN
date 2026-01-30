import gym
from gym import spaces
import numpy as np
import pandas as pd

class ForexEnv(gym.Env):
    """Custom Environment for Forex Trading that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size=24, initial_balance=1000):
        super(ForexEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance

        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: Price window + account info
        # Number of features in df
        self.features = [c for c in df.columns if c not in ['DTYYYYMMDD', '<Time>', 'Datetime', 'Classification']]
        self.num_features = len(self.features)

        # Shape: (window_size, num_features) + (balance, position)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, self.num_features + 2), dtype=np.float32
        )

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_step = self.window_size

        return self._next_observation()

    def _next_observation(self):
        # Get the data for the last window_size steps
        frame = self.df.loc[self.current_step - self.window_size: self.current_step - 1, self.features].values

        # Append additional info (balance, shares held)
        obs = np.column_stack((
            frame,
            np.full(self.window_size, self.balance / self.initial_balance),
            np.full(self.window_size, self.shares_held)
        ))

        return obs.astype(np.float32)

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        self.current_step += 1

        if self.current_step > len(self.df) - 1:
            done = True
        else:
            done = False

        delay_modifier = (self.current_step / len(self.df))
        reward = self.net_worth - self.initial_balance

        obs = self._next_observation()

        return obs, reward, done, {}

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = self.df.loc[self.current_step, "Close"]

        if action == 1: # Buy
            if self.balance > 0:
                self.shares_held = self.balance / current_price
                self.balance = 0
                self.cost_basis = current_price

        elif action == 2: # Sell
            if self.shares_held > 0:
                self.balance = self.shares_held * current_price
                self.shares_held = 0
                self.total_shares_sold += self.shares_held
                self.total_sales_value += self.balance

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held}')
        print(f'Net Worth: {self.net_worth}')
        print(f'Profit: {profit}')
