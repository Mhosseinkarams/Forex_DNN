import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path
import sys

class WalkForwardOptimizer:
    """
    A class to perform Walk-Forward Optimization for RL trading agents.
    """

    def __init__(self, n_windows=4, train_hours=4000, test_hours=1000):
        self.base_path = Path(__file__).resolve().parent.parent
        self.data_path = self.base_path / "Data"
        self.n_windows = n_windows
        self.train_hours = train_hours
        self.test_hours = test_hours

        # Imports
        rl_path = str(self.base_path / "RL_Approach")
        if rl_path not in sys.path: sys.path.append(rl_path)
        from trading_env import ForexEnv
        from metrics import TradingMetrics
        self.ForexEnv = ForexEnv
        self.Metrics = TradingMetrics

    def run(self, filename="GBPUSD_1h_enriched.csv"):
        input_file = self.data_path / filename
        if not input_file.exists():
            input_file = self.data_path / "GBPUSD_1h.csv"

        df = pd.read_csv(input_file)
        total_len = len(df)
        all_window_results = []

        frictions = {'spread': 0.0002, 'commission': 0.00005, 'slippage': 0.0001}

        for i in range(self.n_windows):
            start_train = i * self.test_hours
            end_train = start_train + self.train_hours
            start_test = end_train
            end_test = start_test + self.test_hours

            if end_test > total_len: break

            train_df = df.iloc[start_train:end_train]
            test_df = df.iloc[start_test:end_test]

            print(f"\nWINDOW {i+1} / {self.n_windows}")

            train_env = DummyVecEnv([lambda: self.ForexEnv(train_df, **frictions)])
            model = PPO("MlpPolicy", train_env, verbose=0)
            model.learn(total_timesteps=50000)

            test_env = self.ForexEnv(test_df, **frictions)
            obs = test_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)

            metrics = self.Metrics.calculate_metrics(test_env.equity_curve, test_env.returns)
            self.Metrics.print_metrics(metrics)
            all_window_results.append(metrics)

        return all_window_results

if __name__ == "__main__":
    wfo = WalkForwardOptimizer()
    # wfo.run()
    print("WalkForwardOptimizer class ready.")
