import gym
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path
import sys
import os

class PPOTradingAgent:
    """
    A class to train and evaluate a PPO agent for Forex trading.
    """

    def __init__(self, use_enriched=True):
        self.base_path = Path(__file__).resolve().parent.parent
        self.data_path = self.base_path / "Data"
        self.use_enriched = use_enriched
        self.model = None

        # Add RL_Approach to path to find trading_env
        rl_path = str(self.base_path / "RL_Approach")
        if rl_path not in sys.path:
            sys.path.append(rl_path)
        from trading_env import ForexEnv
        self.ForexEnv = ForexEnv

    def load_data(self):
        filename = "GBPUSD_1h_enriched.csv" if self.use_enriched else "GBPUSD_1h.csv"
        input_file = self.data_path / filename
        if not input_file.exists():
            return None
        return pd.read_csv(input_file)

    def train(self, total_timesteps=100000, model_name="ppo_forex_model"):
        df = self.load_data()
        if df is None:
            return

        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]

        env = DummyVecEnv([lambda: self.ForexEnv(train_df)])

        self.model = PPO("MlpPolicy", env, verbose=1,
                         learning_rate=3e-4, n_steps=2048, batch_size=64,
                         tensorboard_log=str(self.base_path / "ppo_forex_tensorboard"))

        print(f"Starting training for {total_timesteps} steps...")
        self.model.learn(total_timesteps=total_timesteps)

        model_path = self.base_path / "RL_Approach" / model_name
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def evaluate(self, df_test=None):
        if df_test is None:
            df = self.load_data()
            if df is None: return
            train_size = int(len(df) * 0.8)
            df_test = df.iloc[train_size:]

        env = self.ForexEnv(df_test)
        obs = env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

        return env.equity_curve, env.returns

if __name__ == "__main__":
    agent = PPOTradingAgent()
    # Note: For actual use, call agent.train()
    print("PPOTradingAgent class ready.")
