import gym
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import ForexEnv
from pathlib import Path
import sys
import os

# Add Colecting_Data to sys.path to use utils
base_path = Path(__file__).resolve().parent.parent
sys.path.append(str(base_path / "Colecting_Data"))
try:
    from utils import add_technical_indicators
except ImportError:
    def add_technical_indicators(df): return df

def train_ppo():
    # Load data
    data_path = base_path / "Data" / "GBPUSD_1h.csv"
    if not data_path.exists():
        print(f"Data not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    df = add_technical_indicators(df)

    # Split data
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    # Create environments
    env = DummyVecEnv([lambda: ForexEnv(train_df)])
    test_env = DummyVecEnv([lambda: ForexEnv(test_df)])

    # Initialize PPO agent
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_forex_tensorboard/")

    # Train the agent
    print("Starting training...")
    model.learn(total_timesteps=100000)

    # Save the model
    model_path = base_path / "RL_Approach" / "ppo_forex_model"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Evaluate the agent
    obs = test_env.reset()
    for _ in range(len(test_df) - 25):
        action, _states = model.predict(obs)
        obs, rewards, done, info = test_env.step(action)
        if done:
            break

    print("Evaluation complete.")

if __name__ == "__main__":
    try:
        train_ppo()
    except ImportError as e:
        print(f"Requirement missing: {e}")
        print("Please install stable-baselines3: pip install stable-baselines3")
