import gym
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import ForexEnv
from pathlib import Path
import sys

# Set up paths
base_path = Path(__file__).resolve().parent.parent
data_path = base_path / "Data"

def train_ppo(use_enriched=True):
    # Load data
    filename = "GBPUSD_1h_enriched.csv" if use_enriched else "GBPUSD_1h.csv"
    input_file = data_path / filename

    if not input_file.exists():
        if use_enriched:
            print(f"Enriched data not found at {input_file}. Run generate_signals.py first.")
            # Fallback
            input_file = data_path / "GBPUSD_1h.csv"
        else:
            print(f"Data not found at {input_file}")
            return

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    # Temporal split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    # Create environments
    # We can pass custom market friction parameters here
    env = DummyVecEnv([lambda: ForexEnv(train_df, spread=0.0002, commission=0.00005, slippage=0.0001)])

    # Initialize PPO agent with improved hyperparameters for trading
    model = PPO("MlpPolicy", env, verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01, # Encourage exploration
                tensorboard_log="./ppo_forex_tensorboard/")

    print("Starting training...")
    model.learn(total_timesteps=200000)

    # Save the model
    model_path = base_path / "RL_Approach" / "ppo_forex_model"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Evaluate
    test_env = DummyVecEnv([lambda: ForexEnv(test_df, spread=0.0002, commission=0.00005, slippage=0.0001)])
    obs = test_env.reset()

    net_worths = []
    for _ in range(len(test_df) - 25):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = test_env.step(action)
        net_worths.append(info[0]['net_worth'])
        if done:
            break

    final_profit = net_worths[-1] - 1000
    print(f"Evaluation complete. Final Profit: {final_profit:.2f}")

if __name__ == "__main__":
    train_ppo(use_enriched=True)
