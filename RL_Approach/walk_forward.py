import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import ForexEnv
from metrics import calculate_metrics, print_metrics
from pathlib import Path

# Set up paths
base_path = Path(__file__).resolve().parent.parent
data_path = base_path / "Data"

def run_walk_forward(n_windows=4, train_hours=4000, test_hours=1000):
    """
    Implements Walk-Forward Optimization (WFO).
    Trains on a rolling window and tests on the subsequent period.
    """
    # Prefer enriched data with LSTM signals
    input_file = data_path / "GBPUSD_1h_enriched.csv"
    if not input_file.exists():
        input_file = data_path / "GBPUSD_1h.csv"
        print("Warning: Enriched data not found. Using raw data for WFO.")

    df = pd.read_csv(input_file)
    total_len = len(df)

    all_window_results = []

    # Market frictions for realistic simulation
    frictions = {
        'spread': 0.0002,
        'commission': 0.00005,
        'slippage': 0.0001
    }

    for i in range(n_windows):
        # Calculate indices
        # We move forward by test_hours each time
        start_train = i * test_hours
        end_train = start_train + train_hours
        start_test = end_train
        end_test = start_test + test_hours

        if end_test > total_len:
            print(f"Stopping at window {i+1}: end_test exceeds data length.")
            break

        train_df = df.iloc[start_train:end_train]
        test_df = df.iloc[start_test:end_test]

        print(f"\n{'='*20}")
        print(f"WINDOW {i+1} / {n_windows}")
        print(f"Train: {start_train} -> {end_train} ({len(train_df)} hrs)")
        print(f"Test : {start_test} -> {end_test} ({len(test_df)} hrs)")
        print(f"{'='*20}")

        # Training
        train_env = DummyVecEnv([lambda: ForexEnv(train_df, **frictions)])
        model = PPO("MlpPolicy", train_env, verbose=0,
                    learning_rate=3e-4,
                    n_steps=1024,
                    batch_size=64)

        print(f"Training window {i+1}...")
        model.learn(total_timesteps=80000)

        # Testing
        test_env = ForexEnv(test_df, **frictions)
        obs = test_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)

        # Evaluation
        metrics = calculate_metrics(test_env.equity_curve, test_env.returns)
        print_metrics(metrics)
        all_window_results.append(metrics)

    if not all_window_results:
        print("No windows were processed.")
        return

    # Aggregate Results
    summary = {}
    for key in all_window_results[0].keys():
        values = [res[key] for res in all_window_results if not np.isinf(res[key])]
        summary[key] = np.mean(values) if values else 0

    print("\n" + "#"*40)
    print("WALK-FORWARD AGGREGATE SUMMARY")
    print("#"*40)
    print_metrics(summary)

if __name__ == "__main__":
    run_walk_forward()
