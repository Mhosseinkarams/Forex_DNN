# Forex_DNN

A comprehensive Deep Learning and Reinforcement Learning project for Forex trading (GBP/USD), featuring multiple architectures for price prediction and automated trading strategies.

## Project Structure

```text
Forex_DNN/
├── Colecting_Data/         # Data acquisition and preprocessing
│   ├── mt5data.py                # Bridge from MetaTrader 5 to Python
│   ├── preproc_single_inout.py   # Preprocessing for single-output models
│   ├── preproc_multi_inout.py    # Preprocessing for multi-output models
│   ├── preproc_pivot.py          # Preprocessing for pivot/trend change detection
│   └── utils.py                  # Centralized indicators (EMA, RSI, MACD, Ichimoku, etc.)
├── DNN/                    # Deep Neural Network implementations
│   ├── single_inout/       # Dense networks for price prediction
│   │   ├── classifier_binary.py # Predicts direction (Up/Down)
│   │   └── classifier_multi.py  # Predicts direction and movement size
│   ├── multi_inout/        # LSTM-based sequence classification
│   │   ├── classifier_binary.py # Predicts direction using LSTMs
│   │   ├── classifier_multi.py  # Predicts direction and size using LSTMs
│   │   └── classifier_pivot.py  # Predicts major trend pivots
│   ├── Unsup_LSTM/         # LSTM Autoencoders for anomaly detection
│   └── LSTM_GAN/           # GANs for synthetic data and price forecasting
├── RL_Approach/            # Reinforcement Learning strategies
│   ├── trading_env.py      # Custom OpenAI Gym environment for Forex
│   └── ppo_trading.py      # PPO (Proximal Policy Optimization) implementation
├── Data/                   # Storage for raw and preprocessed CSV/NPZ files
└── README.md               # Project documentation
```

## Features

- **Data Pipeline**: Seamless integration with MetaTrader 5 (MT5) for real-time data collection.
- **Technical Indicators**: Automated calculation of EMA (50/200), RSI, MACD, ATR, Ichimoku Cloud, and Candle geometry.
- **Diverse Model Architectures**:
  - **LSTMs**: Sequential models designed to capture temporal dependencies in financial time-series.
  - **Autoencoders**: Unsupervised learning for trend identification and anomaly detection.
  - **GANs**: Generative Adversarial Networks for data augmentation and robust forecasting.
  - **RL (PPO)**: Deep Reinforcement Learning using Proximal Policy Optimization for automated trade execution.

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- Pandas, NumPy, Scikit-learn
- Stable-Baselines3 (for RL Approach)
- Gym (for RL Approach)

### Data Preparation

1.  Run `Colecting_Data/mt5data.py` to fetch data from your MT5 terminal.
2.  Use `Colecting_Data/preproc_single_inout.py` or `preproc_multi_inout.py` to generate the feature-enriched datasets.

### Training Models

Each subdirectory in `DNN/` contains several training scripts. For example:
```bash
python3 DNN/multi_inout/classifier_multi.py
```

For the RL approach:
```bash
python3 RL_Approach/ppo_trading.py
```

## Best Practices Implemented

- **No Data Leakage**: All time-series splits use `shuffle=False` to respect chronological order.
- **Portability**: Path handling is managed via `pathlib` to ensure the project works across different OS environments.
- **Efficiency**: Data processing is optimized for performance using vectorized operations and efficient DataFrame handling.
- **Centralized Logic**: Technical indicators are calculated in a shared utility to ensure consistency across all models.

## License

This project is for educational and research purposes only. Trading Forex involves significant risk.
