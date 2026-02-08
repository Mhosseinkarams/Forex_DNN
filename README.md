# Forex_DNN

A comprehensive Deep Learning and Reinforcement Learning project for Forex trading (GBP/USD), featuring multiple architectures for price prediction and automated trading strategies.

## Project Structure

The project has been refactored into a modular, object-oriented structure. Every component is now a class, and testing notebooks (`.ipynb`) are provided for each module.

```text
Forex_DNN/
├── Colecting_Data/               # Data acquisition and preprocessing
│   ├── mt5data.py                # Class: MT5DataLoader
│   ├── preproc_single_inout.py   # Class: SingleInOutPreprocessor
│   ├── preproc_multi_inout.py    # Class: MultiInOutPreprocessor
│   ├── preproc_pivot.py          # Class: PivotPreprocessor
│   ├── utils.py                  # Class: TechnicalIndicators
│   └── Colecting_Data_Test.ipynb # Visualization and test of preprocessors
├── DNN/                          # Deep Neural Network implementations
│   ├── single_inout/             # Dense networks for price prediction
│   │   ├── classifier_binary.py  # Class: BinaryDNNClassifier
│   │   ├── classifier_multi.py   # Class: MultiDNNClassifier
│   │   └── DNN_Single_InOut_Test.ipynb
│   ├── multi_inout/              # LSTM-based sequence classification
│   │   ├── classifier_binary.py  # Class: BinaryLSTMClassifier
│   │   ├── classifier_multi.py   # Class: MultiLSTMClassifier
│   │   ├── classifier_pivot.py   # Class: PivotLSTMClassifier
│   │   └── DNN_Multi_InOut_Test.ipynb
│   ├── Unsup_LSTM/               # LSTM Autoencoders for anomaly detection
│   │   ├── train.py              # Class: UnsupLSTMAutoencoder
│   │   └── Unsup_LSTM_Test.ipynb
│   └── LSTM_GAN/                 # GANs for synthetic data and forecasting
│       ├── Model.py              # Class: LSTM_GAN
│       └── LSTM_GAN_Test.ipynb
├── RL_Approach/                  # Reinforcement Learning strategies
│   ├── trading_env.py            # Class: ForexEnv (OpenAI Gym)
│   ├── ppo_trading.py            # Class: PPOTradingAgent
│   ├── generate_signals.py       # Class: SignalGenerator (LSTM-to-RL pipeline)
│   ├── metrics.py                # Class: TradingMetrics
│   ├── walk_forward.py           # Class: WalkForwardOptimizer
│   └── RL_Approach_Test.ipynb    # Full RL system testing
├── Data/                         # Storage for CSV/NPZ files
└── README.md                     # Project documentation
```

## Features

- **Object-Oriented Design**: Every component is implemented as a class, allowing easy integration and reusability.
- **Interactive Notebooks**: Test files (`.ipynb`) are included in every directory to provide immediate feedback and performance plots.
- **Realistic Evaluation**:
  - **Market Frictions**: Spreads, commissions, and slippage are incorporated into backtests.
  - **Walk-Forward Optimization**: Industry-standard rolling window validation to prevent overfitting.
  - **Financial Metrics**: Detailed tracking of Sharpe Ratio, Sortino Ratio, Drawdown, and Profit Factor.
- **Hybrid Modeling**: A pipeline to use LSTM predictions as features for Reinforcement Learning agents.

## Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn gym stable-baselines3
```

### Data Preparation

Use the `Colecting_Data_Test.ipynb` notebook to see how data is fetched from MT5 and preprocessed with technical indicators.

### Model Training & Testing

Each directory contains a `*_Test.ipynb` notebook. Open these to:
1.  Initialize the model class.
2.  Load and preprocess data.
3.  Train the model.
4.  Visualize training history and performance plots.

## Best Practices

- **Chronological Splitting**: All time-series data is split using `shuffle=False` to maintain temporal integrity.
- **Dynamic Pathing**: The project uses `pathlib` for robust cross-platform path handling.
- **Vectorized Calculations**: Data processing is optimized for speed using NumPy and Pandas vectorized operations.

## License

This project is for educational and research purposes only. Trading Forex involves significant risk.
