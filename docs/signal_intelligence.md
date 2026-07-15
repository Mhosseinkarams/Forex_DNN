# Signal Intelligence Layer - Architecture & Operator Guide

The **Signal Intelligence Layer** is a production-grade architectural milestone designed to standardize how strategies analyze candle behaviors and record unified candidate signals for subsequent ML, RL, and statistical performance training.

---

## 1. Architectural Blueprint

The Signal Intelligence Layer decouples analytical/predictive logic from trade execution, creating a clean unidirectional pipeline:

```text
  [Market Candle Data]
           |
           v
  +--------------------------------------------+
  |        SIGNAL INTELLIGENCE LAYER           |
  |                                            |
  |  - StrongCandleEngine (Momentum/Regime)    |
  |  - RefusalCandleEngine (Shape/Rejections)  |
  +--------------------------------------------+
           |
           +--------------------------+
           | (StrongCandle)           | (RefusalSignal)
           v                          v
  +--------------------------------------------+
  |              TRADING STRATEGY              |
  |         (MM, SM, UniT, pullback, etc.)     |
  +--------------------------------------------+
           |
           v (SignalCandidate Interface)
  +--------------------------------------------+
  |          UNIFIED SIGNAL RECORDER           |
  |                                            |
  |   Writes comprehensive training samples to  |
  |   flat, retrospectively retrainable CSVs   |
  +--------------------------------------------+
```

---

## 2. Strong Candle Detection Engine

### Purpose
Analyzes every price candle to classify its strength, momentum, and regime. It does **not** generate signals.

- **Module**: `Market_Data_Pipeline/strong_candle_engine.py`
- **Output**: `StrongCandle` dataclass
- **Classifications**: `VERY_STRONG`, `STRONG`, `MEDIUM`, `WEAK`, `INDECISION`, `DOJI`, `EXPANSION`, `CLIMAX`, `EXHAUSTION`

### Configuration
```python
strong_candle_engine = StrongCandleEngine(
    lookback_period=20,
    doji_threshold=0.10,
    exhaustion_wick_ratio=0.60,
    very_strong_body_ratio=0.75,
    strong_body_ratio=0.60,
    climax_range_mult=2.0
)
```

---

## 3. Refusal Candle Engine

### Purpose
Combines geometrical shape (candle anatomy) with structural context to detect zone rejections.

- **Module**: `Market_Data_Pipeline/refusal_candle_engine.py`
- **Output**: `RefusalSignal` dataclass
- **Classifications**: `PERFECT`, `HIGH`, `MEDIUM`, `LOW`, `INVALID`

### Contextual Inputs Evaluated:
- Supply and Demand zones (overlap and depth)
- Protected Highs / Lows and Swings
- CHOCH / BOS and trend context
- Volume spikes and EMA dynamic proximity

---

## 4. Common Scoring Convention

To make combining signals effortless for downstream execution systems, **all** signal intelligence engines output two aligned fields:
1. **`confidence`**: A normalized float in the range `[0.0, 1.0]`.
2. **`quality_score`**: An integer score in the range `[0, 100]`.

---

## 5. Unified Signal Object (`SignalCandidate`)

Every strategy compiles its setups into a standardized `SignalCandidate` object located in `Core/signal_candidate.py`. This is the **sole** interface between strategies and downstream executors or management modules.

---

## 6. Unified Signal Recorder (`SignalRecorder`)

Located in `Collecting_Data/signal_recorder.py`, the `SignalRecorder` appends **every single** generated candidate (including accepted, rejected, cancelled, and executed) to rolling retrospective CSV/Parquet files (`Logs/signal_records.csv`).

This guarantees zero data leakage and preserves exact point-in-time features for offline training.

---

## 7. Retraining and Future Extension Points

Recorded datasets contain reserved outcome and label columns:
- `label_market_state_retrain`
- `label_level_break_retrain`
- `label_trade_quality_retrain`

These can be automatically loaded into the `Trainer` / `Evaluator` pipelines to continuously improve and calibrate the LightGBM classifiers.
