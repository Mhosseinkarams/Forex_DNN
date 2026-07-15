# Performance & Verification Audit: MarketStateClassifier

- **Backend Learning Engine**: LIGHTGBM
- **Timestamp**: 2026-07-15T13:20:14.939444

## Summary Metrics
- **Accuracy**: 0.9922
- **Precision**: 0.9926
- **Recall**: 0.9922
- **F1-Score**: 0.9923

## Class Distributions
- **RANGE**: 110 samples
- **TRANSITION**: 19 samples

## Top Feature Importances
| Feature Name | Relative Importance Score |
| :--- | :--- |
| `ema50_slope` | 211.000000 |
| `ema_separation` | 161.000000 |
| `distance_to_ema600` | 113.000000 |
| `datetime` | 101.000000 |
| `ema800_slope` | 90.000000 |
| `distance_to_ema800` | 83.000000 |
| `ema_compression` | 75.000000 |
| `demand_width` | 73.000000 |
| `ema600_slope` | 70.000000 |
| `rolling_std` | 64.000000 |
| `realized_volatility` | 63.000000 |
| `weekday` | 59.000000 |
| `compression_score` | 56.000000 |
| `distance_to_ema50` | 37.000000 |
| `atr_ratio` | 37.000000 |

## Confusion Matrix
Target classes order: ['TREND', 'RANGE', 'TRANSITION']
```
[[  0   0   0]
 [  0 109   1]
 [  0   0  19]]
```