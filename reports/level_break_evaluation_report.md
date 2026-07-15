# Performance & Verification Audit: LevelBreakProbabilityModel

- **Backend Learning Engine**: LIGHTGBM
- **Timestamp**: 2026-07-15T12:52:35.789740

## Summary Metrics
- **Accuracy**: 1.0000
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000
- **ROC-AUC**: nan

## Class Distributions
- **REJECT**: 5 samples

## Top Feature Importances
| Feature Name | Relative Importance Score |
| :--- | :--- |
| `ema50_slope` | 0.000000 |
| `ema600_slope` | 0.000000 |
| `ema800_slope` | 0.000000 |
| `ema_separation` | 0.000000 |
| `ema_compression` | 0.000000 |
| `distance_to_ema50` | 0.000000 |
| `distance_to_ema600` | 0.000000 |
| `distance_to_ema800` | 0.000000 |
| `candle_body` | 0.000000 |
| `upper_wick` | 0.000000 |
| `lower_wick` | 0.000000 |
| `bos_count_last_n` | 0.000000 |
| `choch_count_last_n` | 0.000000 |
| `time_since_last_bos` | 0.000000 |
| `time_since_last_choch` | 0.000000 |

## Confusion Matrix
Target classes order: ['REJECT', 'BREAK']
```
[[5 0]
 [0 0]]
```