# Performance & Verification Audit: LevelBreakProbabilityModel

- **Backend Learning Engine**: LIGHTGBM
- **Timestamp**: 2026-07-14T17:00:34.489745

## Summary Metrics
- **Accuracy**: 0.5200
- **Precision**: 0.5604
- **Recall**: 0.4766
- **F1-Score**: 0.5152
- **ROC-AUC**: 0.5373

## Class Distributions
- **REJECT**: 93 samples
- **BREAK**: 107 samples

## Top Feature Importances
| Feature Name | Relative Importance Score |
| :--- | :--- |
| `volume` | 99.000000 |
| `protected_high_distance` | 77.000000 |
| `supply_touch_count` | 67.000000 |
| `ema_separation` | 62.000000 |
| `realized_volatility` | 60.000000 |
| `choch_count_last_n` | 59.000000 |
| `ema_compression` | 56.000000 |
| `candle_range` | 54.000000 |
| `distance_to_invalidation_level` | 54.000000 |
| `distance_to_ema50` | 53.000000 |
| `lower_wick` | 52.000000 |
| `trend_score` | 51.000000 |
| `compression_score` | 50.000000 |
| `time_since_last_choch` | 49.000000 |
| `bos_direction` | 49.000000 |

## Confusion Matrix
Target classes order: ['REJECT', 'BREAK']
```
[[53 40]
 [56 51]]
```