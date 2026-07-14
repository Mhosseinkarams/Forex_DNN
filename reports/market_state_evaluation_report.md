# Performance & Verification Audit: MarketStateClassifier

- **Backend Learning Engine**: LIGHTGBM
- **Timestamp**: 2026-07-14T17:00:06.066173

## Summary Metrics
- **Accuracy**: 0.3400
- **Precision**: 0.3398
- **Recall**: 0.3400
- **F1-Score**: 0.3379

## Class Distributions
- **TREND**: 68 samples
- **RANGE**: 69 samples
- **TRANSITION**: 63 samples

## Top Feature Importances
| Feature Name | Relative Importance Score |
| :--- | :--- |
| `demand_width` | 150.000000 |
| `distance_to_ema600` | 147.000000 |
| `time_since_last_bos` | 135.000000 |
| `ema_compression` | 128.000000 |
| `ema50_distance_v1` | 126.000000 |
| `ema_separation` | 123.000000 |
| `session` | 119.000000 |
| `distance_to_nearest_low` | 119.000000 |
| `demand_touch_count` | 114.000000 |
| `supply_freshness` | 111.000000 |
| `ema50_slope` | 108.000000 |
| `lower_wick` | 103.000000 |
| `protected_high_distance` | 102.000000 |
| `distance_to_invalidation_level` | 100.000000 |
| `candle_body` | 97.000000 |

## Confusion Matrix
Target classes order: ['TREND', 'RANGE', 'TRANSITION']
```
[[28 22 18]
 [27 23 19]
 [28 18 17]]
```