# Published ordinary dip inputs: tree usage

Run `1bd4a6c807504fb48a293b22d99bbb2c`. The 19 new dip inputs are added to 132 existing raw inputs: 151 total raw inputs for Rank and each of the seven SVM source models, and 158 Fit-head inputs including the seven SVM outputs. This table describes the new dip inputs only; it is not a performance comparison.

| Dip input | Fit gain % | Fit splits | Rank gain % | Rank splits |
|---|---:|---:|---:|---:|
| `dip_close_to_low_252d` | 0.2032 | 28 | 0.6555 | 22 |
| `dip_deep_unrecovered` | 0.0000 | 0 | 0.0000 | 0 |
| `dip_gap_previous_volume_zscore` | 2.4684 | 30 | 0.2211 | 12 |
| `dip_loss_2d_sigma` | 0.1151 | 14 | 0.6419 | 22 |
| `dip_loss_change_sigma` | 0.0296 | 4 | 0.3812 | 19 |
| `dip_low_age_252d` | 0.6909 | 29 | 0.2937 | 12 |
| `dip_low_history_valid` | 0.0000 | 0 | 0.0000 | 0 |
| `dip_open_below_low_252d` | 0.0000 | 0 | 0.0000 | 0 |
| `dip_open_to_low_252d` | 0.2814 | 15 | 0.6051 | 23 |
| `dip_pressure_easing` | 0.0000 | 0 | 0.0000 | 0 |
| `dip_prior_new_low_252d` | 0.0000 | 0 | 0.0000 | 0 |
| `dip_range_change_1d` | 0.6029 | 10 | 0.2640 | 6 |
| `dip_recovery_valid` | 0.0000 | 0 | 0.0000 | 0 |
| `dip_sma50_slope_10d` | 0.7995 | 16 | 0.6346 | 21 |
| `dip_support_break_depth_20d` | 0.1387 | 12 | 0.0943 | 4 |
| `dip_support_close_distance_20d` | 0.0574 | 2 | 0.1377 | 6 |
| `dip_support_open_distance_20d` | 0.5155 | 12 | 0.2218 | 8 |
| `dip_support_reclaim_20d` | 0.0000 | 0 | 0.0000 | 0 |
| `dip_trend_pullback` | 0.0000 | 0 | 0.0000 | 0 |

Zero splits means the fitted trees did not split on that input. It does not remove the input from the seven SVM source models, demonstrate lack of predictive value, or trigger a prune. Gain percentages are normalized within each model; their raw gains use different objectives.

Exact gains, model hashes, and API references: [ordinary_dip_tree_usage.json](ordinary_dip_tree_usage.json).
