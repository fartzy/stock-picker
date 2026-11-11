# Active price pattern features

Verified against the saved live model inputs on November 11, 2026. Both morning **Rank** and **Fit** use the same 57 opening-pattern features and 19 dip-context inputs. Rank has 151 total inputs; Fit has 158, including seven supplementary SVM outputs. These counts describe saved model inputs, not a guarantee that every input has nonzero tree importance or a valid value for every stock.

Two examples are already represented:

- **Down → down → down → opens down:** `seq3_open3_seasonality`.
- **Up → up → down → down → opens down:** `seq4_open3_seasonality`.

An up/down day in these sequences means **open-to-close movement**. An opening gap compares today's open with yesterday's close. Separate dip inputs measure **close-to-close declines**. These definitions can classify the same stock differently, so they should not be used interchangeably.

For the 57 opening-pattern inputs, the feature value is the stock's average historical open-to-close return on strictly earlier occasions matching the current pattern. The model learns how to combine that value with its other inputs. A pattern match does not itself mean “buy.” Missing historical matches remain unknown.

Sources: [saved live input inventory](archive/active-pattern-features-2026-11-11/model-features.json), [saved-model input reader](../python/stock_picker/training/model_features.py#L30), [pattern mapping](../python/stock_picker/features/open_pattern_seasonality.py#L634), and [strictly earlier pattern averages](../python/stock_picker/features/open_pattern_seasonality.py#L134).

## Opening price variants

The exact names below retain the code's naming. Every name in the next table ends with `_seasonality`; that suffix is omitted in the table for readability.

- **open3:** down below −0.5%, flat from −0.5% through +0.5%, up above +0.5%.
- **open5:** also distinguishes openings below −1.5% or above +1.5%.
- **loc3 / openloc:** low, middle or high relative to yesterday's range. Openings outside that range join the low/high categories.
- **Five-location version:** below yesterday's low, lower third, middle third, upper third, or above yesterday's high.
- **vol:** volatility, not trading volume.

The opening-pattern inputs are recomputed from completed history and the opening price supplied to morning inference. They are not shifted back to yesterday's opening setup. See [bucket thresholds](../python/stock_picker/features/open_pattern_seasonality.py#L177), [range location](../python/stock_picker/features/open_pattern_seasonality.py#L201), and [live recomputation](../python/stock_picker/training/inference.py#L69).

## All 57 active opening pattern inputs

| Pattern | What it recognizes | Exact active names, excluding `_seasonality` |
| --- | --- | --- |
| Exact two-day sequence | UU, UD, DU or DD, followed by today's opening setup. | `seq2_open3`, `story_seq2_open5`, `story_seq2_loc3` |
| Exact three-day sequence | Every three-session U/D ordering, including DDD, UUD and DUD. The fourth version adds the preceding week's performance. | `seq3_open3`, `seq3_open5`, `seq3_openloc`, `story_seq3_week_open3` |
| Exact four-day sequence | Every four-session ordering, including UUDD and DUDU. | `seq4_open3`, `story_seq4_loc3` |
| Consecutive streak | One, two or three-plus consecutive up/down sessions. Streak length is capped at three. | `streak_open3`, `streak_open5`, `story_streak_loc3` |
| Direction changes | Number of flips over three or five sessions, plus yesterday's direction. Five-session flip counts are capped at two. | `flips_3d_open3`, `flips_5d_open3` |
| Latest-day reversal | Yesterday moved opposite the two preceding sessions: UUD or DDU. | `late_reversal_open3`, `story_reversal_open5`, `story_reversal_loc3` |
| Continuation | Three sessions all moving the same way versus a mixed sequence, retaining the latest direction. | `continuation_open3` |
| Yesterday's swing | Up/down and whether its absolute intraday move exceeded the stock's recent daily volatility. | `yday_swing_open5`, `story_yday_swing_loc3` |
| Last two swing sizes | Whether each of the last two sessions had a large/small move, plus the latest direction. | `two_day_swing_open3` |
| Consecutive large down sessions | Zero, one or two-plus unusually large intraday down sessions. | `big_down_run_open3` |
| Net three-session movement | Sum of the last three intraday returns, with either recent wildness/opening gap or today's opening location. | `net3_swing_open3`, `story_net3_loc3` |
| Wildest recent session | Largest absolute intraday move among the last three sessions, relative to recent daily volatility. | `wildest_3d_open3`, `story_wildest_loc3` |
| Changing volatility | Whether 20-session volatility expanded/contracted over the last three sessions and most recent session. | `vol_path_open3`, `vol_path_open5` |
| Yesterday's range | Narrow/normal/wide high-low range relative to ATR; the second version includes the preceding week's performance. | `yday_range_open3`, `story_yday_range_week_open3` |
| Volatility regime | Recent volatility low/normal/high relative to its trailing median. | `vol_regime_open3`, `story_vol_regime_loc3` |
| Net direction × volatility path | Sign of the summed three-session intraday move combined with volatility changes. Despite the name, this is not the exact three-letter sequence. | `seq3_volpath_open3` |
| Large swing × volatility path | Whether yesterday's move was large, combined with volatility changes. | `swing_volpath_open3` |
| Streak direction × volatility | Up/down streak direction combined with expanding/contracting volatility over three sessions. | `streak_vol_open3` |
| Opening location | Today's open below, within or above yesterday's range, using five locations. | `open_in_yday_range` |
| Opening-gap sequence | The two preceding opening gaps plus today's gap, each down/flat/up. | `gap_seq3_open3` |
| Gap trap | Yesterday's up/down opening gap, yesterday's intraday direction, and today's gap. | `gap_trap_open` |
| Streak then crash | Streak ending before yesterday, then whether yesterday fell more than two ATRs during the session. | `streak_crash_open` |
| Repeated declines and extreme move | Four down sessions, whether the last session's absolute move exceeded 2.5 ATRs, and today's five-bucket gap. | `multi_crash_bounce_open` |
| Preceding week's run | Five-session close-to-close gain above 5%, loss below −5%, or neither. | `week_run_open3`, `story_week_run_open5`, `story_week_run_loc3` |
| Chase then fade | One gap-up followed by two gap-down mornings, while all three sessions finished above their own opens. | `chase_then_fade_open3` |
| Three up sessions | UUU versus other three-session sequences. | `three_up_open3`, `story_three_up_open5`, `story_three_up_loc3` |
| Dump then quiet | More than a 20% decline over roughly a month, followed by an intraday move smaller than 1% in magnitude. | `dump_then_quiet_open3`, `story_dump_quiet_loc3` |
| Monthly crash then bounce | More than a 30% monthly decline, whether yesterday gained more than 1% intraday, and today's opening location. | `story_month_crash_loc3` |
| Outside day | Yesterday's high exceeded the preceding high and its low fell below the preceding low. | `story_outside_day_open3` |
| Weak close | Yesterday closed in the bottom quarter of its range. | `story_weak_close_open3` |
| Strong close | Yesterday closed in the top quarter of its range. | `story_strong_close_open3` |
| Two prior gap-downs | The two previous openings both gapped down by more than 0.5%. | `story_two_gap_down_open3` |
| Open versus five-session high | Today's open at/above the highest high of the five completed sessions, or below it. | `story_open_vs_5d_high` |
| Two-day run | UU, DD or mixed, combined with today's gap. | `story_two_day_run_open3` |
| Three same-direction gaps | Three consecutive up/down gaps including today, or mixed. | `story_three_same_gap_open3` |

All formula branches are in [the pattern construction function](../python/stock_picker/features/open_pattern_seasonality.py#L325). The [saved inventory](archive/active-pattern-features-2026-11-11/model-features.json) determines which implementations are active; other patterns present in source code are not automatically active.

## All 19 active dip inputs

These inputs use completed sessions and today's open where stated. The long-term-low window is 252 trading sessions. Support is the lowest low across 20 sessions **before yesterday**, so yesterday's attempted breakdown cannot move its own reference level.

| Exact feature | Meaning |
| --- | --- |
| `dip_close_to_low_252d` | Yesterday's close relative to the prior 252-session low. |
| `dip_open_to_low_252d` | Today's open relative to that low. |
| `dip_low_age_252d` | How many sessions ago that low most recently occurred. |
| `dip_prior_new_low_252d` | Whether yesterday made or tied that long-term low. |
| `dip_open_below_low_252d` | Whether today's open breaks below that low. |
| `dip_low_history_valid` | Whether the full required low-history window is available. A coverage flag. |
| `dip_loss_2d_sigma` | Two-day close-to-close return scaled by volatility measured before those two sessions. |
| `dip_recovery_valid` | Whether the prior two-day closing loss and today's open define a recovery fraction. A coverage flag. |
| `dip_deep_unrecovered` | Both previous close-to-close returns at most −2%; yesterday's close 0–2% above the long-term low; today's open recovers 0–25% of the two-day loss. A further opening decline does not satisfy that recovery interval. |
| `dip_loss_change_sigma` | Change in the absolute size of the daily move, scaled by pre-selloff volatility. It measures move size whether or not both moves were losses. |
| `dip_range_change_1d` | Yesterday's true range relative to the preceding session's true range. |
| `dip_pressure_easing` | Two down closing days, with a smaller second loss, lower second-day volume and an upper-half second-day close. |
| `dip_gap_previous_volume_zscore` | Today's opening gap multiplied by yesterday's volume z-score. |
| `dip_support_break_depth_20d` | Depth of yesterday's low below the pre-existing support level. |
| `dip_support_close_distance_20d` | Yesterday's closing distance from that support. |
| `dip_support_open_distance_20d` | Today's opening distance from that support. |
| `dip_support_reclaim_20d` | Yesterday broke below support but closed at/above it, and today's open holds at/above it. |
| `dip_sma50_slope_10d` | Ten-session change in the completed 50-session moving average. |
| `dip_trend_pullback` | Two-day close-to-close loss of at least 2%, while the prior close stays above a rising 50-session average. |

Flags and continuous measurements are ordinary inputs to the model. Neither their names nor a true flag predetermine the prediction's sign. The standalone `dip_recovery_fraction` and `dip_volume_change_1d` columns are not active in the saved models; retained combined patterns still incorporate related information. See [exact formulas and examples](../python/stock_picker/features/dip_context_metadata.py#L19).

## Other active pattern context

- `pattern_sequence_seasonality_3d`: rolling mean over the last ten occurrences of the three-session sequence in the completed-day snapshot. This older statistic is lagged at morning scoring; it is not one of the 57 recomputed opening features.
- `pooled_setup_seasonality`: historical gap/prior-return setup averages pooled across stocks. This is also a lagged snapshot input. [Training alignment](../python/stock_picker/training/dataset.py#L37).
- `path_cluster_distance`, `path_cluster_margin`, `path_cluster_crowding`, `path_cluster_peer_return_1d`: similarity, separation, prevalence and peer behavior for learned five-session price/volume paths. [Path construction](../python/stock_picker/features/trajectory.py#L26).
- Supporting candle inputs: `overnight_gap`, `day_range_pct`, `body_pct`, `upper_wick_pct`, `lower_wick_pct`, `close_location`. [Candle definitions](../python/stock_picker/features/candle.py#L54).
- Supporting trend inputs include recent returns, momentum spreads, RSI, MACD and distance from the 100-session moving average. The complete inventory records every active input, including non-pattern inputs.

## Separate Intraday Entry helper

The helper reassesses an observed entry against the same day's close. Its active pattern context includes:

| Pattern context | Active inputs |
| --- | --- |
| Drop since open or previous close | `return_from_open`, `return_from_previous_close`, `opening_gap`, `return_from_open_in_prior_vol` |
| Recent direction | `return_1m`, `return_5m`, `return_15m`, `return_30m`, `entry_vs_last_close` |
| Dip and recovery shape | `distance_from_low_so_far`, `drawdown_from_high_so_far`, `range_so_far`, `recovery_fraction_so_far`, `minutes_since_low` |
| Trading activity and volatility | `entry_vs_observed_vwap`, `log_observed_volume`, `volume_acceleration_5m`, `realized_volatility_15m` |
| Clock and remaining session | `minutes_since_open`, `minutes_to_close`, `elapsed_session_fraction` |
| Prior context | `prior_return_1d`, `prior_return_2d`, `prior_volatility_32d`, `prior_rsi_28`, `prior_price_vs_sma_100d`, `prior_volume_ratio_20d` |

The remaining inputs are `morning_rank`, `morning_score_z` and `observed_bar_fraction`, making 30 saved inputs. Saved inputs with zero tree importance can still appear in the model's schema. The helper's [saved metadata](../data/models/intraday_entry_metadata.json#L11) and [shared feature formulas](../python/stock_picker/features/intraday_entry.py#L26) record the exact definitions.

The six market-relative and same-clock dollar-volume features tested in the [November 11 research iteration](../research/intraday_context_2026_11_11/README.md#L7) are not active in production.
