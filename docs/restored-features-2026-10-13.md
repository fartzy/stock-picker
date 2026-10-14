# Restoring older feature prunes — 2026-10-13

All variants retained the 19 new price-pattern inputs. We trained on the same
442,616 rows from 1,799 training tickers, with 100 Rank rounds, and compared
three strictly later date blocks (Sep 17–24, Sep 25–Oct 2, Oct 5–12).
Top-20 return is mean open-to-close return per selected name, not a cumulative
portfolio return. Top-five/top-ten restoration choices were selected using
only each fold's training-model importance.

| Variant | Inputs | Mean top-20 return | Mean Rank IC | Top-20 blocks beating baseline |
| --- | ---: | ---: | ---: | ---: |
| baseline | 125 | 0.223% | 0.1149 | — |
| restore trend | 135 | 0.232% | 0.1048 | 2/3 |
| restore oscillators | 135 | 0.229% | 0.1047 | 1/3 |
| restore volume | 132 | 0.425% | 0.1123 | 3/3 |
| restore candle | 127 | 0.037% | 0.1115 | 1/3 |
| restore volatility | 134 | 0.333% | 0.1227 | 2/3 |
| restore all | 225 | 0.597% | 0.1081 | 2/3 |
| restore top5 | 130 | 0.070% | 0.1059 | 2/3 |
| restore top10 | 135 | 0.260% | 0.1009 | 2/3 |

Seven volume inputs improved top-20 returns in all three forward blocks:
`dollar_volume`, `obv`, `obv_change_20d`, `volume_ratio_10d`,
`volume_ratio_20d`, `volume_ratio_60d`, and `volume_zscore_20d`.
They were restored through the ordinary prune controls. The prior prune
settings are retained in `data/pruned_features/pruned_before_volume_restore_2026-10-13.json`.

On the already-used 200-ticker holdout, volume restoration had Rank IC 0.2837
versus 0.2837, top-20 return +0.669% versus +0.692%, and hit rate 62.6%
versus 62.8%. Top-five return rose from +0.959% to +1.235%. The decision
prioritized the consistent forward-block improvement; the holdout is mixed
and shares calendar dates with training tickers.

The full-history 132-input Rank fit, including all 19 new inputs, was activated
in the normal `day_session_return_rank.pkl` serving slot. The previous live
Rank fits are backed up under `data/models/backups/`. Regular Fit training
uses the same selection/prune rules and now refits through the latest date
after chronological validation.

Full results: `data/feature_search/restored-features-2026-10-13.json` and
`data/feature_search/restored-volume-final-2026-10-13.json`.
