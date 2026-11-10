# Morning dip features — November 9, 2026

**21 measurements are integrated as ordinary features; 19 are active after two ordinary quality prunes.** `dip_recovery_fraction` and `dip_volume_change_1d` remain calculated, persisted and cataloged, with normal unprune controls. The training configuration has **151 raw inputs / 158 Fit-head inputs**, including seven SVM outputs. Improvement remains unproven. See [ordinary_prune_final.json:25](ordinary_prune_final.json#L25).

These model inputs target morning entry and same-day exit. A setup “flag” is an ordinary feature: **1 = condition present, 0 = absent, NaN = unknown** when required observations are unavailable. Separate validity indicators report availability as 1/0. Flags neither order a purchase nor bypass normal learning/pruning. Formulas and examples: [dip_context_metadata.py:19](../../python/stock_picker/features/dip_context_metadata.py#L19).

| Family | Inputs | What the model can learn from |
|---|---:|---|
| Low context | 6 | Distance to the previous 252-session low, its age, new-low events and history availability |
| Recovery | 4 | Two-day loss relative to prior volatility, opening recovery, validity and unrecovered-selloff context |
| Pressure | 5 | Changes in loss size, range and volume; easing pressure; opening gap with prior volume context |
| Support and trend | 6 | Prior support breaks/reclaims, opening distance, moving-average slope and pullbacks |

Family membership is frozen in [experiment_protocol.json:77](experiment_protocol.json#L77).

**Chronological experiment.** The frozen panel contains 1,671,673 eligible stock-days across 1,784 names through November 6, 2026. **MRCY is included on all 1,008 panel dates**; exclusions are the existing 200-name holdout and SPY benchmark. Training uses 252 prior sessions. Two inner periods, July–December 2024 and January–June 2025, selected among nine predeclared Rank variants. Only all 21 improved both ranking correlation and top-20 returns in both inner periods. Selection was frozen before three outer periods: July–December 2025, January–June 2026, and July–November 6, 2026. [Protocol:31](experiment_protocol.json#L31); [independent result audit:1](result_audit/result_audit.json#L1).

**Observed Rank results.** Equal-weight top-20 mean daily open-to-close returns, before costs, over 342 outer sessions; differences are basis points/day versus baseline. Intervals resample whole calendar weeks, with 2,000 replications. [rank_results_summary.json:454](rank_results_summary.json#L454); [pruning_review.json:33](pruning_review.json#L33).

| Variant | Gross daily mean | Difference, bps | Pointwise 95% interval, bps |
|---|---:|---:|---:|
| Baseline | +0.3259% | — | — |
| All 21 | +0.2521% | -7.38 | [-16.29, +2.12] |
| Low context | +0.2408% | -8.51 | [-15.59, -1.59] |
| Recovery | +0.2613% | -6.46 | [-13.81, +1.48] |
| Pressure | +0.3059% | -2.00 | [-10.53, +6.73] |
| Support and trend | +0.3160% | -0.99 | [-9.71, +7.70] |
| Four setup flags | +0.2605% | -6.54 | [-16.27, +3.56] |
| Without four setup flags | +0.3016% | -2.43 | [-11.65, +6.37] |
| Compact | +0.3162% | -0.97 | [-10.04, +9.10] |

For top 20, all 21 underperformed in every outer period: −3.06, −11.77 and −7.52 bps/day. Every alternative has a negative pooled point estimate. Low context gives the strongest consistent concern, but these family-addition comparisons cannot identify an individual harmful column or show that removing a family from all 21 helps. Intervals are exploratory, unadjusted for multiple comparisons; one fixed seed/configuration and previously examined history limit confirmation. Compact cannot become the winner because its outer loss was smallest. Five zero-gain inputs warrant review; tree non-use does not rule out correlated usefulness or SVM influence. [Per-feature review:130](pruning_review.json#L130).

For the predeclared **top-five** portfolio, baseline averaged **+0.5374%** gross per day versus **+0.4705%** for all 21: **−6.69 bps/day**, with a pointwise weekly-bootstrap 95% interval of **[−26.15, +12.71] bps**. All 21 improved top-five returns in **one of three** outer periods. This does not reverse the pooled conclusion or change the frozen selection. [Top-five comparison:1174](rank_results_summary.json#L1174).

**Why two inputs were pruned.** This decision used the actual 443,570-row ordinary Yahoo training population, not the frozen research source:

- **Recovery fraction:** one row supplies 36.80% of median-imputed centered sum squares; the top ten supply 66.27%. DKS on August 25 divides a −$36.97 opening change by a roughly $0.0099945 preceding closing loss, producing −3699.03. Arithmetic is correct; the unbounded representation becomes unstable near zero loss.
- **Volume change:** the top ten rows supply 99.79%. February 25 Yahoo volumes of 30 for HUBB and 32 for UHAL-B contradict independent Massive volumes of approximately 367,726 and 330,327. These tiny denominators produce extreme ratios. Pruning contains amplification; it does not repair upstream bars or other volume features.

Both exclusions are reversible quality judgments, **not demonstrated return improvements**. All 21 fields had zero infinite values; missing values remain explicit. [Prune evidence:19](pruning_review.json#L19); [persisted-input audit:1](ordinary_persisted_input_representation_review.json#L1).

**Timing and verification.** Eight fields use the observed current open; thirteen use completed sessions with canonical exchange-calendar alignment. Current/future HLCV mutation checks passed, and 21,168 MRCY cells exactly matched frozen calculations. Missing sessions remain unknown. The separate cluster-gap timing repair uses prior membership and excludes the stock itself; independent error was below 2.1e−17. All **136 repository test targets passed**; the independent postprocessor also passed **five separate checks**. [Integration audit:6](ordinary_integration_independent_verification.json#L6); [cluster audit:1](cluster_fix_independent_verification.json#L1); [suite:184](ordinary_integration_independent_verification.json#L184).

**Source limitations.** Dated provider metadata confirms frozen SPCX history joins an older ETF and a different June 2026 common-stock listing. Its extreme recovery value affects research scaling; ordinary Yahoo begins at the new listing. QH’s April price-basis discontinuity remains unresolved, not a confirmed split. No invented adjustments or retrospective row deletions were applied. Current-universe survivorship, incomplete action certification and observing an open before assuming execution there also limit interpretation. These are reproducible exploratory comparisons, not certified executable profits. [Source findings:572](pruning_review.json#L572); [protocol limitations:153](experiment_protocol.json#L153).

**Pending final audits — fill only after completion.**

| Work | Current status | Required final entry |
|---|---|---|
| Frozen full Fit | Pending: baseline 132 raw → 139 head versus all 21, 153 raw → 160 head; two OOF blocks, one July–November 2026 outer block | Matched-row/schema audit; Top5/Top20, MAE, actual 0.5% gate with cash days/costs; concentration, uncertainty and convergence warnings |
| Ordinary training | Pending: 19 active new inputs, 151 raw → 158 head | Saved schemas, normal fold/holdout evidence and readiness checks |

The 21 baseline SVM output checkpoints passed schema, row-identity, checksum and finite-value checks. All three baseline SVR fits emitted convergence warnings under the unchanged 10,000-iteration limit: convergence was not established. The augmented SVM fits remain running. [Baseline checkpoint audit:1](baseline_svm_checkpoint_audit.json#L1).

The frozen full-21 Fit experiment cannot validate the separate 19-input production configuration. [Schema distinction:1](production_schema_distinction.json#L1); [Fit protocol:130](experiment_protocol.json#L130).

Next iteration: predeclare one bounded or signed-log recovery representation with loss-depth components; after repairing/marking bad volume bars, test a log-volume ratio. These are **untested hypotheses**, requiring new forward validation and normal full-model pruning comparisons, not tuned fixes selected from these outer results.

**Reproduction.** Large generated panels are Git-ignored. Restore the frozen price cache and pinned context/source copies; a new provider pull is not exact replay. Follow the recorded interpreter invocation and `prices`/`assemble` phases in [prepare_baseline.py:7](prepare_baseline.py#L7), then [prepare_dip_panel.py:1](prepare_dip_panel.py#L1) and [prepare_evaluation_outcomes.py:18](prepare_evaluation_outcomes.py#L18). Verify [panel hashes:18](phase1_data_handoff.json#L18); use [launch commands:10](research_launch_commands.json#L10) and the [final complete audit:6](result_audit_plan.json#L6).
