# Stock prediction research and reliable feature integration

Saved November 11, 2026. Research and proposed implementation steps for the stock-picker app. Morning scoring continues separately; this document does not change models or start training.

The objective is to improve morning open-to-close predictions and reassess the same candidates later using an observed entry price. The most practical next experiments are market- and sector-relative declines, volume relative to the same time on earlier days, and news-conditioned price moves. A supplementary price-path model is also worth benchmarking.

## Requirements to preserve

- Every adopted input needs historical coverage and reliable ongoing collection. A one-off research dataset is insufficient.
- Features are ordinary model inputs: evaluate and prune them individually. A weak feature family does not mean every member must be rejected, and a promising example does not justify keeping every member.
- Give dip features particular attention, while retaining other predictive features. Do not exclude MRCY merely because it motivated the hypothesis.
- Evaluate all eligible saved candidates, including stocks never traded. Report Rank #1 separately because it matters most to the user.
- Morning inputs must exist by the scoring time. Intraday observations belong in the current-entry helper; future intraday information must never enter a morning backtest.
- Use zero assumed trading expenses in prediction studies. Do not subtract the previously proposed 0.25%. Actual broker cash amounts in the trade ledger are separate from research assumptions.
- The primary horizons remain morning entry to the same day's close, and a later observed entry to that same day's close. Overnight performance is a separate target.
- Supplementary models are acceptable and should have their own place on the Models page. The Intraday Entry helper already has a separate panel.
- Preserve original ranks, score/model versions, observation times, and decision snapshots. Do not retrospectively replace the original morning ranking with a later model's ranking.
- Keep BETZ excluded from trade imports, per the user's instruction.

## What the academic evidence supports

Published predictive relationships are conditional on market, horizon, data, and evaluation design. They supply hypotheses and useful methods, not proof that the current app has an edge.

| Research | Finding and actual horizon | Implication for this app |
| --- | --- | --- |
| Gu, Kelly and Xiu, Empirical Asset Pricing via Machine Learning, 2020 | Trees and neural networks improve monthly stock-return prediction by learning nonlinear relationships and interactions. Momentum, liquidity, and volatility are prominent inputs. | Keep a strong tree baseline and improve inputs/context before assuming a more complex architecture will win. [NBER paper](https://www.nber.org/papers/w25398) |
| Lo, Mamaysky and Wang, Foundations of Technical Analysis, 2000 | Formal pattern recognition identifies differences in conditional daily-return distributions for some chart patterns. | Quantified patterns can be tested; named chart patterns alone are not established same-day buy signals. [Paper](https://www.nber.org/papers/w7613) |
| Chan, Stock Price Reaction to News and No-News, 2003 | Subsequent drift/reversal differs between stock moves with identifiable news and moves without identified news. The study is not a same-day entry test. | Include the event context of a decline. Missing news retrieval must not mean confirmed absence of news. [Paper](https://doi.org/10.1016/S0304-405X(03)00146-6) |
| Da, Liu and Schaumburg, A Closer Look at the Short-Term Return Reversal, 2014 | Returns unexplained by fundamentals/cash-flow news are more likely to reverse. Analyst revisions help separate components of returns. | Test unexplained or residual declines; our market/sector residual is a related hypothesis, not an exact replication of their cash-flow measure. [Paper](https://doi.org/10.1287/mnsc.2013.1766) |
| Gao, Han, Li and Zhou, Market Intraday Momentum, 2018 | For the S&P 500 ETF, the previous-close-to-10-AM return predicts the final half-hour return. That first interval includes the overnight move. | Market direction and clock time can matter. This does not establish individual-stock dip reversal. [Paper](https://doi.org/10.1016/j.jfineco.2018.05.009) |
| Heston, Korajczyk and Sadka, Intraday Patterns in the Cross-Section of Stock Returns | Returns show same-clock continuation across days and short-lived reversal at nearby intervals. | Preserve time-of-day structure instead of pooling all snapshots as interchangeable. [Working paper](https://doi.org/10.2139/ssrn.1107590) |
| Baltussen, Da, Lammers and Martens, Hedging Demand and Market Intraday Momentum, 2021 | Earlier returns predict the final half-hour across many futures markets; the paper links the pattern to hedging demand. | Market-level continuation may coexist with individual-stock reversal. [Author-hosted paper](https://academicweb.nd.edu/~zda/intramom.pdf) |
| Baltussen, Da and Soebhag, End-of-Day Reversal, April 2025 working paper | Individual intraday losers exhibit reversal during the final half-hour. The main predictor ends an hour before the close, leaving a half-hour gap before the outcome interval. Printed pp. 1–2 explain the timing. | Test a separate late-session interaction. Less time remaining does not by itself imply every later dip has worse recovery prospects. [Paper](https://academicweb.nd.edu/~zda/EOD.pdf) |
| Jiang, Kelly and Xiu, Re-Imagining Price Trends, 2023 | Image-based models learn predictive price patterns beyond predefined trend rules, with transfer across settings. | A learned price-path representation is a plausible challenger; the paper does not validate our specific entry-to-close setup. [Paper](https://doi.org/10.1111/jofi.13268) |
| Gould and Bonart, Queue Imbalance as a One-Tick-Ahead Price Predictor | Best bid/ask queue imbalance predicts the direction of the next midprice change in their sample. | Potential entry-timing input if we acquire suitable quote data; the horizon is one tick, not the session close. [Paper](https://arxiv.org/abs/1512.03492) |
| Zhang, Zohren and Roberts, DeepLOB, 2019 | A convolutional/recurrent model predicts short-horizon direction from limit-order-book sequences. | Requires depth data. Published classification accuracy must not be presented as an intraday trade win rate. [Paper](https://arxiv.org/abs/1808.03668) |
| Cont, Kukanov and Stoikov, The Price Impact of Order Book Events, 2014 | Order-flow imbalance explains contemporaneous short-interval price changes, with market depth affecting impact. | Useful mechanism and input design; contemporaneous explanation alone is not future-return prediction. [Paper](https://arxiv.org/abs/1011.6402) |
| Kolm, Turiel and Westray, Deep Order Flow Imbalance, 2023 | Stationary order-flow inputs can outperform raw order-book inputs; useful horizons are very short, around a few price changes. | Feature construction matters even with deep networks. A separate microstructure helper would require a new data path. [Paper](https://doi.org/10.1111/mafi.12413) |
| Kakushadze, 101 Formulaic Alphas, 2016 | A library of price/volume formulas, with reported typical holding periods spanning fractions of a day to several days. | Mine for distinct feature ideas, then check overlap with existing features and validate our horizon. [Paper](https://arxiv.org/abs/1601.00991) |
| Glasserman and Lin, Assessing Look-Ahead Bias in Stock Return Predictions Generated By GPT Sentiment Analysis, 2023 | LLM training knowledge and contextual effects can distort historical headline experiments. | Preserve availability times and model versions; favor forward evidence for Jev-derived inputs. [Paper](https://arxiv.org/abs/2309.17322) |
| Lopez-Lira and Tang, Can ChatGPT Forecast Stock Price Movements | LLM headline research distinguishes initial market reaction from subsequent drift. A large reported initial-reaction hit rate is not a tradable future-entry hit rate. | Structured news features are worth testing against future returns after our actual observation time. [Working paper](https://doi.org/10.2139/ssrn.4412788) |

## Existing models and frameworks worth considering

| Model or framework | What it offers | Role and limitation |
| --- | --- | --- |
| [Microsoft Qlib](https://github.com/microsoft/qlib) | Feature pipelines and comparable tree, neural, temporal, and relational model benchmarks. | Useful research infrastructure and factor references. Its standard CSI benchmarks differ from our U.S. intraday problem. [Benchmark definitions](https://github.com/microsoft/qlib/tree/main/examples/benchmarks) |
| [Kronos](https://github.com/shiyu-coder/Kronos) | A financial OHLCV sequence foundation model, with released checkpoints and fine-tuning code. | Highest-priority pretrained price-path candidate. Test forecasts or representations as a separate helper/challenger. Its price-series metrics and Chinese-equity portfolio experiments do not establish U.S. same-day performance. [Paper](https://arxiv.org/abs/2508.02739) |
| [StockMixer](https://github.com/SJTU-DMTai/StockMixer) | A relatively simple MLP architecture mixing indicators, time, and stocks. | A lighter neural challenger to trees, using our data and target. |
| [MASTER](https://github.com/SJTU-DMTai/MASTER) | A market-guided transformer modeling relationships across stocks and time. | Relevant to market context; its original experiments and data processing require careful reproduction. The authors document corrections to released validation data. |
| [Amazon Chronos](https://github.com/amazon-science/chronos-forecasting) and [Google TimesFM](https://github.com/google-research/timesfm) | General time-series foundation models. | Secondary forecasting baselines. Broad forecasting benchmark performance is not proof of stock-return predictability. Check the exact checkpoint's license and training cutoff before use. |
| [FinRL](https://github.com/AI4Finance-Foundation/FinRL) | Reinforcement-learning environments and trading research workflows. | More relevant to action/allocation experiments than the immediate feature-quality problem. |
| [Microsoft RD-Agent](https://github.com/microsoft/RD-Agent/) | Automated factor/model research. | Could support systematic feature iteration later, with a trial registry and protected evaluation periods. Automated search increases the need to control repeated testing. [Paper](https://arxiv.org/abs/2505.15155) |

Do not interpret Kronos's reported relative RankIC improvements as prediction accuracy. Verify checkpoint training cutoffs before historical evaluation. If a model prediction becomes another model's feature, generate training inputs chronologically out of sample rather than using predictions from a model fitted on those same outcomes.

## Proposed features and dependable sources

These are experimental candidates. The proposed priority reflects data feasibility and relevance to our target, not an established profitability ranking.

| Priority | Candidate | Historical and live inputs | Destination |
| --- | --- | --- | --- |
| 1 | Market- and sector-relative drop | Stock, SPY, and sector ETF minute bars; relationships fitted on strictly earlier sessions | Intraday Entry helper; prior-session versions can be ordinary morning inputs |
| 1 | Volume unusual for this clock time | Cumulative or interval volume compared with the same elapsed time over earlier sessions, initially a fixed prior-session window | Intraday Entry helper |
| 2 | Price path conditioned on selling activity | Existing recovery, low-age, volatility, and volume features plus new context; OHLCV-based pressure measures are proxies, not true order-flow imbalance | Intraday Entry helper |
| 2 | Event-conditioned decline | News category, publication time, first observed time, freshness, and feed coverage; e.g. earnings, guidance, dilution, regulatory events, corporate actions | Morning models for timely information; helper for later information |
| 2 | Late-session reversal interaction | Earlier stock-relative move, market/sector movement, and time remaining | Helper or a separately evaluated late-session model |
| 3 | Learned price-path representation | Completed one- or five-minute OHLCV sequences, with documented corporate-action basis and session boundaries | Kronos/StockMixer challenger or supplementary feature |
| Later | Quote/order-flow imbalance | Historical and live quote sizes or depth, with compatible coverage and timestamps | Separate short-horizon entry-timing model |

The existing [Massive minute-bar client](../../python/stock_picker/ingestion/massive_overnight.py#L381) is the starting point for stock and benchmark prices. [Official custom-bar documentation](https://massive.com/docs/rest/stocks/aggregates/custom-bars) explains timestamps, split adjustment, pagination, empty intervals, historical coverage, and real-time/delayed plans. The account's actual entitlement and observed latency must be checked; endpoint availability alone does not prove real-time coverage or the required backfill depth.

## Existing implementation and concrete gaps

The [shared feature builder](../../python/stock_picker/features/intraday_entry.py#L26) already includes morning rank/standardized score, time elapsed/remaining, returns from the open and prior close, recent returns, distance from the observed low, drawdown from the observed high, range position, time since the low, VWAP distance, volume acceleration, recent volatility, and prior-session context. These are not new proposed features. Historical and live paths share the same completed-bar calculation.

The [accepted helper design](../../docs/adr/0028-intraday-entry-helper.md#L1) defines probability of profit, mean remaining return, and probability of losing more than 2% by the close. All use the current entry as the reference. Probability of profit and mean return need not have matching signs: less-frequent gains can be larger than more-frequent losses. The downside probability describes the closing outcome, not whether the stock touches a loss level before then.

The initial helper remains experimental; the first evaluation did not establish improvement over the time-based baseline. Preserve its evaluation report rather than describing feature integration as demonstrated improvement. The historical dataset also has selectively cached lower ranks, so it is not complete coverage of every top-20 candidate. See [dataset coverage notes](../../python/stock_picker/training/intraday_entry_dataset.py#L277).

Two specific collection gaps precede broader use:

1. The [live service](../../python/stock_picker/training/intraday_entry_service.py#L170) currently uses short-lived in-memory caches and deliberately avoids accumulating an archive. Add a persistent collector that runs independently of the Refresh button, and a post-close reconciliation/backfill pass.
2. The [training-news importer](../../python/stock_picker/training/news_ingest.py#L112) skips tickers already marked ingested for that date. The [article schema](../../python/stock_picker/storage/news_store.py#L20) stores publication time and a separate batch ingestion time, but not each article's first-observed time. Intraday news features require incremental refresh, per-article availability history, and classifier-version provenance.

## Collection and training contract

The intended path is provider data → durable raw observations → validated historical/live feature calculation → versioned feature rows → training/evaluation → saved model → live inference → recorded outcome.

- Freeze the morning candidate list, ranks, scoring time, model version, and price basis. Preserve actual scoring provenance; distinguish historical availability assumptions from recorded live timestamps.
- Backfill stock and benchmark minute bars for the same sessions, then collect all selected candidates automatically during each trading day. A proposed initial cadence is once per completed minute for the 10–20 candidates plus shared benchmarks, subject to verified entitlement and rate limits.
- Save raw responses or normalized bars with provider, ticker, session, event time, retrieval time, adjustment basis, revision/version, and coverage status. Store all pages and use idempotent keys.
- Add retry/backoff, restart checkpoints, failure visibility, and a post-close completeness check. A closed browser or missed manual refresh must not interrupt collection.
- Use only completed bars known by the decision time. Do not use the current minute's eventual high/low/volume, final daily volume, or a later low as an input.
- Preserve time zones, daylight saving, early closes, splits, dividends, and session boundaries consistently. Keep actual execution prices distinct from historical entry proxies.
- Record gaps explicitly. No eligible trades can produce an empty minute interval; a missing bar is not automatically a feed failure. Missing news retrieval is not evidence of no news.
- Persist article publication time, first-observed time, provider identity, deduplication key, coverage/failure status, and classifier/model version. Do not backdate a later classification or retrieval into an earlier decision.
- Register each feature's formula, input sources, availability rule, lookback, version, and missing-data behavior. Apply identical transformations in historical and live use.
- Select/prune features using fitting data only. Preserve an evaluation period and forward collection that are not repeatedly used to choose the next winning variant.
- Split by whole chronological dates, weight correlated snapshots appropriately, and report unique dates and stock-days. Repeated checks of one stock-day do not create independent samples.
- Evaluate probability calibration, Brier/log loss, mean-return error, and realized entry-to-close outcomes. Compare against the existing helper and simple same-clock baselines, with Rank #1 shown separately.
- Evaluate zero-expense returns, including non-traded candidates. For a fixed close, a lower entry mechanically increases return; observing a lower price can also change the conditional distribution of the close. Both facts must remain clear.
- Evaluate each useful feature independently and in sensible combinations. Persist which features survived pruning and why, plus each model's fit/calibration/evaluation dates.

## Recommended sequence when work resumes

1. Verify API freshness and backfill coverage, then implement durable candidate/benchmark minute collection and post-close reconciliation.
2. Add market/sector-relative movement and same-clock abnormal volume to the existing shared feature pipeline. Measure each addition and their interaction against the current helper.
3. Add incremental timestamped news collection, then test event-conditioned features using trustworthy availability times.
4. Benchmark one price-path challenger, starting with Kronos or StockMixer. Reuse the same candidates, entry/exit definitions, and evaluation dates.
5. Promote only supported improvements. Keep individually useful features even if the whole candidate family or supplementary model fails to improve results.

The morning scoring run is separate from this research. This save does not authorize interrupting it, changing its active model, or launching competing training jobs.

## Related local research and preserved sources

- [November 11 context experiment](../intraday_context_2026_11_11/README.md#L1): completed market-relative and same-clock-volume comparison, individual training-period feature screening, reusable historical backfill, and remaining forward-collection work. No production promotion was justified by the probability results.

- [Rank dip timing study](../rank_dip_timing_2026_11_09/README.md#L1): expanded timed-entry evidence, from-open versus prior-close anchors, and the limitations of the original small positive table.
- [Morning dip feature study](../morning_dip_model_2026_11_09/README.md#L1): earlier feature experiments and provenance.
- [Intraday Entry architecture](../../docs/adr/0028-intraday-entry-helper.md#L1): current model behavior and target definitions.
- [Source manifest](sources/manifest.json): local copies of source snapshots and selected PDFs, including original URLs where recorded and SHA-256 hashes. These are research references, not data feeds.
- [End-of-Day Reversal PDF](sources/eod_reversal.pdf), [short-term reversal PDF](sources/reversal.pdf), [market intraday momentum PDF](sources/intramom.pdf), and [machine-learning asset-pricing PDF](sources/gkx_full.pdf).

Trade confirmations are maintained separately in the app and [November 11 broker provenance](../../data/trades/broker_orders_2026-11-11.json#L1).
