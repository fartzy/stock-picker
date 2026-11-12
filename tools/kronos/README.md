# Local Kronos daily worker

An isolated inference process for one next-session daily OHLCV forecast. It
does not train Kronos, update the app's models, or supply calibrated profit odds.
The app can later combine forecast prices with the observed opening price as
ordinary Rank/Fit inputs. A minute-bar forecast needs a separate contract.

## Setup and run

From the repository root, with `uv` installed:

```sh
python3 -m tools.kronos.setup_runtime
.runtime/kronos/venv/bin/python -m unittest discover -s tools/kronos/tests -v
.runtime/kronos/venv/bin/python -m tools.kronos.worker --device mps
```

The worker reads JSON objects, one per stdin line, and flushes one JSON response
per line. `{"describe": true}` returns the execution profile, fingerprint and
manifest. `{"requests": [...]}` predicts each request in order while retaining
loaded weights. Use `--device cpu` explicitly on hosts without MPS; changing
device changes the execution fingerprint. There is no silent device fallback.

Each request contains `ticker`, `target_session` (YYYY-MM-DD), `price_basis`
and exactly 128 `bars`. Each bar contains `session`, `open`, `high`, `low`,
`close`, and `volume`. Supply consecutive completed XNYS sessions ending
immediately before the target. The worker checks holidays, strictly prior
dates, a last-close-plus-15-minute buffer, positive finite prices,
nonnegative volume, and coherent historical OHLC. Missing history stays missing.

The initial settings are one daily bar, four sampled paths averaged by Kronos,
temperature 1 and top-p 0.9. Outputs are point forecasts. Raw predicted values
and validation errors are preserved; invalid bars are not sorted or clipped
into apparent validity. A failure does not establish any buy/sell conclusion.

## Provenance and timing

`manifest.json` pins official code, model/tokenizer revisions and every asset's
SHA-256. Setup installs a hash-locked Python 3.12.12 environment and downloads
assets under ignored `.runtime/kronos/`. The app's dependencies are unchanged.
Runtime inference is offline and checks asset hashes before loading weights.
Maintainers regenerate the dependency lock explicitly with `--compile-lock`.

`input_hash` covers the normalized actual price/volume window, ticker, target
and explicit price basis. `request_key` additionally includes the manifest and
execution fingerprint: device, Python/platform, package versions, dependency
lock, and our worker/preprocessing source. CPU and MPS results never share a
cache identity. Each series receives a stable seed independent of request order.

The `amount` input is explicitly `volume × mean(open, high, low, close)`, matching
[pinned upstream lines 531–532](https://github.com/shiyu-coder/Kronos/blob/67b630e67f6a18c9e9be918d9b4337c960db1e9a/model/kronos.py#L531).
It is a proxy, not observed dollar turnover. Existing PriceStore bars are labeled
`price_store_ohlcv_as_stored`; their adjustment convention is not independently
asserted by this worker. Corrected source values change the input hash.

The [paper's forecasting appendix](https://arxiv.org/html/2508.02739v1) reports
pretraining through June 2024; earlier targets are rejected. This paper-reported
boundary is distinct from historical checkpoint availability. Forecasts generated
after the target session's open are marked `reconstructed`; their real
`generated_at` is retained. They are never represented as archived morning calls.
`context_not_before` is the calendar close plus that conservative buffer, not
the time a provider actually published or we first observed a bar. Reconstruction
status is determined from forecast completion time, including a call crossing the open.

## Feasibility benchmark

```sh
.runtime/kronos/venv/bin/python -m tools.kronos.benchmark \
  --target-session 2026-11-12 --limit 20 --device mps \
  --output research/kronos_runtime_2026_11_12/mps.json
```

This reads local PriceStore/universe data without modifying it. Reports retain
source-file hashes, all raw forecasts, coverage/validity failures, load and warm
latency, reverse-order repeatability checks, and explicitly labeled linear
scaling estimates. No full-universe run, historical backfill, daily scheduler,
feature-pipeline activation or production-model training is performed here.

Official references: [Kronos usage](https://github.com/shiyu-coder/Kronos),
[MIT source license](https://github.com/shiyu-coder/Kronos/blob/master/LICENSE),
[model](https://huggingface.co/NeoQuasar/Kronos-small),
[tokenizer](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base),
[Hugging Face local loading](https://huggingface.co/docs/huggingface_hub/v0.33.1/package_reference/mixins).
