# Kronos daily runtime feasibility

November 12, 2026. Phase 1 only: no production model, main feature pipeline,
daily schedule, or historical training dataset changed.

Kronos-small and its tokenizer run locally in a separate, hash-pinned Python
environment. These tests forecast one daily bar from 128 completed XNYS sessions,
using four sampled paths averaged by the upstream predictor. They do not measure
prediction accuracy or produce calibrated profit probabilities.

| Final fixed configuration | CPU | Apple MPS |
| --- | ---: | ---: |
| Eligible stocks | 20 | 20 |
| Coherent forecast bars | 19 | 19 |
| Worker/model load | 1.31 s | 1.01 s |
| First forecast | 0.047 s | 0.194 s |
| Mean subsequent forecast | 0.0381 s | 0.0231 s |
| Estimated 2,000 forecasts | 76 s | 46 s |
| Estimated 500,000 forecasts | 5.29 h | 3.21 h |

Scaling estimates are linear **inference-only extrapolations**, excluding history
preparation, storage, retries and scheduler overhead. No 2,000-stock or 500,000-row
run was started. Device load is measured in an already initialized benchmark
process; operating-system and disk caches affect startup.

HURN, TTMI and MRCY were included. CSQR, AADX and ACA lacked the required complete
history and were skipped by the benchmark, not excluded from the trading app.
AAOI's forecast violated OHLC consistency and remains an explicit invalid result;
the original values were preserved without repairs. Three forecasts repeated in
reverse request order produced exactly identical keys and values on each device.

The earlier exploratory measurement used a less-specific Python manifest version.
The final manifest pins Python 3.12.12; because the manifest participates in the
seed and request identity, final forecasts differ from that exploratory run.
No validity thresholds or sampling settings were tuned to these results.

All targets are November 12, and these forecasts were generated **after the open**.
They are marked reconstructed and cannot serve as evidence of forecasts available
that morning. Source Parquet hashes, price basis, actual generation time, model
revisions, parameters, execution fingerprints and raw outputs are retained in
[cpu.json](cpu.json) and [mps.json](mps.json).

Checks: 16 contract/calendar/validation/coverage tests passed. A real offline JSON-lines
worker also returned a clean describe response, recovered from a malformed message,
isolated an oversized JSON integer, and produced identical repeated forecasts
while retaining its loaded models. A mocked call crossing the market open verifies
that reconstruction status uses completion time. The calendar buffer is recorded as
`context_not_before`, never as an actual provider observation timestamp.
Each execution fingerprint matches the current worker, contract and lock files.

Next integration step: persist forecasts by target and execution contract, expose
four open-relative features through identical historical/live calculations, and
add resumable historical and daily collection. Coverage and missing forecasts must
remain explicit. See [worker setup and contract](../../tools/kronos/README.md).
