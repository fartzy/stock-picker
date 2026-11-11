import assert from "node:assert/strict";
import { mapConcurrent, requestJson } from "../src/requests.js";
import { fetchPaperBook } from "../src/api.js";

const originalFetch = globalThis.fetch;
try {
  let request;
  globalThis.fetch = async (path, options) => {
    request = { path, ...options };
    return { ok: true, json: async () => ({ days: [] }) };
  };
  const completed = new AbortController();
  await fetchPaperBook("both", undefined, 5, 5, completed.signal);
  assert.equal(request.path, "/api/paper-book?kind=both&fit_top_k=5&rank_top_k=5");
  completed.abort();
  assert.equal(request.signal.aborted, false, "Completed reads release their abort listener");
  await fetchPaperBook("both", undefined, undefined, undefined);
  assert.equal(request.path, "/api/paper-book?kind=both", "All picks remain available");

  globalThis.fetch = (_path, { signal }) => new Promise((_resolve, reject) => {
    request = { signal };
    if (signal.aborted) reject(signal.reason);
    else signal.addEventListener("abort", () => reject(signal.reason), { once: true });
  });
  await assert.rejects(requestJson("/stalled", { timeoutMs: 5 }), /Loading took too long/);
  assert.equal(request.signal.aborted, true, "Timeout releases the connection");
  const leaving = new AbortController();
  const pending = requestJson("/leaving", { signal: leaving.signal });
  leaving.abort();
  await assert.rejects(pending, { name: "AbortError" });
  assert.equal(request.signal.aborted, true, "Unmount cancellation reaches fetch");
  await assert.rejects(requestJson("/already-left", { signal: leaving.signal }), { name: "AbortError" });

  globalThis.fetch = async (_path, { signal }) => ({
    ok: true,
    json: () => new Promise((_resolve, reject) => {
      signal.addEventListener("abort", () => reject(signal.reason), { once: true });
    }),
  });
  await assert.rejects(requestJson("/stalled-body", { timeoutMs: 5 }), /Loading took too long/);

  globalThis.fetch = async () => ({ ok: false, status: 503, json: async () => ({ detail: "History unavailable" }) });
  await assert.rejects(requestJson("/failed"), /History unavailable/);
} finally {
  globalThis.fetch = originalFetch;
}

const controller = new AbortController();
let active = 0;
let peak = 0;
const values = await mapConcurrent([0, 1, 2, 3, 4], 2, async (value) => {
  active += 1;
  peak = Math.max(peak, active);
  await new Promise((resolve) => setImmediate(resolve));
  active -= 1;
  return value * 2;
}, controller.signal);
assert.equal(peak, 2, "At most two history requests run at once");
assert.deepEqual(values, [0, 2, 4, 6, 8], "Input order is preserved");

const cancelled = new AbortController();
const started = [];
const release = [];
const queued = mapConcurrent([0, 1, 2, 3, 4], 2, (value) => {
  started.push(value);
  return new Promise((resolve) => release.push(resolve));
}, cancelled.signal);
const rejected = assert.rejects(queued, { name: "AbortError" });
assert.deepEqual(started, [0, 1]);
cancelled.abort();
release.forEach((resolve) => resolve());
await rejected;
assert.deepEqual(started, [0, 1], "Leaving the tab stops launching queued days");
await assert.rejects(mapConcurrent([0], 2, async () => assert.fail("Canceled work must not start"), cancelled.signal), { name: "AbortError" });
