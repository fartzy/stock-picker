/** Bound history reads and release browser connections when their view closes. */
export async function requestJson<T>(path: string, options: {
  signal?: AbortSignal;
  body?: unknown;
  timeoutMs?: number;
} = {}): Promise<T> {
  const controller = new AbortController();
  const cancel = () => controller.abort(options.signal?.reason);
  if (options.signal?.aborted) cancel();
  else options.signal?.addEventListener("abort", cancel, { once: true });
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, options.timeoutMs ?? 15_000);
  try {
    const response = await fetch(path, {
      signal: controller.signal,
      ...(options.body === undefined ? {} : {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(options.body),
      }),
    });
    if (!response.ok) {
      const value: unknown = await response.json().catch(() => null);
      const detail = value && typeof value === "object" && "detail" in value ? value.detail : null;
      throw new Error(typeof detail === "string" ? detail : `Request failed (HTTP ${response.status})`);
    }
    return await response.json() as T;
  } catch (error) {
    if (timedOut) throw new Error("Loading took too long. Please retry.");
    throw error;
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener("abort", cancel);
  }
}

/** Keep history work bounded; never start the next item after cancellation. */
export async function mapConcurrent<T, R>(
  items: readonly T[], concurrency: number,
  work: (item: T) => Promise<R>, signal: AbortSignal,
): Promise<R[]> {
  if (!Number.isInteger(concurrency) || concurrency < 1) throw new Error("Invalid concurrency");
  const results: R[] = new Array(items.length);
  let cursor = 0;
  let stopped = false;
  async function worker() {
    while (!stopped && cursor < items.length) {
      signal.throwIfAborted();
      const index = cursor++;
      try { results[index] = await work(items[index]); }
      catch (error) { stopped = true; throw error; }
    }
    signal.throwIfAborted();
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, worker));
  signal.throwIfAborted();
  return results;
}
