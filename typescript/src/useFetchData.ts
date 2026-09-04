import { useEffect, useState } from "react";

interface FetchState<T> {
  data: T | null;
  error: string | null;
}

interface UseFetchDataOptions {
  // Re-run the fetch when any of these change, in addition to on mount --
  // e.g. a `refreshCount` bumped after a mutation, or a derived value like a
  // ticker list that isn't known until an earlier fetch resolves.
  deps?: unknown[];
  // Also re-run on a timer while mounted (for "live-ish" data via polling
  // instead of a websocket -- see components/CorrelationHeatmap.tsx's
  // fetchPrunedFeatures for a non-polled deps-only example).
  intervalMs?: number;
}

export function useFetchData<T>(fetcher: () => Promise<T>, options?: UseFetchDataOptions): FetchState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const deps = options?.deps ?? [];
  const intervalMs = options?.intervalMs;

  useEffect(() => {
    let cancelled = false;

    function run() {
      fetcher()
        .then((result) => {
          if (!cancelled) setData(result);
        })
        .catch((err) => {
          if (!cancelled) setError(String(err));
        });
    }

    run();
    const intervalId = intervalMs ? setInterval(run, intervalMs) : undefined;

    return () => {
      cancelled = true;
      if (intervalId) clearInterval(intervalId);
    };
    // `deps` is caller-supplied by design -- see UseFetchDataOptions above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, error };
}
