import { useEffect, useState } from "react";

interface FetchState<T> {
  data: T | null;
  error: string | null;
}

export function useFetchData<T>(fetcher: () => Promise<T>): FetchState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetcher()
      .then(setData)
      .catch((err) => setError(String(err)));
    // Runs once on mount by design -- fetcher must be a referentially-stable,
    // zero-arg reference (as all current call sites are). A fetcher that closes
    // over changing props/state won't re-run here; give this hook a `deps`
    // array instead of adding one at a parameterized call site.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { data, error };
}
