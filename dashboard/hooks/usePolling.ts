import { useState, useEffect, useRef, useCallback } from 'react';

interface UsePollingResult<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs: number,
  enabled = true,
): UsePollingResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const fetchData = useCallback(async () => {
    try {
      const result = await fetcherRef.current();
      if (mountedRef.current) {
        setData(result);
        setError(null);
        setLoading(false);
      }
    } catch (err) {
      if (mountedRef.current) {
        setError(err instanceof Error ? err.message : 'Request failed');
        setLoading(false);
      }
    }
  }, []);

  const refetch = useCallback(() => {
    setLoading(true);
    fetchData();
  }, [fetchData]);

  useEffect(() => {
    mountedRef.current = true;

    if (enabled) {
      fetchData();
      intervalRef.current = setInterval(fetchData, intervalMs);
    } else {
      setLoading(false);
    }

    return () => {
      mountedRef.current = false;
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [fetchData, intervalMs, enabled]);

  return { data, loading, error, refetch };
}
