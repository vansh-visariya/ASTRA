import { useCallback } from 'react';
import { getSystemMetrics } from '@/lib/api/endpoints';
import { useApi } from './useApi';
import { usePolling } from './usePolling';

export function useMetrics(wsConnected: boolean) {
  const fetcher = useCallback(() => getSystemMetrics(), []);
  const apiResult = useApi(fetcher);
  const pollingResult = usePolling(fetcher, 3000, !wsConnected);

  const result = wsConnected ? apiResult : pollingResult;
  return { ...result, refetch: apiResult.refetch };
}
