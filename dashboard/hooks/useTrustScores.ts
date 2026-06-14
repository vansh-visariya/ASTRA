import { useCallback } from 'react';
import { getTrustScore, getAllTrustScores } from '@/lib/api/endpoints';
import type { TrustData, ApiListResponse } from '@/lib/api/types';
import { useApi } from './useApi';
import { usePolling } from './usePolling';

export function useTrustScores(wsConnected: boolean, userId?: number | string) {
  const fetcher = useCallback(
    () => (userId ? getTrustScore(userId) : getAllTrustScores()),
    [userId],
  );
  const apiResult = useApi<TrustData | ApiListResponse<TrustData>>(fetcher);
  const pollingResult = usePolling<TrustData | ApiListResponse<TrustData>>(fetcher, 5000, !wsConnected);

  const result = wsConnected ? apiResult : pollingResult;
  return { data: result.data, loading: result.loading, error: result.error, refetch: apiResult.refetch };
}
