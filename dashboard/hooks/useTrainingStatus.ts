import { useCallback } from 'react';
import { getTrainingStatus } from '@/lib/api/endpoints';
import type { TrainingStatus } from '@/lib/api/types';
import { useApi } from './useApi';
import { usePolling } from './usePolling';

export function useTrainingStatus(wsConnected: boolean) {
  const fetcher = useCallback(() => getTrainingStatus(), []);
  const apiResult = useApi<TrainingStatus>(fetcher);
  const pollingResult = usePolling(fetcher, 3000, !wsConnected);

  const result = wsConnected ? apiResult : pollingResult;
  return { data: result.data, loading: result.loading, error: result.error, refetch: apiResult.refetch };
}
