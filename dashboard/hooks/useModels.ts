import { useCallback } from 'react';
import {
  getModels,
  registerHfModel,
} from '@/lib/api/endpoints';
import type { Model, ApiListResponse } from '@/lib/api/types';
import { useApi } from './useApi';
import { usePolling } from './usePolling';

export function useModels(wsConnected: boolean) {
  const fetcher = useCallback(() => getModels(), []);
  const apiResult = useApi<ApiListResponse<Model>>(fetcher);
  const pollingResult = usePolling(fetcher, 10000, !wsConnected);

  const result = wsConnected ? apiResult : pollingResult;

  return {
    data: result.data,
    loading: result.loading,
    error: result.error,
    refetch: apiResult.refetch,
    registerHfModel,
  };
}
