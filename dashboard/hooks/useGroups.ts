import { useCallback } from 'react';
import {
  getGroups,
  getGroup,
} from '@/lib/api/endpoints';
import type { Group, ApiListResponse } from '@/lib/api/types';
import { useApi } from './useApi';
import { usePolling } from './usePolling';

export function useGroups(wsConnected: boolean) {
  const fetcher = useCallback(() => getGroups(), []);
  const apiResult = useApi<ApiListResponse<Group>>(fetcher);
  const pollingResult = usePolling(fetcher, 2000, !wsConnected);

  const result = wsConnected ? apiResult : pollingResult;

  return {
    data: result.data,
    loading: result.loading,
    error: result.error,
    refetch: apiResult.refetch,
  };
}

export function useGetGroup(groupId: string, wsConnected: boolean) {
  const fetcher = useCallback(() => getGroup(groupId), [groupId]);
  const apiResult = useApi(fetcher);
  const pollingResult = usePolling(fetcher, 2000, !wsConnected);

  const result = wsConnected ? apiResult : pollingResult;
  return { data: result.data, loading: result.loading, error: result.error, refetch: apiResult.refetch };
}
