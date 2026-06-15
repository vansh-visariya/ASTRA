import { useCallback } from 'react';
import {
  getJoinRequests,
  requestJoin,
  approveJoin,
  rejectJoin,
  getMyJoinStatus,
  activateJoin,
} from '@/lib/api/endpoints';
import type { JoinRequest, ApiListResponse } from '@/lib/api/types';
import { useApi } from './useApi';
import { usePolling } from './usePolling';

export function useJoinRequests(wsConnected: boolean, groupId?: string) {
  const fetcher = useCallback(() => getJoinRequests(groupId), [groupId]);
  const apiResult = useApi<ApiListResponse<JoinRequest>>(fetcher);
  const pollingResult = usePolling(fetcher, 2000, !wsConnected && !!groupId);

  const result = wsConnected ? apiResult : pollingResult;

  return {
    data: !groupId ? null : result.data,
    loading: !groupId ? false : result.loading,
    error: result.error,
    refetch: apiResult.refetch,
    requestJoin,
    approveJoin,
    rejectJoin,
    getMyJoinStatus,
    activateJoin,
  };
}
