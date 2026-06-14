import { useCallback } from 'react';
import { getLogs } from '@/lib/api/endpoints';
import type { LogEntry, ApiListResponse } from '@/lib/api/types';
import { useApi } from './useApi';
import { usePolling } from './usePolling';

export function useLogs(
  wsConnected: boolean,
  groupId?: string,
  eventType?: string,
) {
  const fetcher = useCallback(
    () => getLogs({ group_id: groupId, event_type: eventType, limit: 100 }),
    [groupId, eventType],
  );
  const apiResult = useApi<ApiListResponse<LogEntry>>(fetcher);
  const pollingResult = usePolling(fetcher, 2000, !wsConnected);

  const result = wsConnected ? apiResult : pollingResult;
  return { data: result.data, loading: result.loading, error: result.error, refetch: apiResult.refetch };
}
