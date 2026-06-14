import { useCallback } from 'react';
import {
  getNotifications,
  getUnreadCount,
  markNotificationRead,
  markAllNotificationsRead,
} from '@/lib/api/endpoints';
import type { Notification, ApiListResponse } from '@/lib/api/types';
import { useApi } from './useApi';
import { usePolling } from './usePolling';

export function useNotifications(
  wsConnected: boolean,
  params?: { limit?: number; unread_only?: boolean },
) {
  const fetcher = useCallback(() => getNotifications(params), [params]);
  const apiResult = useApi<ApiListResponse<Notification>>(fetcher);
  const pollingResult = usePolling(fetcher, 10000, !wsConnected);

  const result = wsConnected ? apiResult : pollingResult;

  return {
    data: result.data,
    loading: result.loading,
    error: result.error,
    refetch: apiResult.refetch,
    markRead: markNotificationRead,
    markAllRead: markAllNotificationsRead,
  };
}

export function useUnreadCount(wsConnected: boolean) {
  const fetcher = useCallback(() => getUnreadCount(), []);
  const apiResult = useApi<{ count: number }>(fetcher);
  const pollingResult = usePolling(fetcher, 30000, !wsConnected);

  const result = wsConnected ? apiResult : pollingResult;
  return { data: result.data, loading: result.loading, error: result.error, refetch: apiResult.refetch };
}
