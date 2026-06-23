'use client';

import { useState, useEffect } from 'react';
import {
  Users, Clock, RefreshCw,
} from 'lucide-react';
import { useWS } from '@/components/WebSocketProvider';
import { useGroups, useJoinRequests } from '@/hooks';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorState } from '@/components/ui/ErrorState';
import { EmptyState } from '@/components/ui/EmptyState';
import type { Group } from '@/lib/api/types';

export default function ClientGroupsPage() {
  const { isConnected } = useWS();
  const { data: groupsData, loading, error, refetch } = useGroups(isConnected);
  const { requestJoin, activateJoin, getMyJoinStatus } = useJoinRequests(isConnected);

  const [joinStatuses, setJoinStatuses] = useState<Record<string, string>>({});
  const [joining, setJoining] = useState<string | null>(null);
  const [statusLoading, setStatusLoading] = useState(true);

  const groups: Group[] = (groupsData as any)?.groups || [];

  // Sync join statuses from server on mount and when groups change
  useEffect(() => {
    let cancelled = false;
    const syncStatuses = async () => {
      const statuses: Record<string, string> = {};
      for (const g of groups) {
        try {
          const res = await getMyJoinStatus(g.group_id);
          statuses[g.group_id] = normalizeStatus(res.status || 'none');
        } catch {
          statuses[g.group_id] = 'none';
        }
      }
      if (!cancelled) {
        setJoinStatuses(statuses);
        setStatusLoading(false);
      }
    };
    syncStatuses();
    return () => { cancelled = true; };
  }, [groups, getMyJoinStatus]);

  const handleJoinRequest = async (groupId: string) => {
    setJoining(groupId);
    try {
      await requestJoin({ group_id: groupId });
      setJoinStatuses((prev) => ({ ...prev, [groupId]: 'pending' }));
    } catch {
      setJoinStatuses((prev) => ({ ...prev, [groupId]: 'error' }));
    } finally {
      setJoining(null);
    }
  };

  const handleActivate = async (groupId: string) => {
    setJoining(groupId);
    try {
      await activateJoin(groupId);
      setJoinStatuses((prev) => ({ ...prev, [groupId]: 'joined' }));
    } catch {
      // stay as approved
    } finally {
      setJoining(null);
    }
  };

  // Normalize server statuses to local states
  const normalizeStatus = (raw: string): string => {
    if (raw === 'activated' || raw === 'joined') return 'joined';
    return raw;
  };

  if (loading && !groups.length) return <LoadingSpinner message="Loading groups..." />;
  if (error) return <ErrorState message={error} onRetry={refetch} />;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Available Groups</h1>
          <p className="text-slate-400 text-sm mt-1">Browse and join federated learning groups</p>
        </div>
        <button onClick={refetch} className="btn-secondary inline-flex items-center gap-2 !px-3 !py-2">
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {groups.length === 0 ? (
        <EmptyState icon={Users} title="No Groups Available" message="There are no groups available to join at the moment." />
      ) : (
        <div className="grid gap-4">
          {groups.map((group) => {
            const js = joinStatuses[group.group_id] || 'none';
            const clientCount = Object.keys(group.clients || {}).length;

            return (
              <div key={group.group_id} className="glass-card p-5">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="text-lg font-semibold text-white">{group.group_id}</h3>
                      {js !== 'none' && <StatusBadge status={js} />}
                    </div>
                    <p className="text-slate-400 text-sm">
                      Model: <span style={{ color: 'var(--color-success)' }}>{group.model_id}</span>
                    </p>
                    <div className="flex items-center gap-4 mt-3 text-sm text-slate-500">
                      <span className="flex items-center gap-1"><Users size={14} /> {clientCount} clients</span>
                      <span className="flex items-center gap-1"><Clock size={14} /> W{group.window_size}/T{group.time_limit}s</span>
                    </div>
                  </div>

                  <div className="ml-4">
                    {js === 'pending' ? (
                      <button disabled className="btn-secondary !px-4 !py-2 opacity-50 cursor-not-allowed inline-flex items-center gap-2">
                        <Clock size={14} /> Pending
                      </button>
                    ) : js === 'joined' ? (
                      <button disabled className="btn-success !px-4 !py-2 opacity-50 cursor-not-allowed">
                        Joined
                      </button>
                    ) : js === 'approved' ? (
                      <button
                        onClick={() => handleActivate(group.group_id)}
                        disabled={joining === group.group_id}
                        className="btn-success inline-flex items-center gap-2"
                      >
                        {joining === group.group_id ? 'Joining...' : 'Join Group'}
                      </button>
                    ) : (
                      <button
                        onClick={() => handleJoinRequest(group.group_id)}
                        disabled={joining === group.group_id}
                        className="btn-primary inline-flex items-center gap-2"
                      >
                        {joining === group.group_id ? 'Requesting...' : 'Request to Join'}
                      </button>
                    )}
                  </div>
                </div>

                {js === 'error' && (
                  <div className="mt-3 p-3 rounded-lg text-sm" style={{ background: 'var(--color-error-bg)', color: 'var(--color-error)' }}>
                    Failed to submit join request. Please try again.
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
