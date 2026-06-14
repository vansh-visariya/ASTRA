'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Layers, Eye, EyeOff, Plus, RefreshCw, ArrowUpRight } from 'lucide-react';
import { useWS } from '@/components/WebSocketProvider';
import { useGroups } from '@/hooks';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { GroupTable } from '@/components/groups/GroupTable';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorState } from '@/components/ui/ErrorState';
import { EmptyState } from '@/components/ui/EmptyState';
import { useAuth } from '@/components/AuthContext';
import { controlGroup } from '@/lib/api/endpoints';
import type { Group } from '@/lib/api/types';

export default function GroupsPage() {
  const { token, user } = useAuth();
  const { isConnected } = useWS();
  const { data, loading, error, refetch } = useGroups(!isConnected);
  const [showToken, setShowToken] = useState<Record<string, boolean>>({});
  const router = useRouter();

  const groups: Group[] = (data as any)?.groups || [];

  const handleControl = async (groupId: string, action: 'start' | 'pause' | 'resume' | 'stop') => {
    try {
      await controlGroup(groupId, action);
      refetch();
    } catch {
      // snackbar would go here
    }
  };

  if (loading && !groups.length) return <LoadingSpinner message="Loading groups..." />;
  if (error) return <ErrorState message={error} onRetry={refetch} />;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between animate-fade-in">
        <div>
          <h1 className="text-2xl font-bold text-white">Groups</h1>
          <p className="text-slate-400 text-sm mt-1">Manage federated learning groups</p>
        </div>
        <div className="flex gap-2">
          <button onClick={refetch} className="btn-secondary !px-3 !py-2.5 inline-flex items-center gap-1.5">
            <RefreshCw size={14} /> Refresh
          </button>
          {user?.role === 'admin' && (
            <Link href="/dashboard/create" className="btn-primary px-4 py-2.5 text-sm flex items-center gap-2">
              <Plus size={15} /> Create Group
            </Link>
          )}
        </div>
      </div>

      {groups.length === 0 ? (
        <EmptyState
          icon={Layers}
          title="No groups yet"
          message="Create your first federated learning group to get started."
          action={
            user?.role === 'admin' ? (
              <Link href="/dashboard/create" className="btn-primary inline-flex px-5 py-2.5 text-sm items-center gap-2">
                <Plus size={15} /> Create Group
              </Link>
            ) : undefined
          }
        />
      ) : (
        <div className="glass-card overflow-hidden animate-fade-in">
          <GroupTable groups={groups} onAction={handleControl} onDetail={(id) => router.push(`/dashboard/groups/${id}`)} />
        </div>
      )}
    </div>
  );
}
