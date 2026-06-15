'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useState } from 'react';
import {
  Cpu, WifiOff, RefreshCw,
  Layers, Activity, Zap, Terminal, CheckCircle, AlertCircle,
} from 'lucide-react';
import { useWS } from '@/components/WebSocketProvider';
import { useTrainingStatus } from '@/hooks';
import { activateJoin } from '@/lib/api/endpoints';
import { StatCard } from '@/components/ui/StatCard';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorState } from '@/components/ui/ErrorState';
import { EmptyState } from '@/components/ui/EmptyState';
import { SessionCard } from '@/components/training/SessionCard';

export default function ClientTrainingPage() {
  const { isConnected } = useWS();
  const { data: status, loading, error, refetch } = useTrainingStatus(!isConnected);
  const router = useRouter();
  const [activating, setActivating] = useState<string | null>(null);

  const handleActivate = async (groupId: string) => {
    setActivating(groupId);
    try {
      await activateJoin(groupId);
      refetch();
      router.push('/client/groups');
    } catch {
      setActivating(null);
    }
  };

  if (loading) return <LoadingSpinner message="Loading training status..." />;
  if (error) return <ErrorState message={error} onRetry={refetch} />;

  const hasNoSessions = !status?.sessions?.length;
  const hasPending = (status?.pending_activations?.length || 0) > 0;
  const connectedCount = status?.connected_clients?.length || 0;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between animate-fade-in">
        <div>
          <h1 className="text-2xl font-bold text-white">Training Monitor</h1>
          <p className="text-slate-400 text-sm mt-1">Real-time status of your federated learning sessions</p>
        </div>
        <button onClick={refetch} className="btn-secondary !px-3 !py-2.5 inline-flex items-center gap-1.5">
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard label="Active Sessions" value={status?.sessions?.length || 0} icon={Cpu} accent="emerald" delay={0.05} />
        <StatCard label="Connected Clients" value={connectedCount} icon={connectedCount > 0 ? Activity : WifiOff} accent="info" delay={0.1} />
        <div className="stat-card accent-blue p-5 animate-fade-in" style={{ animationDelay: '0.15s', opacity: 0 }}>
          <div className="flex items-center justify-between mb-4">
            <span className="text-slate-400 text-xs font-medium uppercase tracking-wider">Training Active</span>
            <div className="w-9 h-9 rounded-xl flex items-center justify-center"
              style={{ background: status?.has_active_training ? 'var(--color-success-bg)' : 'rgba(100,116,139,0.1)' }}>
              {status?.has_active_training ? <Zap size={17} style={{ color: 'var(--color-success)' }} /> : <Activity size={17} className="text-slate-500" />}
            </div>
          </div>
          <p className="text-lg font-bold" style={{ color: status?.has_active_training ? 'var(--color-success)' : 'var(--color-muted)' }}>
            {status?.has_active_training ? 'In Progress' : 'Idle'}
          </p>
        </div>
      </div>

      {hasPending && (
        <div className="glass-card p-5 animate-fade-in" style={{ animationDelay: '0.1s', opacity: 0, borderColor: 'var(--color-warning-border)' }}>
          <div className="flex items-center gap-3 mb-3">
            <AlertCircle size={18} style={{ color: 'var(--color-warning)' }} />
            <h3 className="text-white font-semibold text-sm">Approved Groups — Activation Required</h3>
          </div>
          <p className="text-slate-400 text-sm mb-3">These groups have approved your join request. Activate them to start training.</p>
          <div className="space-y-2">
            {status!.pending_activations.map((pa) => (
              <div key={pa.group_id} className="flex items-center justify-between p-3 rounded-xl" style={{ background: 'rgba(30,41,59,0.4)' }}>
                <div className="flex items-center gap-3">
                  <Layers size={16} style={{ color: 'var(--color-warning)' }} />
                  <span className="text-white text-sm font-medium">{pa.group_id}</span>
                  <span className="text-slate-500 text-xs">({pa.model_id})</span>
                </div>
                <button
                  onClick={() => handleActivate(pa.group_id)}
                  disabled={activating === pa.group_id}
                  className="btn-emerald text-white text-xs px-3 py-1.5"
                >
                  {activating === pa.group_id ? 'Activating...' : 'Activate'}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {hasNoSessions ? (
        <EmptyState
          icon={Cpu}
          title="No Active Training Sessions"
          message="Join a group and activate it to start training"
          action={
            <Link href="/client/groups" className="btn-emerald inline-flex text-white text-sm px-5 py-2.5 items-center gap-2">
              <Layers size={15} /> Browse Groups
            </Link>
          }
        />
      ) : (
        <div className="space-y-4">
          {status!.sessions.map((session, idx) => (
            <SessionCard key={session.client_id || session.group_id} session={session} index={idx} />
          ))}
        </div>
      )}

      <div className="glass-card p-5 animate-fade-in" style={{ animationDelay: '0.3s', opacity: 0 }}>
        <h3 className="text-white font-semibold text-sm mb-3 flex items-center gap-2">
          <Terminal size={15} className="text-indigo-400" /> How Training Works
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          {[
            { step: '1', title: 'Join a Group', desc: 'Request to join from Available Groups page' },
            { step: '2', title: 'Activate Membership', desc: 'Once approved, activate your join request' },
            { step: '3', title: 'Run Python Client', desc: 'Start the CLI client with your credentials' },
            { step: '4', title: 'Monitor Here', desc: 'Watch real-time metrics on this page' },
          ].map((item) => (
            <div key={item.step} className="p-3 rounded-xl" style={{ background: 'rgba(15,23,42,0.4)' }}>
              <span className="w-6 h-6 rounded-lg flex items-center justify-center text-[11px] font-bold mb-2 inline-flex"
                style={{ color: 'var(--color-info)', background: 'rgba(59,130,246,0.12)' }}>
                {item.step}
              </span>
              <p className="text-white text-xs font-medium">{item.title}</p>
              <p className="text-slate-500 text-[11px] mt-0.5">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
