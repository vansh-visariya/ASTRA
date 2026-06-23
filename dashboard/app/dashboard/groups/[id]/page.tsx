'use client';

import { useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { useAuth } from '@/components/AuthContext';
import {
  Layers, ArrowLeft, Play, Pause, Square, Clock, Users,
  Shield, Activity, TrendingUp, ScrollText, RefreshCw,
  Download, Box, Zap
} from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useWS } from '@/components/WebSocketProvider';
import { useGetGroup, useLogs, useJoinRequests } from '@/hooks';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { MetricBar } from '@/components/ui/MetricBar';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorState } from '@/components/ui/ErrorState';
import { EmptyState } from '@/components/ui/EmptyState';
import { controlGroup, approveJoin, rejectJoin } from '@/lib/api/endpoints';
import type { Group, Client, LogEntry } from '@/lib/api/types';

export default function GroupDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { token, user } = useAuth();
  const router = useRouter();
  const [activeTab, setActiveTab] = useState('overview');
  const [logFilter, setLogFilter] = useState<string | null>(null);
  const { isConnected } = useWS();

  const { data: groupData, loading, error, refetch } = useGetGroup(id, !isConnected);
  const { data: logsData } = useLogs(!isConnected, id, logFilter || undefined);
  const { data: joinData, approveJoin: doApprove, rejectJoin: doReject } = useJoinRequests(!isConnected, id);

  const group: Group | null = (groupData as any)?.group || null;
  const logs: LogEntry[] = (logsData as any)?.logs || [];
  const joinRequests: any[] = (joinData as any)?.requests || [];

  const handleApprove = async (requestId: number) => {
    await doApprove({ request_id: requestId });
    refetch();
  };

  const handleReject = async (requestId: number) => {
    await doReject({ request_id: requestId });
    refetch();
  };

  const handleControl = async (action: 'start' | 'pause' | 'resume' | 'stop') => {
    await controlGroup(id, action);
    refetch();
  };

  if (loading) return <LoadingSpinner message="Loading group..." />;
  if (!group) return <ErrorState message="Group not found" onRetry={refetch} />;

  const clients: Client[] = Object.entries(group.clients || {}).map(([client_id, info]) => ({
    client_id,
    group_id: group.group_id,
    status: info.status || 'unknown',
    last_update: (info.last_update as number) || 0,
    update_count: info.update_count || 0,
    trust_score: info.trust_score || 0,
    joined_at: info.joined_at || '',
  }));
  const accuracy = group.latest_accuracy || 0;
  const loss = group.latest_loss || 0;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <button onClick={() => router.push('/dashboard/groups')} className="p-2 hover:bg-gray-800 rounded-lg transition">
          <ArrowLeft size={20} className="text-gray-400" />
        </button>
        <div className="flex-1">
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-white">{group.group_id}</h1>
            <StatusBadge status={group.is_training ? 'training' : group.status.toLowerCase()} />
          </div>
          <p className="text-slate-400 text-sm mt-1">{group.model_id} · Version {group.model_version || 0}</p>
        </div>

        {user?.role === 'admin' && (
          <div className="flex gap-2">
            {group.is_training ? (
              <>
                <button onClick={() => handleControl('pause')} className="btn-secondary inline-flex items-center gap-1.5 !px-3 !py-1.5 text-xs">
                  <Pause size={14} /> Pause
                </button>
                <button onClick={() => handleControl('stop')} className="btn-destructive inline-flex items-center gap-1.5 !px-3 !py-1.5 text-xs">
                  <Square size={14} /> Stop
                </button>
              </>
            ) : group.status === 'PAUSED' ? (
              <button onClick={() => handleControl('resume')} className="btn-success inline-flex items-center gap-1.5 !px-3 !py-1.5 text-xs">
                <Play size={14} /> Resume
              </button>
            ) : group.status !== 'COMPLETED' ? (
              <div className="text-sm text-slate-400 italic">Auto-starts when clients join</div>
            ) : null}
          </div>
        )}
      </div>

      <div className="flex gap-2 border-b" style={{ borderColor: 'rgba(100,100,100,0.2)' }}>
        {['overview', 'participants', 'models', 'logs', 'privacy'].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-3 text-sm font-medium capitalize transition border-b-2 ${
              activeTab === tab ? 'text-white' : 'text-slate-400 hover:text-white border-transparent'
            }`}
            style={{ borderBottomColor: activeTab === tab ? 'rgba(255,255,255,0.3)' : 'transparent' }}
          >
            {tab}
          </button>
        ))}
      </div>

      {activeTab === 'overview' && (
        <div className="space-y-6">
          {!group.is_training && group.status !== 'COMPLETED' && (
            <div className="glass-card p-4 flex items-start gap-3" style={{ borderColor: 'var(--color-info-border)' }}>
              <Activity size={20} style={{ color: 'var(--color-info)' }} className="mt-0.5" />
              <div>
                <p className="text-sm font-medium text-white">Auto-Start Enabled</p>
                <p className="text-xs text-slate-400 mt-1">Training will automatically start when the first client joins this group.</p>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="stat-card accent-info p-5">
              <div className="flex items-center justify-between mb-4">
                <span className="text-slate-400 text-xs font-medium uppercase tracking-wider">Clients</span>
                <Users size={17} className="text-gray-300" />
              </div>
              <p className="text-2xl font-bold text-white">{group.client_count ?? Object.keys(group.clients || {}).length}</p>
            </div>
            <div className="stat-card accent-success p-5">
              <div className="flex items-center justify-between mb-4">
                <span className="text-slate-400 text-xs font-medium uppercase tracking-wider">Accuracy</span>
                <TrendingUp size={17} className="text-gray-300" />
              </div>
              <p className="text-2xl font-bold text-white">{((accuracy) * 100).toFixed(1)}%</p>
              {group.metrics_source === 'server' && (
                <p className="text-[10px] text-emerald-400 mt-1">Server-verified</p>
              )}
              {group.metrics_source === 'unverified' && (
                <p className="text-[10px] text-amber-400 mt-1">No validation dataset</p>
              )}
            </div>
            <div className="stat-card accent-amber p-5">
              <div className="flex items-center justify-between mb-4">
                <span className="text-slate-400 text-xs font-medium uppercase tracking-wider">Loss</span>
                <Activity size={17} className="text-gray-300" />
              </div>
              <p className="text-2xl font-bold text-white">{loss.toFixed(4)}</p>
            </div>
            <div className="stat-card accent-violet p-5">
              <div className="flex items-center justify-between mb-4">
                <span className="text-slate-400 text-xs font-medium uppercase tracking-wider">Version</span>
                <Layers size={17} className="text-gray-300" />
              </div>
              <p className="text-2xl font-bold text-white">v{group.model_version || 0}</p>
            </div>
          </div>

          <div className="glass-card p-5">
            <h3 className="text-sm font-semibold text-white mb-4">Async Window</h3>
            <MetricBar
              value={group.window_status?.pending_updates ?? 0}
              max={group.window_size}
              colorMode="static"
              label={`Window: ${group.window_status?.pending_updates ?? 0} / ${group.window_size} updates (${group.time_limit}s timeout)`}
            />
          </div>

          {group.training_manifest && (
            <div className="glass-card p-5">
              <div className="flex items-center gap-2 mb-4">
                <ScrollText size={16} className="text-cyan-400" />
                <h3 className="text-sm font-semibold text-white">Training Contract</h3>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                {group.training_manifest.expected_delta_bytes != null && (
                  <div>
                    <p className="text-slate-500">Expected delta</p>
                    <p className="text-white font-mono mt-0.5">{group.training_manifest.expected_delta_bytes.toLocaleString()} bytes</p>
                  </div>
                )}
                {group.training_manifest.is_peft != null && (
                  <div>
                    <p className="text-slate-500">PEFT</p>
                    <p className={`font-medium mt-0.5 ${group.training_manifest.is_peft ? 'text-purple-300' : 'text-slate-300'}`}>
                      {group.training_manifest.is_peft ? 'Yes' : 'No'}
                    </p>
                  </div>
                )}
                {group.training_manifest.lr != null && (
                  <div>
                    <p className="text-slate-500">Learning rate</p>
                    <p className="text-white font-mono mt-0.5">{group.training_manifest.lr}</p>
                  </div>
                )}
                {group.training_manifest.epochs != null && (
                  <div>
                    <p className="text-slate-500">Epochs</p>
                    <p className="text-white font-mono mt-0.5">{group.training_manifest.epochs}</p>
                  </div>
                )}
                {group.training_manifest.batch_size != null && (
                  <div>
                    <p className="text-slate-500">Batch size</p>
                    <p className="text-white font-mono mt-0.5">{group.training_manifest.batch_size}</p>
                  </div>
                )}
                {group.training_manifest.target_modules != null && (
                  <div className="col-span-2">
                    <p className="text-slate-500">Target modules</p>
                    <p className="text-white font-mono mt-0.5">{group.training_manifest.target_modules.join(', ')}</p>
                  </div>
                )}
                {group.training_manifest.val_dataset != null && (
                  <div>
                    <p className="text-slate-500">Val dataset</p>
                    <p className="text-white font-mono mt-0.5">{group.training_manifest.val_dataset}</p>
                  </div>
                )}
                {group.training_manifest.lora_rank != null && (
                  <div>
                    <p className="text-slate-500">LoRA rank</p>
                    <p className="text-white font-mono mt-0.5">{group.training_manifest.lora_rank}</p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'participants' && (
        <div className="space-y-6">
          {user?.role === 'admin' && joinRequests.filter((r: any) => r.status === 'pending').length > 0 && (
            <div className="glass-card p-5" style={{ borderColor: 'var(--color-warning-border)' }}>
              <h3 className="text-sm font-semibold text-white mb-4">Pending Join Requests</h3>
              <div className="space-y-3">
                {joinRequests.filter((r: any) => r.status === 'pending').map((req: any) => (
                  <div key={req.id} className="flex items-center justify-between p-4 rounded-xl" style={{ background: 'rgba(30,41,59,0.4)' }}>
                    <div>
                      <p className="text-white text-sm font-medium">{req.username || `User ${req.user_id}`}</p>
                      <p className="text-slate-500 text-xs">{new Date(req.requested_at || req.created_at).toLocaleString()}</p>
                    </div>
                    <div className="flex gap-2">
                      <button onClick={() => handleApprove(req.id)} className="btn-success !px-3 !py-1.5 text-xs">Approve</button>
                      <button onClick={() => handleReject(req.id)} className="btn-destructive !px-3 !py-1.5 text-xs">Reject</button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {clients.length === 0 ? (
            <EmptyState icon={Users} title="No participants connected" />
          ) : (
            <div className="glass-card overflow-hidden">
              <table className="w-full">
                <thead style={{ background: 'rgba(6,9,15,0.5)' }}>
                  <tr>
                    <th className="text-left p-4 text-slate-500 text-[11px] font-semibold uppercase tracking-wider">Client</th>
                    <th className="text-left p-4 text-slate-500 text-[11px] font-semibold uppercase tracking-wider">Status</th>
                    <th className="text-left p-4 text-slate-500 text-[11px] font-semibold uppercase tracking-wider">Updates</th>
                    <th className="text-left p-4 text-slate-500 text-[11px] font-semibold uppercase tracking-wider">Trust</th>
                  </tr>
                </thead>
                <tbody>
                  {clients.map((client) => (
                    <tr key={client.client_id} className="border-t hover:bg-white/[0.02] transition-colors" style={{ borderColor: 'rgba(51,65,85,0.3)' }}>
                      <td className="p-4 text-white font-mono text-sm">{client.client_id}</td>
                      <td className="p-4"><StatusBadge status={client.status} /></td>
                      <td className="p-4 text-slate-300 text-sm">{client.update_count || 0}</td>
                      <td className="p-4"><MetricBar value={client.trust_score || 0} max={1} colorMode="trust" /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {activeTab === 'logs' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-slate-400 text-sm">{logs.length} events</span>
            <div className="flex gap-2">
              <select
                value={logFilter || ''}
                onChange={(e) => setLogFilter(e.target.value || null)}
                className="input-field !w-auto !py-1.5 text-sm"
              >
                <option value="">All Events</option>
                <option value="client_joined">Client Joined</option>
                <option value="client_update">Client Update</option>
                <option value="aggregation">Aggregation</option>
                <option value="training_started">Training Started</option>
              </select>
              <button onClick={refetch} className="btn-secondary !px-2.5 !py-2">
                <RefreshCw size={14} />
              </button>
            </div>
          </div>

          {logs.length === 0 ? (
            <EmptyState icon={Clock} title="No logs yet" message="Events will appear here as clients train and push updates" />
          ) : (
            <div className="glass-card overflow-hidden">
              <div className="max-h-[500px] overflow-y-auto">
                {logs.map((log, idx) => (
                  <div key={idx} className="p-3 border-b transition-colors hover:bg-white/[0.02]" style={{ borderColor: 'rgba(51,65,85,0.3)' }}>
                    <div className="flex items-start gap-3">
                      <span className="text-slate-600 text-xs font-mono min-w-[70px] pt-0.5">
                        {typeof log.timestamp === 'string'
                          ? new Date(log.timestamp).toLocaleTimeString()
                          : new Date((log.timestamp as number) * 1000).toLocaleTimeString()}
                      </span>
                      <div className="flex-1 min-w-0">
                        <span className="text-slate-300 text-xs font-medium uppercase">
                          {(log.type || '').replace(/_/g, ' ')}
                        </span>
                        <p className="text-white text-sm mt-0.5">{log.message}</p>
                        {log.details && Object.keys(log.details).length > 0 && (
                          <pre className="text-slate-600 text-xs mt-1 font-mono truncate">
                            {JSON.stringify(log.details)}
                          </pre>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'privacy' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="glass-card p-5">
            <h3 className="text-sm font-semibold text-white mb-4">Differential Privacy</h3>
            <div className="flex justify-between p-3 rounded-xl" style={{ background: 'rgba(30,41,59,0.4)' }}>
              <span className="text-slate-400 text-sm">Status</span>
              <span className="text-white text-sm">{(group as any)?.config?.dp_enabled ? 'Enabled' : 'Disabled'}</span>
            </div>
          </div>
          <div className="glass-card p-5">
            <h3 className="text-sm font-semibold text-white mb-4">Server Config</h3>
            <div className="space-y-2">
              {[
                ['Aggregator', (group as any)?.config?.aggregator || 'fedavg'],
                ['Learning Rate', (group as any)?.config?.lr || '—'],
                ['Differential Privacy', (group as any)?.config?.dp_enabled ? 'Enabled' : 'Disabled'],
              ].map(([label, value]) => (
                <div key={label} className="flex justify-between p-3 rounded-xl" style={{ background: 'rgba(30,41,59,0.4)' }}>
                  <span className="text-slate-400 text-sm">{label}</span>
                  <span className="text-white text-sm">{value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'models' && (
        <div className="space-y-6">
          <div className="glass-card p-5">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <Box size={18} className="text-slate-400" />
                <h3 className="text-sm font-semibold text-white">Global Model</h3>
              </div>
              <button
                onClick={() => window.open(`http://localhost:8000/api/models/${group.group_id}/download`, '_blank')}
                className="btn-secondary inline-flex items-center gap-1.5 !px-3 !py-1.5 text-xs"
              >
                <Download size={13} /> Download v{group.model_version || 0}
              </button>
            </div>
            <div className="grid grid-cols-3 gap-4">
              <div className="p-4 rounded-xl" style={{ background: 'rgba(30,41,59,0.4)' }}>
                <p className="text-slate-500 text-xs">Model</p>
                <p className="text-white font-medium mt-1">{group.model_id}</p>
              </div>
              <div className="p-4 rounded-xl" style={{ background: 'rgba(30,41,59,0.4)' }}>
                <p className="text-slate-500 text-xs">Version</p>
                <p className="text-white font-medium mt-1">v{group.model_version || 0}</p>
              </div>
              <div className="p-4 rounded-xl" style={{ background: 'rgba(30,41,59,0.4)' }}>
                <p className="text-slate-500 text-xs">Rounds</p>
                <p className="text-white font-medium mt-1">{group.completed_rounds || 0}</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
