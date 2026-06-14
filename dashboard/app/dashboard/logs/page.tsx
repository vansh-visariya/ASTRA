'use client';

import { useState } from 'react';
import { Clock, RefreshCw } from 'lucide-react';
import { useWS } from '@/components/WebSocketProvider';
import { useLogs } from '@/hooks';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorState } from '@/components/ui/ErrorState';
import { EmptyState } from '@/components/ui/EmptyState';
import type { LogEntry } from '@/lib/api/types';

const EVENT_TYPES = ['training_started', 'aggregation', 'client_joined', 'client_rejected'];

const TYPE_DOT_COLOR: Record<string, string> = {
  training_started: 'var(--color-success)',
  aggregation: 'var(--color-info)',
  client_joined: 'var(--color-accent-violet)',
  client_rejected: 'var(--color-error)',
};

const TYPE_LABEL_COLOR: Record<string, string> = {
  training_started: 'var(--color-success)',
  aggregation: 'var(--color-info)',
  client_joined: '#d1d5db',
  client_rejected: 'var(--color-error)',
};

export default function LogsPage() {
  const [filter, setFilter] = useState<string | null>(null);
  const { isConnected } = useWS();
  const { data, loading, error, refetch } = useLogs(!isConnected, undefined, filter || undefined);

  const logs: LogEntry[] = (data as any)?.logs || [];

  const formatTime = (timestamp: number | string) => {
    const ts = typeof timestamp === 'string' ? Date.parse(timestamp) / 1000 : timestamp;
    return new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  if (loading && !logs.length) return <LoadingSpinner message="Loading logs..." />;
  if (error) return <ErrorState message={error} onRetry={refetch} />;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between animate-fade-in">
        <div>
          <h1 className="text-2xl font-bold text-white">Event Logs</h1>
          <p className="text-slate-400 text-sm mt-1">Server events and training history</p>
        </div>
        <div className="flex gap-2">
          <select
            value={filter || ''}
            onChange={(e) => setFilter(e.target.value || null)}
            className="input-field !w-auto !py-2 text-sm"
          >
            <option value="">All Events</option>
            {EVENT_TYPES.map((type) => (
              <option key={type} value={type}>{type.replace(/_/g, ' ')}</option>
            ))}
          </select>
          <button onClick={refetch} className="btn-secondary !px-3 !py-2">
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      {logs.length === 0 ? (
        <EmptyState icon={Clock} title="No logs yet" message="Events will appear here when training starts" />
      ) : (
        <div className="glass-card overflow-hidden animate-fade-in">
          <div className="max-h-[650px] overflow-y-auto">
            {logs.map((log, idx) => (
              <div
                key={idx}
                className="p-4 border-b transition-colors hover:bg-white/[0.02]"
                style={{ borderColor: 'rgba(51,65,85,0.3)' }}
              >
                <div className="flex items-start gap-4">
                  <div className="text-slate-600 text-xs font-mono min-w-[72px] mt-0.5">
                    {formatTime(log.timestamp as unknown as number)}
                  </div>
                  <div
                    className="w-2 h-2 rounded-full mt-1.5 shrink-0"
                    style={{ background: TYPE_DOT_COLOR[log.type] || 'var(--color-muted)' }}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-0.5">
                      <span
                        className="text-[11px] font-semibold uppercase tracking-wider"
                        style={{ color: TYPE_LABEL_COLOR[log.type] || 'var(--color-muted)' }}
                      >
                        {log.type.replace(/_/g, ' ')}
                      </span>
                      {log.group_id && (
                        <span className="text-[10px] text-slate-600 font-mono px-1.5 py-0.5 rounded"
                          style={{ background: 'rgba(30,41,59,0.5)' }}>
                          {log.group_id}
                        </span>
                      )}
                    </div>
                    <p className="text-slate-300 text-sm">{log.message}</p>
                    {log.details && Object.keys(log.details).length > 0 && (
                      <pre className="text-slate-600 text-[11px] mt-1.5 font-mono leading-relaxed">
                        {JSON.stringify(log.details, null, 2)}
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
  );
}
