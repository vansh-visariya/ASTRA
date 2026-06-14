import React from 'react';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { LaunchCommand } from '@/components/training/LaunchCommand';
import { buildClientCommand } from '@/lib/commands';
import type { TrainingSession } from '@/lib/api/types';

interface SessionCardProps {
  session: TrainingSession;
  index: number;
}

export function SessionCard({ session, index }: SessionCardProps) {
  const {
    group_id,
    model_id,
    status,
    is_training,
    client_id,
    local_accuracy,
    local_loss,
    updates_sent,
    trust_score,
    last_update,
    global_model_version,
    global_accuracy,
    global_loss,
    window_status,
  } = session;

  const isConnected = status === 'active' || status === 'training';
  const command = buildClientCommand(
    'http://localhost:8000',
    client_id || undefined,
    group_id,
  );

  const metrics = [
    { label: 'Local Accuracy', value: local_accuracy != null ? `${(local_accuracy * 100).toFixed(1)}%` : '–' },
    { label: 'Local Loss', value: local_loss != null ? local_loss.toFixed(4) : '–' },
    { label: 'Updates Sent', value: String(updates_sent || 0) },
    { label: 'Trust Score', value: trust_score != null ? `${(trust_score * 100).toFixed(0)}%` : '–' },
    { label: 'Last Update', value: last_update ? `${Math.floor((Date.now() / 1000 - last_update) / 60)}m ago` : '–' },
  ];

  return (
    <div
      className="session-card p-5 animate-fade-in"
      style={{ animationDelay: `${0.15 + index * 0.05}s`, opacity: 0 }}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <h3 className="text-white font-semibold text-sm">{group_id}</h3>
          <StatusBadge status={isConnected ? 'active' : 'offline'} />
        </div>
        <span className="text-slate-500 text-xs">Model: {model_id}</span>
      </div>

      <div className="session-metrics mb-4">
        {metrics.map((m) => (
          <div key={m.label} className="session-metric">
            <p className="text-slate-500 text-[10px] uppercase tracking-wider mb-1">{m.label}</p>
            <p className="text-white font-semibold text-sm">{m.value}</p>
          </div>
        ))}
      </div>

      <div
        className="flex items-center justify-between p-3 rounded-lg mb-3"
        style={{ background: 'rgba(15, 15, 15, 0.6)', border: '1px solid rgba(100, 100, 100, 0.15)' }}
      >
        <div className="flex items-center gap-4">
          <span className="text-slate-500 text-xs">Global v{global_model_version || 0}</span>
          <span className="text-slate-500 text-xs">
            Acc: {global_accuracy != null ? `${(global_accuracy * 100).toFixed(1)}%` : '–'}
          </span>
          <span className="text-slate-500 text-xs">
            Loss: {global_loss != null ? global_loss.toFixed(4) : '–'}
          </span>
        </div>
        <span className="text-slate-600 text-xs">
          Window: {window_status?.current_size || 0}/{window_status?.max_size || 0}
        </span>
      </div>

      <LaunchCommand command={command} />
    </div>
  );
}
