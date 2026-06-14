import React from 'react';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { Layers } from 'lucide-react';
import type { Group } from '@/lib/api/types';

interface GroupTableProps {
  groups: Group[];
  onAction: (groupId: string, action: 'start' | 'pause' | 'resume' | 'stop') => void;
  onDetail: (groupId: string) => void;
}

export function GroupTable({ groups, onAction, onDetail }: GroupTableProps) {
  if (groups.length === 0) return null;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-slate-500 text-xs uppercase tracking-wider">
            <th className="text-left py-3 px-4 font-medium">Group ID</th>
            <th className="text-left py-3 px-4 font-medium">Model</th>
            <th className="text-left py-3 px-4 font-medium">Status</th>
            <th className="text-left py-3 px-4 font-medium">Clients</th>
            <th className="text-left py-3 px-4 font-medium">Config</th>
            <th className="text-right py-3 px-4 font-medium">Actions</th>
          </tr>
        </thead>
        <tbody>
          {groups.map((group) => (
            <tr
              key={group.group_id}
              className="border-t border-white/5 hover:bg-white/[0.02] transition-colors cursor-pointer"
              onClick={() => onDetail(group.group_id)}
            >
              <td className="py-3 px-4">
                <div className="flex items-center gap-2">
                  <Layers size={14} className="text-slate-600" />
                  <span className="text-white font-medium">{group.group_id}</span>
                </div>
              </td>
              <td className="py-3 px-4 text-slate-400">{group.model_id}</td>
              <td className="py-3 px-4">
                <StatusBadge status={group.is_training ? 'training' : group.status.toLowerCase()} />
              </td>
              <td className="py-3 px-4 text-slate-400">
                {Object.keys(group.clients || {}).length}
              </td>
              <td className="py-3 px-4 text-slate-500 text-xs">
                W{group.window_size}/T{group.time_limit}s
              </td>
              <td className="py-3 px-4 text-right" onClick={(e) => e.stopPropagation()}>
                <div className="flex items-center justify-end gap-1.5">
                  {group.is_training ? (
                    <button
                      onClick={() => onAction(group.group_id, 'pause')}
                      className="btn-secondary text-xs !px-3 !py-1"
                    >
                      Pause
                    </button>
                  ) : group.status === 'PAUSED' ? (
                    <button
                      onClick={() => onAction(group.group_id, 'resume')}
                      className="btn-success text-xs !px-3 !py-1"
                    >
                      Resume
                    </button>
                  ) : group.status === 'COMPLETED' || group.status === 'FAILED' ? (
                    <button
                      onClick={() => onAction(group.group_id, 'start')}
                      className="btn-success text-xs !px-3 !py-1"
                    >
                      Restart
                    </button>
                  ) : (
                    <button
                      onClick={() => onAction(group.group_id, 'start')}
                      className="btn-primary text-xs !px-3 !py-1"
                    >
                      Start
                    </button>
                  )}
                  {(group.is_training || group.status === 'TRAINING') && (
                    <button
                      onClick={() => onAction(group.group_id, 'stop')}
                      className="btn-destructive text-xs !px-3 !py-1"
                    >
                      Stop
                    </button>
                  )}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
