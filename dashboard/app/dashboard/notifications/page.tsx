'use client';

import { useState } from 'react';
import { Bell, Check, CheckCheck, AlertCircle, AlertTriangle, CheckCircle, Info } from 'lucide-react';
import { useWS } from '@/components/WebSocketProvider';
import { useNotifications } from '@/hooks';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorState } from '@/components/ui/ErrorState';
import { EmptyState } from '@/components/ui/EmptyState';
import type { Notification } from '@/lib/api/types';

const PRIORITY_ICONS: Record<string, React.ElementType> = {
  error: AlertCircle,
  warning: AlertTriangle,
  success: CheckCircle,
};

const PRIORITY_COLORS: Record<string, { icon: string; border: string }> = {
  error: { icon: 'var(--color-error)', border: 'var(--color-error-border)' },
  warning: { icon: 'var(--color-warning)', border: 'var(--color-warning-border)' },
  success: { icon: 'var(--color-success)', border: 'var(--color-success-border)' },
  info: { icon: 'var(--color-info)', border: 'var(--color-info-border)' },
};

export default function AdminNotificationsPage() {
  const [filter, setFilter] = useState<'all' | 'unread'>('all');
  const { isConnected } = useWS();
  const {
    data,
    loading,
    error,
    refetch,
    markRead,
    markAllRead,
  } = useNotifications(!isConnected, { limit: 50, unread_only: filter === 'unread' });

  const notifications: Notification[] = (data as any)?.notifications || [];
  const unreadCount = notifications.filter((n) => !n.read).length;

  const handleMarkAllRead = async () => {
    await markAllRead();
    refetch();
  };

  const handleMarkRead = async (id: number) => {
    await markRead(id);
  };

  if (loading && !notifications.length) return <LoadingSpinner message="Loading notifications..." />;
  if (error) return <ErrorState message={error} onRetry={refetch} />;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between animate-fade-in">
        <div>
          <h1 className="text-2xl font-bold text-white">Notifications</h1>
          <p className="text-slate-400 text-sm mt-1">
            {unreadCount > 0 ? `${unreadCount} unread notifications` : 'All caught up!'}
          </p>
        </div>
        {unreadCount > 0 && (
          <button onClick={handleMarkAllRead} className="btn-secondary inline-flex items-center gap-2">
            <CheckCheck size={14} /> Mark all as read
          </button>
        )}
      </div>

      <div className="flex gap-1 animate-fade-in" style={{ animationDelay: '0.05s', opacity: 0 }}>
        <button
          onClick={() => setFilter('all')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition ${filter === 'all' ? 'bg-white/10 text-white' : 'text-slate-400 hover:text-white'}`}
        >
          All
        </button>
        <button
          onClick={() => setFilter('unread')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition ${filter === 'unread' ? 'bg-white/10 text-white' : 'text-slate-400 hover:text-white'}`}
        >
          Unread
          {unreadCount > 0 && (
            <span className="ml-1.5 px-1.5 py-0.5 text-white text-[11px] rounded-full" style={{ background: 'var(--color-error)' }}>
              {unreadCount}
            </span>
          )}
        </button>
      </div>

      {notifications.length === 0 ? (
        <EmptyState
          icon={Bell}
          title="No Notifications"
          message={filter === 'unread' ? "You've read all your notifications" : 'No notifications yet'}
        />
      ) : (
        <div className="space-y-3 animate-fade-in" style={{ animationDelay: '0.1s', opacity: 0 }}>
          {notifications.map((n) => {
            const Icon = PRIORITY_ICONS[n.priority] || Info;
            const colors = PRIORITY_COLORS[n.priority] || PRIORITY_COLORS.info;

            return (
              <div
                key={n.id}
                className={`glass-card p-4 ${n.read ? 'opacity-60' : ''}`}
                style={!n.read ? { borderColor: colors.border } : undefined}
              >
                <div className="flex items-start gap-3">
                  <Icon size={18} style={{ color: colors.icon }} className="shrink-0 mt-0.5" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <h3 className="text-white font-medium text-sm">{n.title}</h3>
                        <p className="text-slate-400 text-sm mt-1">{n.message}</p>
                      </div>
                      {!n.read && (
                        <button
                          onClick={() => handleMarkRead(n.id)}
                          className="shrink-0 p-1.5 text-slate-500 hover:text-white rounded-lg transition"
                          title="Mark as read"
                        >
                          <Check size={14} />
                        </button>
                      )}
                    </div>
                    <div className="flex items-center gap-3 mt-3 text-xs text-slate-500">
                      <span>{new Date(n.created_at).toLocaleString()}</span>
                      {n.group_id && <span>Group: {n.group_id}</span>}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
