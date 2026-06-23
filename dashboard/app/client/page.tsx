'use client';

import Link from 'next/link';
import {
  Users, Shield, Activity,
  TrendingUp, CheckCircle,
  ArrowUpRight, Upload, Bell, AlertCircle,
} from 'lucide-react';
import { useWS } from '@/components/WebSocketProvider';
import { useGroups, useTrustScores, useNotifications } from '@/hooks';
import { StatCard } from '@/components/ui/StatCard';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorState } from '@/components/ui/ErrorState';
import { useAuth } from '@/components/AuthContext';
import type { Notification, TrustData, ApiListResponse } from '@/lib/api/types';

export default function ClientDashboard() {
  const { user } = useAuth();
  const { isConnected } = useWS();
  const { data: groupsData, loading: groupsLoading, error: groupsError, refetch: refetchGroups } = useGroups(isConnected);
  const { data: trustData, loading: trustLoading } = useTrustScores(isConnected, user?.id);
  const { data: notifData } = useNotifications(isConnected, { limit: 5 });

  const groups = (groupsData as any)?.groups || [];
  const trustScore = (trustData as TrustData | null)?.score ?? 1.0;
  const recentNotifications: Notification[] = (notifData as any)?.notifications || [];

  const loading = groupsLoading || trustLoading;
  const error = groupsError;

  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 17) return 'Good afternoon';
    return 'Good evening';
  };

  if (loading && !groups.length) return <LoadingSpinner message="Loading dashboard..." />;
  if (error) return <ErrorState message={error} onRetry={refetchGroups} />;

  return (
    <div className="space-y-6">
      <div className="animate-fade-in">
        <h1 className="text-2xl font-bold text-white">{getGreeting()}, {user?.full_name || 'Client'}</h1>
        <p className="text-slate-400 text-sm mt-1">Here's your federated learning overview</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Available Groups" value={groups.length} icon={Users} accent="info" delay={0.05} />
        <StatCard label="Groups Joined" value={0} icon={CheckCircle} accent="success" delay={0.1} />
        <StatCard label="Trust Score" value={`${(trustScore * 100).toFixed(0)}%`} icon={Shield} accent="violet" delay={0.15} />
        <StatCard label="Rounds Done" value={0} icon={TrendingUp} accent="blue" delay={0.2} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 animate-fade-in" style={{ animationDelay: '0.25s', opacity: 0 }}>
        <Link href="/client/groups" className="glass-card p-5 group cursor-pointer">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-11 h-11 rounded-xl flex items-center justify-center transition-all duration-300 group-hover:scale-110"
                style={{ background: 'rgba(255,255,255,0.06)' }}>
                <Users className="text-gray-300" size={20} />
              </div>
              <div>
                <h3 className="text-white font-semibold text-sm">Join a Group</h3>
                <p className="text-slate-500 text-xs mt-0.5">Browse and request to join training groups</p>
              </div>
            </div>
            <ArrowUpRight size={16} className="text-slate-600 group-hover:text-white transition-colors" />
          </div>
        </Link>
        <Link href="/client/upload" className="glass-card p-5 group cursor-pointer">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-11 h-11 rounded-xl flex items-center justify-center transition-all duration-300 group-hover:scale-110"
                style={{ background: 'rgba(255,255,255,0.06)' }}>
                <Upload className="text-gray-300" size={20} />
              </div>
              <div>
                <h3 className="text-white font-semibold text-sm">Upload Delta</h3>
                <p className="text-slate-500 text-xs mt-0.5">Submit your pre-computed model delta</p>
              </div>
            </div>
            <ArrowUpRight size={16} className="text-slate-600 group-hover:text-white transition-colors" />
          </div>
        </Link>
      </div>

      <div className="glass-card p-5 animate-fade-in" style={{ animationDelay: '0.35s', opacity: 0 }}>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-white uppercase tracking-wider">Recent Notifications</h2>
          <Link href="/client/notifications" className="text-gray-300 text-xs font-medium hover:text-white transition flex items-center gap-1">
            View all <ArrowUpRight size={12} />
          </Link>
        </div>

        {recentNotifications.length > 0 ? (
          <div className="space-y-2">
            {recentNotifications.slice(0, 5).map((notif) => (
              <div
                key={notif.id}
                className={`flex items-start gap-3 p-3 rounded-xl transition-colors ${notif.read ? 'opacity-60' : ''}`}
                style={{ background: notif.read ? 'transparent' : 'rgba(30, 41, 59, 0.3)' }}
              >
                {notif.priority === 'error' || notif.priority === 'warning' ? (
                  <AlertCircle className="text-slate-400 shrink-0 mt-0.5" size={16} />
                ) : (
                  <Activity className="text-gray-300 shrink-0 mt-0.5" size={16} />
                )}
                <div className="flex-1 min-w-0">
                  <p className="text-white text-sm font-medium">{notif.title}</p>
                  <p className="text-slate-500 text-xs truncate mt-0.5">{notif.message}</p>
                </div>
                <span className="text-slate-600 text-[10px] shrink-0 font-mono">
                  {new Date(notif.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <Bell className="mx-auto mb-2 text-slate-700" size={28} />
            <p className="text-slate-500 text-sm">No notifications yet</p>
          </div>
        )}
      </div>
    </div>
  );
}
