'use client';

import { useState, useEffect } from 'react';
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
import { getAnnouncements } from '@/lib/api/endpoints';
import type { Notification, TrustData, Announcement, ApiListResponse } from '@/lib/api/types';

export default function ClientDashboard() {
  const { user } = useAuth();
  const { isConnected } = useWS();
  const { data: groupsData, loading: groupsLoading, error: groupsError, refetch: refetchGroups } = useGroups(isConnected);
  const { data: trustData, loading: trustLoading } = useTrustScores(isConnected, user?.id);
  const { data: notifData } = useNotifications(isConnected, { limit: 5 });

  const groups = (groupsData as any)?.groups || [];
  const trustScore = (trustData as TrustData | null)?.score ?? 1.0;
  const recentNotifications: Notification[] = (notifData as any)?.notifications || [];

  const groupsJoined = groups.filter((g: any) => {
    const clientInfo = g.clients?.[user?.id || ''];
    return clientInfo && (clientInfo.status === 'active' || clientInfo.status === 'joined');
  }).length;

  const roundsDone = groups.reduce((sum: number, g: any) => {
    const clientInfo = g.clients?.[user?.id || ''];
    return sum + (clientInfo?.updates_count || 0);
  }, 0);

  const [announcements, setAnnouncements] = useState<Announcement[]>([]);

  const loading = groupsLoading || trustLoading;
  const error = groupsError;

  useEffect(() => {
    if (groups.length > 0) {
      const fetchAnnouncements = async () => {
        const all: Announcement[] = [];
        for (const g of groups.slice(0, 5)) {
          try {
            const res: any = await getAnnouncements(g.group_id);
            all.push(...(res?.announcements || []).slice(0, 3));
          } catch {}
        }
        all.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
        setAnnouncements(all.slice(0, 5));
      };
      fetchAnnouncements();
    }
  }, [groups]);

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
        <StatCard label="Groups Joined" value={groupsJoined} icon={CheckCircle} accent="success" delay={0.1} />
        <StatCard label="Trust Score" value={`${(trustScore * 100).toFixed(0)}%`} icon={Shield} accent="violet" delay={0.15} />
        <StatCard label="Rounds Done" value={roundsDone} icon={TrendingUp} accent="blue" delay={0.2} />
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

      {groups.filter((g: any) => g.clients?.[user?.id || '']?.status === 'active').length > 0 && (
        <div className="glass-card p-5 animate-fade-in" style={{ animationDelay: '0.3s', opacity: 0 }}>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-white uppercase tracking-wider">Your Groups</h2>
          </div>
          <div className="space-y-2">
            {groups
              .filter((g: any) => g.clients?.[user?.id || '']?.status === 'active')
              .slice(0, 5)
              .map((g: any) => (
                <Link
                  key={g.group_id}
                  href={`/client/chat/${g.group_id}`}
                  className="flex items-center justify-between p-3 rounded-xl hover:bg-white/5 transition"
                >
                  <div>
                    <p className="text-white text-sm font-medium">{g.group_id}</p>
                    <p className="text-slate-500 text-xs">{g.model_id} · v{g.model_version || 0}</p>
                  </div>
                  <span className="text-slate-600 text-xs">Chat →</span>
                </Link>
              ))}
          </div>
        </div>
      )}

      {announcements.length > 0 && (
        <div className="glass-card p-5 animate-fade-in" style={{ animationDelay: '0.32s', opacity: 0 }}>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-white uppercase tracking-wider">Announcements</h2>
          </div>
          <div className="space-y-2">
            {announcements.map((a) => (
              <div
                key={a.id}
                className="p-3 rounded-xl"
                style={{
                  background: a.priority === 'error' ? 'rgba(239,68,68,0.08)' :
                    a.priority === 'warning' ? 'rgba(245,158,11,0.08)' : 'rgba(30,41,59,0.3)',
                }}
              >
                <div className="flex items-start justify-between">
                  <p className="text-white text-sm">{a.message}</p>
                  <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium shrink-0 ml-2 ${
                    a.priority === 'error' ? 'bg-red-500/20 text-red-400' :
                    a.priority === 'warning' ? 'bg-amber-500/20 text-amber-400' :
                    'bg-blue-500/20 text-blue-400'
                  }`}>
                    {a.priority}
                  </span>
                </div>
                <p className="text-slate-500 text-[10px] mt-1">
                  {a.author_name} · {a.group_id} · {new Date(a.created_at).toLocaleString()}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

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
