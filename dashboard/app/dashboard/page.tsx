'use client';

import { Layers, Users, Activity, Zap, Shield, TrendingUp, Plus, ArrowUpRight } from 'lucide-react';
import Link from 'next/link';
import { useMemo } from 'react';
import { useWS } from '@/components/WebSocketProvider';
import { useMetrics } from '@/hooks';
import { StatCard } from '@/components/ui/StatCard';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorState } from '@/components/ui/ErrorState';
import { useAuth } from '@/components/AuthContext';

export default function DashboardPage() {
  const { user } = useAuth();
  const { isConnected } = useWS();
  const { data: metrics, loading, error, refetch } = useMetrics(!isConnected);

  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 17) return 'Good afternoon';
    return 'Good evening';
  };

  const formatPercent = (value?: number) => `${((value || 0) * 100).toFixed(1)}%`;
  const formatLoss = (value?: number) => (value ?? 0).toFixed(4);

  const statCards = useMemo(() => [
    { label: 'Total Groups', value: metrics?.total_groups || 0, icon: Layers, accent: 'info' as const },
    { label: 'Active Groups', value: metrics?.active_groups || 0, icon: Activity, accent: 'success' as const },
    { label: 'Total Participants', value: metrics?.total_participants || 0, icon: Users, accent: 'violet' as const },
    { label: 'Active Participants', value: metrics?.active_participants || 0, icon: Zap, accent: 'amber' as const },
    { label: 'DP Enabled', value: metrics?.dp_enabled_groups || 0, icon: Shield, accent: 'warning' as const },
    { label: 'Total Rounds', value: metrics?.total_aggregations || 0, icon: TrendingUp, accent: 'muted' as const },
  ], [metrics]);

  if (loading) return <LoadingSpinner message="Loading dashboard..." />;
  if (error) return <ErrorState message={error} onRetry={refetch} />;

  return (
    <div className="space-y-6">
      <div className="animate-fade-in">
        <h1 className="text-2xl font-bold text-white">{getGreeting()}, {user?.name || 'Admin'}</h1>
        <p className="text-slate-400 text-sm mt-1">Here's your federated learning overview</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {statCards.map((stat, idx) => (
          <StatCard
            key={stat.label}
            label={stat.label}
            value={stat.value}
            icon={stat.icon}
            accent={stat.accent}
            delay={idx * 0.05}
          />
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="stat-card accent-success p-5 animate-fade-in" style={{ animationDelay: '0.35s', opacity: 0 }}>
          <span className="text-slate-400 text-xs font-medium uppercase tracking-wider">Latest Accuracy</span>
          <p className="text-3xl font-bold text-white tracking-tight mt-3">{formatPercent(metrics?.latest_accuracy)}</p>
          <p className="text-slate-500 text-xs mt-2">{metrics?.latest_group_id ? `Group: ${metrics.latest_group_id}` : 'No data'}</p>
        </div>
        <div className="stat-card accent-info p-5 animate-fade-in" style={{ animationDelay: '0.4s', opacity: 0 }}>
          <span className="text-slate-400 text-xs font-medium uppercase tracking-wider">Latest Loss</span>
          <p className="text-3xl font-bold text-white tracking-tight mt-3">{formatLoss(metrics?.latest_loss)}</p>
          <p className="text-slate-500 text-xs mt-2">Round v{metrics?.latest_version || 0}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 animate-fade-in" style={{ animationDelay: '0.45s', opacity: 0 }}>
        <Link href="/dashboard/groups" className="glass-card p-5 group cursor-pointer">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-11 h-11 rounded-xl flex items-center justify-center transition-all duration-300 group-hover:scale-110"
                style={{ background: 'rgba(255,255,255,0.06)' }}>
                <Layers size={20} className="text-gray-300" />
              </div>
              <div>
                <h3 className="text-white font-semibold text-sm">Manage Groups</h3>
                <p className="text-slate-500 text-xs mt-0.5">View and control federated groups</p>
              </div>
            </div>
            <ArrowUpRight size={16} className="text-slate-600 group-hover:text-white transition-colors" />
          </div>
        </Link>
        <Link href="/dashboard/create" className="glass-card p-5 group cursor-pointer">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-11 h-11 rounded-xl flex items-center justify-center transition-all duration-300 group-hover:scale-110"
                style={{ background: 'rgba(255,255,255,0.06)' }}>
                <Plus size={20} className="text-gray-300" />
              </div>
              <div>
                <h3 className="text-white font-semibold text-sm">Create New Group</h3>
                <p className="text-slate-500 text-xs mt-0.5">Start a new federated learning experiment</p>
              </div>
            </div>
            <ArrowUpRight size={16} className="text-slate-600 group-hover:text-white transition-colors" />
          </div>
        </Link>
      </div>
    </div>
  );
}
