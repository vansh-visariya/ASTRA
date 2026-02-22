'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/components/AuthContext';
import { Layers, Users, Activity, Shield, Zap, TrendingUp, Plus } from 'lucide-react';
import Link from 'next/link';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface SystemMetrics {
  total_groups: number;
  active_groups: number;
  total_participants: number;
  active_participants: number;
  dp_enabled_groups: number;
  total_aggregations: number;
  latest_group_id?: string | null;
  latest_accuracy?: number;
  latest_loss?: number;
  latest_version?: number;
  latest_timestamp?: number;
}

export default function DashboardPage() {
  const { token } = useAuth();
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const res = await fetch(`${API_URL}/api/system/metrics`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        if (res.ok) {
          const data = await res.json();
          setMetrics(data);
        }
      } catch (e) {
        console.error('Failed to fetch metrics:', e);
      }
    };
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 3000);
    return () => clearInterval(interval);
  }, [token]);

  const formatPercent = (value?: number) => `${((value || 0) * 100).toFixed(1)}%`;
  const formatLoss = (value?: number) => (value ?? 0).toFixed(4);
  const formatVersion = (value?: number) => `v${value || 0}`;

  const statCards: { label: string; value: string | number; icon: any; color: string }[] = [
    { label: 'Total Groups', value: metrics?.total_groups || 0, icon: Layers, color: 'indigo' },
    { label: 'Active Groups', value: metrics?.active_groups || 0, icon: Activity, color: 'green' },
    { label: 'Total Participants', value: metrics?.total_participants || 0, icon: Users, color: 'blue' },
    { label: 'Active Participants', value: metrics?.active_participants || 0, icon: Zap, color: 'yellow' },
    { label: 'DP Enabled', value: metrics?.dp_enabled_groups || 0, icon: Shield, color: 'purple' },
    { label: 'Total Aggregations', value: metrics?.total_aggregations || 0, icon: TrendingUp, color: 'pink' },
    { label: 'Latest Accuracy', value: formatPercent(metrics?.latest_accuracy), icon: TrendingUp, color: 'green' },
    { label: 'Latest Loss', value: formatLoss(metrics?.latest_loss), icon: Activity, color: 'red' },
    { label: 'Latest Round', value: formatVersion(metrics?.latest_version), icon: Layers, color: 'indigo' },
  ];

  const colorMap: Record<string, string> = {
    indigo: 'bg-indigo-600',
    green: 'bg-green-600',
    blue: 'bg-blue-600',
    yellow: 'bg-yellow-600',
    purple: 'bg-purple-600',
    pink: 'bg-pink-600',
    red: 'bg-red-600',
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        <p className="text-gray-400">System overview</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {statCards.map((stat, idx) => (
          <div key={idx} className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <div className="flex items-center justify-between mb-3">
              <span className="text-gray-400 text-sm">{stat.label}</span>
              <div className={`w-8 h-8 rounded-lg ${colorMap[stat.color]} flex items-center justify-center`}>
                <stat.icon size={16} className="text-white" />
              </div>
            </div>
            <p className="text-3xl font-bold text-white">{stat.value}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Link href="/dashboard/groups" className="bg-gray-900 border border-gray-800 rounded-xl p-6 hover:border-indigo-500 transition group">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-indigo-600/20 rounded-xl flex items-center justify-center group-hover:bg-indigo-600 transition">
              <Layers size={24} className="text-indigo-400" />
            </div>
            <div>
              <h3 className="text-white font-semibold">Manage Groups</h3>
              <p className="text-gray-400 text-sm">View and control federated groups</p>
            </div>
          </div>
        </Link>

        <Link href="/dashboard/create" className="bg-gray-900 border border-gray-800 rounded-xl p-6 hover:border-green-500 transition group">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-green-600/20 rounded-xl flex items-center justify-center group-hover:bg-green-600 transition">
              <Plus size={24} className="text-green-400" />
            </div>
            <div>
              <h3 className="text-white font-semibold">Create New Group</h3>
              <p className="text-gray-400 text-sm">Start a new federated learning experiment</p>
            </div>
          </div>
        </Link>
      </div>
    </div>
  );
}
