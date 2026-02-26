'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/components/AuthContext';
import Link from 'next/link';
import { 
  Users, TrainFront, Shield, Activity,
  TrendingUp, Clock, CheckCircle, AlertCircle, Bell
} from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface DashboardStats {
  groupsAvailable: number;
  groupsJoined: number;
  trustScore: number;
  roundsCompleted: number;
}

interface Notification {
  id: number;
  type: string;
  priority: string;
  title: string;
  message: string;
  created_at: string;
  read: boolean;
}

export default function ClientDashboard() {
  const { token, user } = useAuth();
  const [stats, setStats] = useState<DashboardStats>({
    groupsAvailable: 0,
    groupsJoined: 0,
    trustScore: 1.0,
    roundsCompleted: 0
  });
  const [recentNotifications, setRecentNotifications] = useState<Notification[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      if (!token) return;
      
      try {
        // Fetch available groups
        const groupsRes = await fetch(`${API_URL}/api/groups`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        const groupsData = await groupsRes.json();
        
        // Fetch trust score
        const trustRes = await fetch(`${API_URL}/api/trust/scores/${user?.id}`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        const trustData = await trustRes.json();
        
        // Fetch recent notifications
        const notifRes = await fetch(`${API_URL}/api/notifications?limit=5`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        const notifData = await notifRes.json();

        setStats({
          groupsAvailable: groupsData.count || 0,
          groupsJoined: 0, // Will be calculated
          trustScore: trustData.score || 1.0,
          roundsCompleted: 0
        });
        
        setRecentNotifications(notifData.notifications || []);
      } catch (e) {
        console.error('Failed to fetch data:', e);
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, [token, user]);

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'error': return 'text-red-400';
      case 'warning': return 'text-yellow-400';
      case 'success': return 'text-green-400';
      default: return 'text-blue-400';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Welcome back, {user?.name || 'Client'}!</h1>
        <p className="text-gray-400">Here's your federated learning overview</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Available Groups</p>
              <p className="text-2xl font-bold text-white mt-1">{stats.groupsAvailable}</p>
            </div>
            <div className="w-12 h-12 bg-indigo-600/20 rounded-lg flex items-center justify-center">
              <Users className="text-indigo-400" size={24} />
            </div>
          </div>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Groups Joined</p>
              <p className="text-2xl font-bold text-white mt-1">{stats.groupsJoined}</p>
            </div>
            <div className="w-12 h-12 bg-emerald-600/20 rounded-lg flex items-center justify-center">
              <CheckCircle className="text-emerald-400" size={24} />
            </div>
          </div>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Trust Score</p>
              <p className="text-2xl font-bold text-white mt-1">{(stats.trustScore * 100).toFixed(0)}%</p>
            </div>
            <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
              stats.trustScore > 0.7 ? 'bg-green-600/20' : 'bg-yellow-600/20'
            }`}>
              <Shield className={stats.trustScore > 0.7 ? 'text-green-400' : 'text-yellow-400'} size={24} />
            </div>
          </div>
          <div className="mt-3 h-2 bg-gray-800 rounded-full overflow-hidden">
            <div 
              className={`h-full rounded-full transition-all ${
                stats.trustScore > 0.7 ? 'bg-green-500' : 'bg-yellow-500'
              }`}
              style={{ width: `${stats.trustScore * 100}%` }}
            />
          </div>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Rounds Completed</p>
              <p className="text-2xl font-bold text-white mt-1">{stats.roundsCompleted}</p>
            </div>
            <div className="w-12 h-12 bg-blue-600/20 rounded-lg flex items-center justify-center">
              <TrendingUp className="text-blue-400" size={24} />
            </div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Link 
          href="/client/groups"
          className="bg-gray-900 border border-gray-800 rounded-xl p-6 hover:border-emerald-500/50 transition group"
        >
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 bg-emerald-600/20 rounded-xl flex items-center justify-center group-hover:bg-emerald-600/30 transition">
              <Users className="text-emerald-400" size={28} />
            </div>
            <div>
              <h3 className="text-white font-semibold text-lg">Join a Group</h3>
              <p className="text-gray-400 text-sm">Browse and request to join training groups</p>
            </div>
          </div>
        </Link>

        <Link 
          href="/client/training"
          className="bg-gray-900 border border-gray-800 rounded-xl p-6 hover:border-emerald-500/50 transition group"
        >
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 bg-indigo-600/20 rounded-xl flex items-center justify-center group-hover:bg-indigo-600/30 transition">
              <TrainFront className="text-indigo-400" size={28} />
            </div>
            <div>
              <h3 className="text-white font-semibold text-lg">Start Training</h3>
              <p className="text-gray-400 text-sm">Begin local model training</p>
            </div>
          </div>
        </Link>
      </div>

      {/* Recent Notifications */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">Recent Notifications</h2>
          <Link href="/client/notifications" className="text-emerald-400 text-sm hover:underline">
            View all
          </Link>
        </div>
        
        {recentNotifications.length > 0 ? (
          <div className="space-y-3">
            {recentNotifications.slice(0, 5).map((notif) => (
              <div 
                key={notif.id}
                className={`flex items-start gap-3 p-3 rounded-lg ${
                  notif.read ? 'bg-gray-800/50' : 'bg-gray-800'
                }`}
              >
                {notif.priority === 'error' || notif.priority === 'warning' ? (
                  <AlertCircle className="text-yellow-400 shrink-0" size={18} />
                ) : (
                  <Activity className={getPriorityColor(notif.priority) + ' shrink-0'} size={18} />
                )}
                <div className="flex-1 min-w-0">
                  <p className="text-white text-sm font-medium">{notif.title}</p>
                  <p className="text-gray-400 text-sm truncate">{notif.message}</p>
                </div>
                <span className="text-gray-500 text-xs shrink-0">
                  {new Date(notif.created_at).toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <Bell className="mx-auto mb-2 opacity-50" size={32} />
            <p>No notifications yet</p>
          </div>
        )}
      </div>
    </div>
  );
}
