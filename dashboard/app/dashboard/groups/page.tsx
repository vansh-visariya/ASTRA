'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/components/AuthContext';
import Link from 'next/link';
import { Layers, Eye, Play, Pause, Square, RefreshCw, Lock, Clock, Users } from 'lucide-react';
import { useRouter } from 'next/navigation';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Group {
  group_id: string;
  model_id: string;
  status: string;
  is_training: boolean;
  is_locked: boolean;
  join_token: string;
  config: { local_epochs: number; batch_size: number; lr: number; dp_enabled: boolean };
  window_config: { window_size: number; time_limit: number };
  window_status: { pending_updates: number; trigger_reason: string; time_remaining: number };
  client_count: number;
  model_version: number;
}

export default function GroupsPage() {
  const { token, user } = useAuth();
  const [groups, setGroups] = useState<Group[]>([]);
  const [loading, setLoading] = useState(true);
  const [showToken, setShowToken] = useState<Record<string, boolean>>({});
  const router = useRouter();

  const fetchGroups = async () => {
    try {
      const res = await fetch(`${API_URL}/api/groups`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setGroups(data.groups || []);
      }
    } catch (e) {
      console.error('Failed to fetch groups:', e);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchGroups();
    const interval = setInterval(fetchGroups, 2000);
    return () => clearInterval(interval);
  }, [token]);

  const controlGroup = async (groupId: string, action: 'start' | 'pause' | 'resume' | 'stop') => {
    try {
      await fetch(`${API_URL}/api/groups/${groupId}/${action}`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      fetchGroups();
    } catch (e) {
      console.error(`Failed to ${action} group:`, e);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'TRAINING': return 'bg-green-900/50 text-green-400 border-green-800';
      case 'PAUSED': return 'bg-yellow-900/50 text-yellow-400 border-yellow-800';
      case 'COMPLETED': return 'bg-blue-900/50 text-blue-400 border-blue-800';
      case 'FAILED': return 'bg-red-900/50 text-red-400 border-red-800';
      default: return 'bg-gray-800 text-gray-400 border-gray-700';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Groups</h1>
          <p className="text-gray-400">Manage federated learning groups</p>
        </div>
        <div className="flex gap-3">
          <button onClick={fetchGroups} className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition">
            <RefreshCw size={18} className="text-gray-400" />
          </button>
          {user?.role === 'coordinator' && (
            <Link href="/dashboard/create" className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 rounded-lg text-white font-medium transition">
              + Create Group
            </Link>
          )}
        </div>
      </div>

      {groups.length === 0 ? (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-12 text-center">
          <Layers size={48} className="mx-auto text-gray-600 mb-4" />
          <h3 className="text-white font-semibold mb-2">No groups yet</h3>
          <p className="text-gray-400 mb-4">Create your first federated learning group to get started.</p>
          {user?.role === 'coordinator' && (
            <Link href="/dashboard/create" className="inline-flex px-4 py-2 bg-indigo-600 hover:bg-indigo-700 rounded-lg text-white font-medium transition">
              Create Group
            </Link>
          )}
        </div>
      ) : (
        <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
          <table className="w-full">
            <thead className="bg-gray-950">
              <tr>
                <th className="text-left p-4 text-gray-400 text-sm font-medium">Group</th>
                <th className="text-left p-4 text-gray-400 text-sm font-medium">Model</th>
                <th className="text-left p-4 text-gray-400 text-sm font-medium">Status</th>
                <th className="text-left p-4 text-gray-400 text-sm font-medium">Clients</th>
                <th className="text-left p-4 text-gray-400 text-sm font-medium">Async Window</th>
                <th className="text-left p-4 text-gray-400 text-sm font-medium">Version</th>
                <th className="text-left p-4 text-gray-400 text-sm font-medium">Join Token</th>
                <th className="text-right p-4 text-gray-400 text-sm font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {groups.map((group) => (
                <tr key={group.group_id} className="border-t border-gray-800 hover:bg-gray-800/50 transition">
                  <td className="p-4">
                    <div className="flex items-center gap-2">
                      <Layers size={18} className="text-indigo-400" />
                      <span className="text-white font-medium">{group.group_id}</span>
                      {group.is_locked && <Lock size={14} className="text-yellow-500" />}
                    </div>
                  </td>
                  <td className="p-4 text-gray-300">{group.model_id}</td>
                  <td className="p-4">
                    <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getStatusColor(group.status)}`}>
                      {group.status}
                    </span>
                  </td>
                  <td className="p-4 text-gray-300">
                    <div className="flex items-center gap-1">
                      <Users size={14} className="text-gray-500" />
                      {group.client_count}
                    </div>
                  </td>
                  <td className="p-4 text-gray-300">
                    <div className="flex items-center gap-1">
                      <Clock size={14} className="text-gray-500" />
                      {group.window_config.window_size} / {group.window_config.time_limit}s
                    </div>
                  </td>
                  <td className="p-4 text-gray-300">v{group.model_version}</td>
                  <td className="p-4">
                    <code className="text-xs bg-gray-800 px-2 py-1 rounded text-gray-300">
                      {showToken[group.group_id] ? group.join_token : '••••••••••••'}
                    </code>
                    <button
                      onClick={() => setShowToken({ ...showToken, [group.group_id]: !showToken[group.group_id] })}
                      className="ml-2 text-gray-500 hover:text-white"
                    >
                      {showToken[group.group_id] ? <Eye size={14} /> : <Eye size={14} />}
                    </button>
                  </td>
                  <td className="p-4">
                    <div className="flex items-center justify-end gap-2">
                      {!group.is_training && group.status !== 'COMPLETED' && user?.role === 'coordinator' && (
                        <button onClick={() => controlGroup(group.group_id, 'start')} className="p-2 bg-green-600 hover:bg-green-700 rounded-lg transition" title="Start">
                          <Play size={14} className="text-white" />
                        </button>
                      )}
                      {group.is_training && user?.role === 'coordinator' && (
                        <>
                          <button onClick={() => controlGroup(group.group_id, 'pause')} className="p-2 bg-yellow-600 hover:bg-yellow-700 rounded-lg transition" title="Pause">
                            <Pause size={14} className="text-white" />
                          </button>
                          <button onClick={() => controlGroup(group.group_id, 'stop')} className="p-2 bg-red-600 hover:bg-red-700 rounded-lg transition" title="Stop">
                            <Square size={14} className="text-white" />
                          </button>
                        </>
                      )}
                      {group.status === 'PAUSED' && user?.role === 'coordinator' && (
                        <button onClick={() => controlGroup(group.group_id, 'resume')} className="p-2 bg-green-600 hover:bg-green-700 rounded-lg transition" title="Resume">
                          <Play size={14} className="text-white" />
                        </button>
                      )}
                      <button onClick={() => router.push(`/dashboard/groups/${group.group_id}`)} className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition" title="View Details">
                        <Layers size={14} className="text-white" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
