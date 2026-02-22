'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/components/AuthContext';
import { 
  Users, Lock, Clock, CheckCircle, 
  XCircle, Loader2, Plus, RefreshCw
} from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Group {
  group_id: string;
  model_id: string;
  status: string;
  join_token: string;
  window_config: {
    window_size: number;
    time_limit: number;
  };
  client_count: number;
  active_clients: string[];
}

interface JoinRequest {
  status: string;
  message?: string;
}

export default function ClientGroupsPage() {
  const { token } = useAuth();
  const [groups, setGroups] = useState<Group[]>([]);
  const [loading, setLoading] = useState(true);
  const [joining, setJoining] = useState<string | null>(null);
  const [joinStatus, setJoinStatus] = useState<Record<string, JoinRequest>>({});

  const fetchGroups = async () => {
    if (!token) return;
    setLoading(true);
    
    try {
      const res = await fetch(`${API_URL}/api/groups`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const data = await res.json();
      setGroups(data.groups || []);
      
      // Check join status for each group
      const statuses: Record<string, JoinRequest> = {};
      for (const group of data.groups || []) {
        try {
          const statusRes = await fetch(`${API_URL}/api/groups/my-requests/${group.group_id}`, {
            headers: { 'Authorization': `Bearer ${token}` }
          });
          const statusData = await statusRes.json();
          statuses[group.group_id] = { status: statusData.status || 'none' };
        } catch {
          statuses[group.group_id] = { status: 'none' };
        }
      }
      setJoinStatus(statuses);
    } catch (e) {
      console.error('Failed to fetch groups:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchGroups();
  }, [token]);

  const handleJoinRequest = async (groupId: string) => {
    if (!token) return;
    setJoining(groupId);
    
    try {
      const res = await fetch(`${API_URL}/api/groups/join-request`, {
        method: 'POST',
        headers: { 
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          group_id: groupId,
          metadata: {
            requested_at: new Date().toISOString()
          }
        })
      });
      
      const data = await res.json();
      
      setJoinStatus(prev => ({
        ...prev,
        [groupId]: { 
          status: res.ok ? 'pending' : 'error',
          message: data.detail || 'Request submitted'
        }
      }));
    } catch (e) {
      setJoinStatus(prev => ({
        ...prev,
        [groupId]: { status: 'error', message: 'Failed to submit request' }
      }));
    } finally {
      setJoining(null);
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'pending':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-yellow-500/20 text-yellow-400 text-xs">
            <Clock size={12} /> Pending Approval
          </span>
        );
      case 'approved':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-green-500/20 text-green-400 text-xs">
            <CheckCircle size={12} /> Approved
          </span>
        );
      case 'rejected':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-red-500/20 text-red-400 text-xs">
            <XCircle size={12} /> Rejected
          </span>
        );
      default:
        return null;
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
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Available Groups</h1>
          <p className="text-gray-400">Browse and join federated learning groups</p>
        </div>
        <button
          onClick={fetchGroups}
          className="flex items-center gap-2 px-4 py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-700 transition"
        >
          <RefreshCw size={16} /> Refresh
        </button>
      </div>

      {groups.length === 0 ? (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-12 text-center">
          <Users className="mx-auto mb-4 text-gray-600" size={48} />
          <h3 className="text-lg font-medium text-white mb-2">No Groups Available</h3>
          <p className="text-gray-400">There are no groups available to join at the moment.</p>
        </div>
      ) : (
        <div className="grid gap-4">
          {groups.map((group) => (
            <div 
              key={group.group_id}
              className="bg-gray-900 border border-gray-800 rounded-xl p-6"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3">
                    <h3 className="text-lg font-semibold text-white">{group.group_id}</h3>
                    {getStatusBadge(joinStatus[group.group_id]?.status || 'none')}
                  </div>
                  <p className="text-gray-400 text-sm mt-1">
                    Model: <span className="text-emerald-400">{group.model_id}</span>
                  </p>
                  
                  <div className="flex items-center gap-4 mt-3 text-sm text-gray-500">
                    <span className="flex items-center gap-1">
                      <Users size={14} /> {group.client_count} clients
                    </span>
                    <span className="flex items-center gap-1">
                      <Clock size={14} /> Window: {group.window_config.window_size} updates
                    </span>
                    <span className="flex items-center gap-1">
                      <Lock size={14} /> Token required
                    </span>
                  </div>
                </div>
                
                <div className="ml-4">
                  {joinStatus[group.group_id]?.status === 'pending' ? (
                    <button
                      disabled
                      className="px-4 py-2 bg-gray-800 text-gray-400 rounded-lg flex items-center gap-2 cursor-not-allowed"
                    >
                      <Clock size={16} /> Pending
                    </button>
                  ) : joinStatus[group.group_id]?.status === 'approved' ? (
                    <button
                      className="px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition flex items-center gap-2"
                    >
                      <CheckCircle size={16} /> Join Group
                    </button>
                  ) : joining === group.group_id ? (
                    <button
                      disabled
                      className="px-4 py-2 bg-gray-800 text-white rounded-lg flex items-center gap-2"
                    >
                      <Loader2 size={16} className="animate-spin" /> Requesting...
                    </button>
                  ) : (
                    <button
                      onClick={() => handleJoinRequest(group.group_id)}
                      className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition flex items-center gap-2"
                    >
                      <Plus size={16} /> Request to Join
                    </button>
                  )}
                </div>
              </div>
              
              {joinStatus[group.group_id]?.message && joinStatus[group.group_id]?.status !== 'pending' && (
                <div className={`mt-4 p-3 rounded-lg text-sm ${
                  joinStatus[group.group_id]?.status === 'error' 
                    ? 'bg-red-900/30 text-red-400' 
                    : 'bg-gray-800 text-gray-400'
                }`}>
                  {joinStatus[group.group_id]?.message}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
