'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, BarChart, Bar } from 'recharts';
import { Activity, Users, Shield, Lock, Database, Play, Pause, Square, Settings, Brain, Wifi, Gauge, Layers, Zap, Plus, Trash2, Eye, EyeOff, RefreshCw, Clock, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Group {
  group_id: string;
  model_id: string;
  status: string;
  is_training: boolean;
  is_locked: boolean;
  join_token: string;
  window_config: { window_size: number; time_limit: number };
  window_status: { pending_updates: number; window_size: number; time_elapsed: number; time_limit: number; time_remaining: number; trigger_reason: string };
  client_count: number;
  model_version: number;
  active_clients: string[];
  metrics_history: any[];
}

interface Client {
  client_id: string;
  group_id: string;
  status: string;
  trust_score: number;
  data_metadata?: { modality: string; samples: number };
}

export default function Dashboard() {
  const [connected, setConnected] = useState(false);
  const [groups, setGroups] = useState<Group[]>([]);
  const [clients, setClients] = useState<Client[]>([]);
  const [serverStatus, setServerStatus] = useState<any>(null);
  const [activeTab, setActiveTab] = useState('groups');
  const [loading, setLoading] = useState(false);
  const [showToken, setShowToken] = useState<Record<string, boolean>>({});
  
  const [newGroup, setNewGroup] = useState({
    group_id: '',
    model_id: 'simple_cnn_mnist',
    window_size: 3,
    time_limit: 20,
    local_epochs: 2,
    batch_size: 32,
    lr: 0.01
  });

  const fetchData = useCallback(async () => {
    try {
      const [groupsRes, clientsRes, statusRes] = await Promise.all([
        fetch(`${API_URL}/api/groups`).catch(() => ({ json: () => ({ groups: [] }) })),
        fetch(`${API_URL}/api/clients`).catch(() => ({ json: () => ({ clients: [] }) })),
        fetch(`${API_URL}/api/server/status`).catch(() => ({ json: () => ({}) }))
      ]);
      
      const groupsData = await groupsRes.json();
      const clientsData = await clientsRes.json();
      const statusData = await statusRes.json();
      
      setGroups(groupsData.groups || []);
      setClients(clientsData.clients || []);
      setServerStatus(statusData);
      setConnected(true);
    } catch {
      setConnected(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const createGroup = async () => {
    if (!newGroup.group_id) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/api/groups`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newGroup)
      });
      if (res.ok) {
        setNewGroup({ group_id: '', model_id: 'simple_cnn_mnist', window_size: 3, time_limit: 20 });
        fetchData();
      }
    } catch (e) {
      console.error('Failed to create group:', e);
    }
    setLoading(false);
  };

  const controlGroup = async (groupId: string, action: 'start' | 'pause' | 'resume' | 'stop') => {
    setLoading(true);
    try {
      await fetch(`${API_URL}/api/groups/${groupId}/${action}`, { method: 'POST' });
      fetchData();
    } catch (e) {
      console.error(`Failed to ${action} group:`, e);
    }
    setLoading(false);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'TRAINING': return 'bg-green-900/50 text-green-400';
      case 'PAUSED': return 'bg-yellow-900/50 text-yellow-400';
      case 'COMPLETED': return 'bg-blue-900/50 text-blue-400';
      case 'FAILED': return 'bg-red-900/50 text-red-400';
      default: return 'bg-gray-700 text-gray-400';
    }
  };

  const getTriggerColor = (trigger: string) => {
    switch (trigger) {
      case 'size': return 'bg-blue-900/50 text-blue-400';
      case 'time': return 'bg-orange-900/50 text-orange-400';
      default: return 'bg-gray-700 text-gray-400';
    }
  };

  const tabs = [
    { id: 'groups', label: 'Groups', icon: Layers },
    { id: 'clients', label: 'Clients', icon: Users },
    { id: 'models', label: 'Models', icon: Brain },
    { id: 'security', label: 'Security', icon: Shield },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  const renderGroups = () => (
    <div className="space-y-6">
      <div className="bg-gray-800 border border-gray-700 rounded-xl p-5">
        <h3 className="font-semibold text-lg mb-4">Create New Group</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-7 gap-4">
          <input
            type="text"
            placeholder="Group ID"
            value={newGroup.group_id}
            onChange={(e) => setNewGroup({ ...newGroup, group_id: e.target.value })}
            className="bg-gray-900 border border-gray-700 rounded-lg p-3"
          />
          <select
            value={newGroup.model_id}
            onChange={(e) => setNewGroup({ ...newGroup, model_id: e.target.value })}
            className="bg-gray-900 border border-gray-700 rounded-lg p-3"
          >
            <option value="simple_cnn_mnist">CNN (MNIST)</option>
            <option value="simple_cnn_cifar">CNN (CIFAR)</option>
            <option value="transformer">Transformer</option>
          </select>
          <input
            type="number"
            placeholder="Epochs"
            value={newGroup.local_epochs}
            onChange={(e) => setNewGroup({ ...newGroup, local_epochs: parseInt(e.target.value) })}
            className="bg-gray-900 border border-gray-700 rounded-lg p-3"
          />
          <input
            type="number"
            placeholder="Batch Size"
            value={newGroup.batch_size}
            onChange={(e) => setNewGroup({ ...newGroup, batch_size: parseInt(e.target.value) })}
            className="bg-gray-900 border border-gray-700 rounded-lg p-3"
          />
          <input
            type="number"
            step="0.001"
            placeholder="LR"
            value={newGroup.lr}
            onChange={(e) => setNewGroup({ ...newGroup, lr: parseFloat(e.target.value) })}
            className="bg-gray-900 border border-gray-700 rounded-lg p-3"
          />
          <input
            type="number"
            placeholder="Window"
            value={newGroup.window_size}
            onChange={(e) => setNewGroup({ ...newGroup, window_size: parseInt(e.target.value) })}
            className="bg-gray-900 border border-gray-700 rounded-lg p-3"
          />
          <button
            onClick={createGroup}
            disabled={loading || !newGroup.group_id}
            className="px-4 py-3 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 rounded-lg font-medium transition flex items-center justify-center gap-2"
          >
            <Plus size={18} /> Create
          </button>
        </div>
      </div>

      {groups.length === 0 ? (
        <div className="bg-gray-800 border border-gray-700 rounded-xl-center">
           p-8 text<Layers size={48} className="mx-auto text-gray-600 mb-4" />
          <p className="text-gray-400">No groups yet. Create one above to get started.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {groups.map((group) => (
            <div key={group.group_id} className="bg-gray-800 border border-gray-700 rounded-xl p-5">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <Layers size={20} className="text-indigo-400" />
                  <h4 className="font-semibold text-lg">{group.group_id}</h4>
                  {group.is_locked && <Lock size={14} className="text-yellow-500" />}
                </div>
                <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(group.status)}`}>
                  {group.status}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="bg-gray-900 rounded-lg p-3">
                  <p className="text-gray-400 text-xs">Model</p>
                  <p className="font-medium text-sm">{group.model_id}</p>
                </div>
                <div className="bg-gray-900 rounded-lg p-3">
                  <p className="text-gray-400 text-xs">Clients</p>
                  <p className="font-medium text-sm">{group.client_count}</p>
                </div>
                <div className="bg-gray-900 rounded-lg p-3">
                  <p className="text-gray-400 text-xs">Version</p>
                  <p className="font-medium text-sm">v{group.model_version}</p>
                </div>
                <div className="bg-gray-900 rounded-lg p-3">
                  <p className="text-gray-400 text-xs">Join Token</p>
                  <p className="font-medium text-xs font-mono">
                    {showToken[group.group_id] ? (group.debug_token || group.join_token) : '••••••••••••'}
                    <button onClick={() => setShowToken({ ...showToken, [group.group_id]: !showToken[group.group_id] })} className="ml-2 text-gray-400 hover:text-white">
                      {showToken[group.group_id] ? <EyeOff size={12} /> : <Eye size={12} />}
                    </button>
                  </p>
                  {group.debug_token && (
                    <p className="text-xs text-red-400 mt-1">DEBUG: {group.debug_token}</p>
                  )}
                </div>
              </div>

              <div className="bg-gray-900 rounded-lg p-4 mb-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-gray-400 text-sm">Async Window</span>
                  <span className="text-xs text-gray-500">
                    {group.window_status?.pending_updates || 0} / {group.window_config?.window_size || 3} updates
                  </span>
                </div>
                
                <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden mb-3">
                  <div 
                    className="h-full bg-indigo-500 rounded-full transition-all duration-500"
                    style={{ width: `${Math.min(100, ((group.window_status?.pending_updates || 0) / (group.window_config?.window_size || 3)) * 100)}%` }}
                  />
                </div>
                
                <div className="flex items-center justify-between text-sm mb-2">
                  <div className="flex items-center gap-2">
                    <Clock size={14} className="text-gray-400" />
                    <span className="text-gray-400">Timer:</span>
                    <span className="font-medium">
                      {group.window_status?.time_remaining?.toFixed(1) || 0}s remaining
                    </span>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-gray-400 text-xs">Trigger:</span>
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${getTriggerColor(group.window_status?.trigger_reason || 'waiting')}`}>
                      {group.window_status?.trigger_reason === 'size' ? 'Size-based' : 
                       group.window_status?.trigger_reason === 'time' ? 'Time-based' : 
                       `Waiting (${group.window_config?.window_size} updates OR ${group.window_config?.time_limit}s)`}
                    </span>
                  </div>
                </div>
              </div>

              <div className="flex gap-2">
                {!group.is_training && group.status !== 'COMPLETED' && (
                  <button onClick={() => controlGroup(group.group_id, 'start')} className="flex-1 px-3 py-2 bg-green-600 hover:bg-green-700 rounded-lg text-sm font-medium transition flex items-center justify-center gap-2">
                    <Play size={14} /> Start
                  </button>
                )}
                {group.is_training && (
                  <>
                    <button onClick={() => controlGroup(group.group_id, 'pause')} className="flex-1 px-3 py-2 bg-yellow-600 hover:bg-yellow-700 rounded-lg text-sm font-medium transition flex items-center justify-center gap-2">
                      <Pause size={14} /> Pause
                    </button>
                    <button onClick={() => controlGroup(group.group_id, 'stop')} className="flex-1 px-3 py-2 bg-red-600 hover:bg-red-700 rounded-lg text-sm font-medium transition flex items-center justify-center gap-2">
                      <Square size={14} /> Stop
                    </button>
                  </>
                )}
                {!group.is_training && group.status === 'PAUSED' && (
                  <button onClick={() => controlGroup(group.group_id, 'resume')} className="flex-1 px-3 py-2 bg-green-600 hover:bg-green-700 rounded-lg text-sm font-medium transition flex items-center justify-center gap-2">
                    <Play size={14} /> Resume
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  const renderClients = () => (
    <div className="bg-gray-800 rounded-xl overflow-hidden border border-gray-700">
      <div className="p-5 border-b border-gray-700">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold">Connected Clients</h3>
          <button onClick={fetchData} className="p-2 hover:bg-gray-700 rounded-lg transition">
            <RefreshCw size={18} />
          </button>
        </div>
      </div>
      {clients.length === 0 ? (
        <div className="p-8 text-center text-gray-400">
          <Users size={48} className="mx-auto mb-4 opacity-50" />
          <p>No clients connected</p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-900">
              <tr>
                <th className="text-left p-4 text-gray-400 text-sm font-medium">Client</th>
                <th className="text-left p-4 text-gray-400 text-sm font-medium">Group</th>
                <th className="text-left p-4 text-gray-400 text-sm font-medium">Status</th>
                <th className="text-left p-4 text-gray-400 text-sm font-medium">Trust</th>
                <th className="text-left p-4 text-gray-400 text-sm font-medium">Data</th>
              </tr>
            </thead>
            <tbody>
              {clients.map((client, idx) => (
                <tr key={idx} className="border-t border-gray-700 hover:bg-gray-750 transition">
                  <td className="p-4 font-medium">{client.client_id}</td>
                  <td className="p-4 text-gray-400">{client.group_id}</td>
                  <td className="p-4">
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                      client.status === 'active' ? 'bg-green-900/50 text-green-400' : 'bg-gray-700 text-gray-400'
                    }`}>
                      {client.status}
                    </span>
                  </td>
                  <td className="p-4">
                    <div className="flex items-center gap-2">
                      <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
                        <div className={`h-full rounded-full ${
                          client.trust_score > 0.7 ? 'bg-green-500' : client.trust_score > 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                        }`} style={{ width: `${client.trust_score * 100}%` }} />
                      </div>
                      <span className="text-sm text-gray-400">{(client.trust_score * 100).toFixed(0)}%</span>
                    </div>
                  </td>
                  <td className="p-4 text-gray-400">
                    {client.data_metadata?.modality || 'N/A'}
                    {client.data_metadata?.samples && ` (${client.data_metadata.samples})`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );

  const renderModels = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {['simple_cnn_mnist', 'simple_cnn_cifar', 'transformer'].map((model, idx) => (
        <div key={idx} className="bg-gray-800 border border-gray-700 rounded-xl p-5">
          <div className="flex items-center justify-between mb-3">
            <Brain size={24} className="text-indigo-400" />
            <span className="px-2 py-1 rounded text-xs bg-gray-700 text-gray-400">Available</span>
          </div>
          <h4 className="font-semibold mb-1">{model}</h4>
          <p className="text-gray-400 text-sm">Vision • CNN/Transformer</p>
        </div>
      ))}
    </div>
  );

  const renderSecurity = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div className="bg-gray-800 border border-gray-700 rounded-xl p-5">
        <div className="flex items-center gap-3 mb-4">
          <Shield size={24} className="text-green-400" />
          <h3 className="font-semibold">Robust Aggregation</h3>
        </div>
        <div className="space-y-3">
          <div className="flex justify-between p-3 bg-gray-900 rounded-lg">
            <span className="text-gray-400">Method</span>
            <span className="font-medium">Hybrid (Trust + Staleness)</span>
          </div>
          <div className="flex justify-between p-3 bg-gray-900 rounded-lg">
            <span className="text-gray-400">Filtering</span>
            <span className="font-medium">Multi-stage</span>
          </div>
        </div>
      </div>
      <div className="bg-gray-800 border border-gray-700 rounded-xl p-5">
        <div className="flex items-center gap-3 mb-4">
          <Lock size={24} className="text-indigo-400" />
          <h3 className="font-semibold">Differential Privacy</h3>
        </div>
        <div className="space-y-3">
          <div className="flex justify-between p-3 bg-gray-900 rounded-lg">
            <span className="text-gray-400">Status</span>
            <span className="text-yellow-400 font-medium">Configurable</span>
          </div>
          <div className="flex justify-between p-3 bg-gray-900 rounded-lg">
            <span className="text-gray-400">Clip Norm</span>
            <span className="font-medium">1.0</span>
          </div>
        </div>
      </div>
    </div>
  );

  const renderSettings = () => (
    <div className="bg-gray-800 border border-gray-700 rounded-xl p-5">
      <h3 className="font-semibold mb-4">Server Controls</h3>
      <div className="flex gap-3">
        <button className="flex items-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg font-medium transition">
          <Play size={18} /> Start Server
        </button>
        <button className="flex items-center gap-2 px-6 py-3 bg-red-600 hover:bg-red-700 rounded-lg font-medium transition">
          <Square size={18} /> Stop Server
        </button>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center">
                <Layers size={20} className="text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold">Federated AI Platform</h1>
                <p className="text-xs text-gray-500">Distributed Learning Dashboard</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-700 rounded-lg">
                <Wifi size={14} className={connected ? 'text-green-400' : 'text-red-400'} />
                <span className="text-sm text-gray-400">{connected ? 'Connected' : 'Offline'}</span>
              </div>
              <div className="text-right">
                <p className="text-sm font-medium">{groups.length} Groups</p>
                <p className="text-xs text-gray-500">{clients.length} Clients</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      <nav className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-1 py-2">
            {tabs.map((tab) => (
              <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition ${
                  activeTab === tab.id ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:text-white hover:bg-gray-700'
                }`}>
                <tab.icon size={16} /> {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 py-6">
        {activeTab === 'groups' && renderGroups()}
        {activeTab === 'clients' && renderClients()}
        {activeTab === 'models' && renderModels()}
        {activeTab === 'security' && renderSecurity()}
        {activeTab === 'settings' && renderSettings()}
      </main>
    </div>
  );
}
