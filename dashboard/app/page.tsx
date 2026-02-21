'use client';

import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { Activity, Users, Shield, Lock, Database, Play, Pause, Square, Settings, Brain, Wifi, Gauge, Layers, Zap } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const mockMetrics = Array.from({ length: 20 }, (_, i) => ({
  step: i * 10,
  accuracy: 0.1 + (i / 20) * 0.75 + Math.random() * 0.05,
  loss: 2.5 - (i / 20) * 1.8 + Math.random() * 0.1,
}));

const mockClients = [
  { id: 'client_001', status: 'active', trust: 0.98, latency: 45, accuracy: 0.89, gradient: 0.23 },
  { id: 'client_002', status: 'active', trust: 0.95, latency: 52, accuracy: 0.87, gradient: 0.31 },
  { id: 'client_003', status: 'active', trust: 0.72, latency: 120, accuracy: 0.76, gradient: 0.89 },
  { id: 'client_004', status: 'slow', trust: 0.68, latency: 2500, accuracy: 0.71, gradient: 0.45 },
  { id: 'client_005', status: 'quarantined', trust: 0.25, latency: 0, accuracy: 0.0, gradient: 5.2 },
];

const mockModels = [
  { id: 'simple_cnn_mnist', type: 'Vision', params: '421K', peft: false, status: 'active' },
  { id: 'clip-vit-base', type: 'MultiModal', params: '151M', peft: true, status: 'available' },
  { id: 'resnet50', type: 'Vision', params: '25.6M', peft: false, status: 'available' },
];

export default function Dashboard() {
  const [connected, setConnected] = useState(false);
  const [serverStatus, setServerStatus] = useState<any>(null);
  const [metrics] = useState(mockMetrics);
  const [activeTab, setActiveTab] = useState('overview');
  const [experimentRunning, setExperimentRunning] = useState(false);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch(`${API_URL}/api/server/status`);
        const data = await res.json();
        setServerStatus(data);
        setConnected(true);
      } catch {
        setConnected(false);
      }
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Gauge },
    { id: 'clients', label: 'Clients', icon: Users },
    { id: 'models', label: 'Models', icon: Brain },
    { id: 'security', label: 'Security', icon: Shield },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  const renderOverview = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <div className="bg-gray-800 rounded-xl p-5 border border-gray-700">
        <div className="flex items-center justify-between mb-3">
          <span className="text-gray-400 text-sm">Status</span>
          <div className={`w-2 h-2 rounded-full ${experimentRunning ? 'bg-green-500 animate-pulse' : 'bg-yellow-500'}`} />
        </div>
        <p className="text-2xl font-bold">{experimentRunning ? 'Training' : 'Idle'}</p>
        <p className="text-gray-500 text-xs mt-1">Round: {serverStatus?.global_version || 0}</p>
      </div>

      <div className="bg-gray-800 rounded-xl p-5 border border-gray-700">
        <div className="flex items-center justify-between mb-3">
          <span className="text-gray-400 text-sm">Active Clients</span>
          <Users size={18} className="text-indigo-400" />
        </div>
        <p className="text-2xl font-bold">{mockClients.filter(c => c.status === 'active').length}</p>
        <p className="text-gray-500 text-xs mt-1">Total: {mockClients.length}</p>
      </div>

      <div className="bg-gray-800 rounded-xl p-5 border border-gray-700">
        <div className="flex items-center justify-between mb-3">
          <span className="text-gray-400 text-sm">Accuracy</span>
          <Activity size={18} className="text-green-400" />
        </div>
        <p className="text-2xl font-bold text-green-400">
          {metrics.length > 0 ? (metrics[metrics.length - 1].accuracy * 100).toFixed(1) : '0.0'}%
        </p>
        <p className="text-gray-500 text-xs mt-1">+2.3% from last round</p>
      </div>

      <div className="bg-gray-800 rounded-xl p-5 border border-gray-700">
        <div className="flex items-center justify-between mb-3">
          <span className="text-gray-400 text-sm">Privacy (ε)</span>
          <Lock size={18} className="text-indigo-400" />
        </div>
        <p className="text-2xl font-bold">1.45</p>
        <p className="text-gray-500 text-xs mt-1">Differential Privacy</p>
      </div>

      <div className="col-span-1 md:col-span-2 bg-gray-800 rounded-xl p-5 border border-gray-700">
        <h3 className="font-semibold mb-4">Accuracy Over Time</h3>
        <ResponsiveContainer width="100%" height={250}>
          <AreaChart data={metrics}>
            <defs>
              <linearGradient id="colorAcc" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="step" stroke="#6b7280" fontSize={12} />
            <YAxis stroke="#6b7280" fontSize={12} domain={[0, 1]} />
            <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }} />
            <Area type="monotone" dataKey="accuracy" stroke="#6366f1" strokeWidth={2} fillOpacity={1} fill="url(#colorAcc)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="col-span-1 md:col-span-2 bg-gray-800 rounded-xl p-5 border border-gray-700">
        <h3 className="font-semibold mb-4">Loss Over Time</h3>
        <ResponsiveContainer width="100%" height={250}>
          <AreaChart data={metrics}>
            <defs>
              <linearGradient id="colorLoss" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="step" stroke="#6b7280" fontSize={12} />
            <YAxis stroke="#6b7280" fontSize={12} />
            <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }} />
            <Area type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2} fillOpacity={1} fill="url(#colorLoss)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  const renderClients = () => (
    <div className="bg-gray-800 rounded-xl overflow-hidden border border-gray-700">
      <div className="p-5 border-b border-gray-700 flex items-center justify-between">
        <h3 className="font-semibold">Connected Clients</h3>
        <button className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 rounded-lg text-sm font-medium transition">
          + Add Client
        </button>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-900">
            <tr>
              <th className="text-left p-4 text-gray-400 text-sm font-medium">Client</th>
              <th className="text-left p-4 text-gray-400 text-sm font-medium">Status</th>
              <th className="text-left p-4 text-gray-400 text-sm font-medium">Trust</th>
              <th className="text-left p-4 text-gray-400 text-sm font-medium">Latency</th>
              <th className="text-left p-4 text-gray-400 text-sm font-medium">Accuracy</th>
              <th className="text-left p-4 text-gray-400 text-sm font-medium">Grad Norm</th>
            </tr>
          </thead>
          <tbody>
            {mockClients.map((client, idx) => (
              <tr key={idx} className="border-t border-gray-700 hover:bg-gray-750 transition">
                <td className="p-4 font-medium">{client.id}</td>
                <td className="p-4">
                  <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                    client.status === 'active' ? 'bg-green-900/50 text-green-400' :
                    client.status === 'slow' ? 'bg-yellow-900/50 text-yellow-400' :
                    'bg-red-900/50 text-red-400'
                  }`}>
                    {client.status}
                  </span>
                </td>
                <td className="p-4">
                  <div className="flex items-center gap-2">
                    <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div className={`h-full rounded-full ${
                        client.trust > 0.7 ? 'bg-green-500' : client.trust > 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                      }`} style={{ width: `${client.trust * 100}%` }} />
                    </div>
                    <span className="text-sm text-gray-400">{(client.trust * 100).toFixed(0)}%</span>
                  </div>
                </td>
                <td className="p-4 text-gray-400">{client.latency > 0 ? `${client.latency}ms` : '-'}</td>
                <td className="p-4">{client.accuracy > 0 ? `${(client.accuracy * 100).toFixed(1)}%` : '-'}</td>
                <td className="p-4 text-gray-400">{client.gradient.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  const renderModels = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {mockModels.map((model, idx) => (
        <div key={idx} className="bg-gray-800 border border-gray-700 rounded-xl p-5 hover:border-indigo-500/50 transition cursor-pointer">
          <div className="flex items-center justify-between mb-3">
            <Brain size={24} className="text-indigo-400" />
            <span className={`px-2 py-1 rounded text-xs ${model.status === 'active' ? 'bg-green-900/50 text-green-400' : 'bg-gray-700 text-gray-400'}`}>
              {model.status}
            </span>
          </div>
          <h4 className="font-semibold mb-1">{model.id}</h4>
          <p className="text-gray-400 text-sm mb-3">{model.type} • {model.params} params</p>
          {model.peft && <span className="inline-block px-2 py-1 bg-indigo-900/50 text-indigo-400 text-xs rounded">PEFT</span>}
        </div>
      ))}
      <div className="bg-gray-800/50 border-2 border-dashed border-gray-700 rounded-xl p-5 flex flex-col items-center justify-center cursor-pointer hover:border-indigo-500/50 transition min-h-[140px]">
        <Zap size={32} className="text-gray-500 mb-2" />
        <p className="text-gray-500">Add New Model</p>
      </div>
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
            <span className="font-medium">Hybrid</span>
          </div>
          <div className="flex justify-between p-3 bg-gray-900 rounded-lg">
            <span className="text-gray-400">Trim Ratio</span>
            <span className="font-medium">10%</span>
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
            <span className="text-green-400 font-medium">Enabled</span>
          </div>
          <div className="flex justify-between p-3 bg-gray-900 rounded-lg">
            <span className="text-gray-400">Epsilon (ε)</span>
            <span className="font-medium">1.45</span>
          </div>
        </div>
      </div>
    </div>
  );

  const renderSettings = () => (
    <div className="bg-gray-800 border border-gray-700 rounded-xl p-5">
      <h3 className="font-semibold mb-4">Experiment Controls</h3>
      <div className="flex gap-3 mb-6">
        {!experimentRunning ? (
          <button onClick={() => setExperimentRunning(true)} className="flex items-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg font-medium transition">
            <Play size={18} /> Start
          </button>
        ) : (
          <>
            <button onClick={() => setExperimentRunning(false)} className="flex items-center gap-2 px-6 py-3 bg-yellow-600 hover:bg-yellow-700 rounded-lg font-medium transition">
              <Pause size={18} /> Pause
            </button>
            <button className="flex items-center gap-2 px-6 py-3 bg-red-600 hover:bg-red-700 rounded-lg font-medium transition">
              <Square size={18} /> Stop
            </button>
          </>
        )}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-gray-400 text-sm mb-2">Clients</label>
          <input type="number" defaultValue={10} className="w-full bg-gray-900 border border-gray-700 rounded-lg p-3" />
        </div>
        <div>
          <label className="block text-gray-400 text-sm mb-2">Epochs</label>
          <input type="number" defaultValue={2} className="w-full bg-gray-900 border border-gray-700 rounded-lg p-3" />
        </div>
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
            <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-700 rounded-lg">
              <Wifi size={14} className={connected ? 'text-green-400' : 'text-red-400'} />
              <span className="text-sm text-gray-400">{connected ? 'Connected' : 'Offline'}</span>
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
        {activeTab === 'overview' && renderOverview()}
        {activeTab === 'clients' && renderClients()}
        {activeTab === 'models' && renderModels()}
        {activeTab === 'security' && renderSecurity()}
        {activeTab === 'settings' && renderSettings()}
      </main>
    </div>
  );
}
