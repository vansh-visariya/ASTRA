'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/components/AuthContext';
import { Layers, Brain, Clock, Shield, Zap, ArrowLeft, Plus } from 'lucide-react';
import Link from 'next/link';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Model {
  model_id: string;
  model_type: string;
  architecture: string;
  total_params: number;
  is_peft: boolean;
  source: string;
}

export default function CreateGroupPage() {
  const { token, user } = useAuth();
  const router = useRouter();
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(false);
  const [form, setForm] = useState({
    group_id: '',
    model_id: 'simple_cnn_mnist',
    window_size: 1,
    time_limit: 20,
    local_epochs: 2,
    batch_size: 32,
    lr: 0.01,
    dp_enabled: false,
    aggregator: 'fedavg',
  });

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await fetch(`${API_URL}/api/models`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        if (res.ok) {
          const data = await res.json();
          setModels(data.models || []);
        }
      } catch (e) {
        console.error('Failed to fetch models:', e);
      }
    };
    fetchModels();
  }, [token]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const res = await fetch(`${API_URL}/api/groups`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(form)
      });
      
      if (res.ok) {
        router.push('/dashboard/groups');
      } else {
        const error = await res.json();
        alert(error.detail || 'Failed to create group');
      }
    } catch (e) {
      console.error('Failed to create group:', e);
    }
    setLoading(false);
  };

  const formatParams = (num: number) => {
    if (num >= 1e9) return `${(num / 1e9).toFixed(1)}B`;
    if (num >= 1e6) return `${(num / 1e6).toFixed(1)}M`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(1)}K`;
    return num.toString();
  };

  if (user?.role !== 'coordinator') {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-gray-400">Access denied. Coordinators only.</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="flex items-center gap-4">
        <Link href="/dashboard/groups" className="p-2 hover:bg-gray-800 rounded-lg transition">
          <ArrowLeft size={20} className="text-gray-400" />
        </Link>
        <div>
          <h1 className="text-2xl font-bold text-white">Create Group</h1>
          <p className="text-gray-400">Configure a new federated learning experiment</p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Layers size={20} className="text-indigo-400" />
            Basic Information
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-gray-400 text-sm mb-2">Group ID</label>
              <input
                type="text"
                value={form.group_id}
                onChange={(e) => setForm({ ...form, group_id: e.target.value })}
                className="w-full bg-gray-950 border border-gray-800 rounded-lg py-3 px-4 text-white focus:outline-none focus:border-indigo-500"
                placeholder="e.g., experiment_001"
                required
              />
            </div>
            <div>
              <label className="block text-gray-400 text-sm mb-2">Aggregator</label>
              <select
                value={form.aggregator}
                onChange={(e) => setForm({ ...form, aggregator: e.target.value })}
                className="w-full bg-gray-950 border border-gray-800 rounded-lg py-3 px-4 text-white focus:outline-none focus:border-indigo-500"
              >
                <option value="fedavg">FedAvg</option>
                <option value="robust">Robust Aggregation</option>
                <option value="trimmed_mean">Trimmed Mean</option>
                <option value="median">Coordinate Median</option>
              </select>
            </div>
          </div>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Brain size={20} className="text-indigo-400" />
            Model Configuration
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {models.map((model) => (
              <button
                key={model.model_id}
                type="button"
                onClick={() => setForm({ ...form, model_id: model.model_id })}
                className={`p-4 rounded-lg border text-left transition ${
                  form.model_id === model.model_id
                    ? 'border-indigo-500 bg-indigo-900/20'
                    : 'border-gray-800 hover:border-gray-700 bg-gray-950'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className={`text-xs px-2 py-0.5 rounded ${
                    model.model_type === 'vision' ? 'bg-blue-900/50 text-blue-400' :
                    model.model_type === 'text' ? 'bg-purple-900/50 text-purple-400' :
                    'bg-green-900/50 text-green-400'
                  }`}>
                    {model.model_type}
                  </span>
                  {model.is_peft && <span className="text-xs text-yellow-400">PEFT</span>}
                </div>
                <p className="text-white font-medium text-sm truncate">{model.architecture}</p>
                <p className="text-gray-500 text-xs mt-1">{formatParams(model.total_params)} params</p>
              </button>
            ))}
          </div>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Zap size={20} className="text-indigo-400" />
            Training Configuration
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-gray-400 text-sm mb-2">Local Epochs</label>
              <input
                type="number"
                value={form.local_epochs}
                onChange={(e) => setForm({ ...form, local_epochs: parseInt(e.target.value) })}
                className="w-full bg-gray-950 border border-gray-800 rounded-lg py-3 px-4 text-white focus:outline-none focus:border-indigo-500"
                min="1"
              />
            </div>
            <div>
              <label className="block text-gray-400 text-sm mb-2">Batch Size</label>
              <input
                type="number"
                value={form.batch_size}
                onChange={(e) => setForm({ ...form, batch_size: parseInt(e.target.value) })}
                className="w-full bg-gray-950 border border-gray-800 rounded-lg py-3 px-4 text-white focus:outline-none focus:border-indigo-500"
                min="1"
              />
            </div>
            <div>
              <label className="block text-gray-400 text-sm mb-2">Learning Rate</label>
              <input
                type="number"
                step="0.001"
                value={form.lr}
                onChange={(e) => setForm({ ...form, lr: parseFloat(e.target.value) })}
                className="w-full bg-gray-950 border border-gray-800 rounded-lg py-3 px-4 text-white focus:outline-none focus:border-indigo-500"
                min="0"
              />
            </div>
            <div>
              <label className="block text-gray-400 text-sm mb-2">DP Enabled</label>
              <button
                type="button"
                onClick={() => setForm({ ...form, dp_enabled: !form.dp_enabled })}
                className={`w-full py-3 rounded-lg border transition ${
                  form.dp_enabled
                    ? 'bg-green-900/20 border-green-500 text-green-400'
                    : 'bg-gray-950 border-gray-800 text-gray-400'
                }`}
              >
                {form.dp_enabled ? 'Enabled' : 'Disabled'}
              </button>
            </div>
          </div>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Clock size={20} className="text-indigo-400" />
            Async Window Configuration
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-gray-400 text-sm mb-2">Window Size (N updates)</label>
              <input
                type="number"
                value={form.window_size}
                onChange={(e) => setForm({ ...form, window_size: parseInt(e.target.value) })}
                className="w-full bg-gray-950 border border-gray-800 rounded-lg py-3 px-4 text-white focus:outline-none focus:border-indigo-500"
                min="1"
              />
              <p className="text-gray-500 text-xs mt-1">Aggregate when N updates received</p>
            </div>
            <div>
              <label className="block text-gray-400 text-sm mb-2">Time Limit (seconds)</label>
              <input
                type="number"
                value={form.time_limit}
                onChange={(e) => setForm({ ...form, time_limit: parseInt(e.target.value) })}
                className="w-full bg-gray-950 border border-gray-800 rounded-lg py-3 px-4 text-white focus:outline-none focus:border-indigo-500"
                min="1"
              />
              <p className="text-gray-500 text-xs mt-1">OR after T seconds elapsed</p>
            </div>
          </div>
        </div>

        <div className="flex justify-end gap-4">
          <Link href="/dashboard/groups" className="px-6 py-3 bg-gray-800 hover:bg-gray-700 rounded-lg text-white font-medium transition">
            Cancel
          </Link>
          <button
            type="submit"
            disabled={loading || !form.group_id}
            className="px-6 py-3 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 rounded-lg text-white font-medium transition flex items-center gap-2"
          >
            {loading ? (
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <>
                <Plus size={18} />
                Create Group
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
