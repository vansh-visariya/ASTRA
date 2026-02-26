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
  const [modelChoice, setModelChoice] = useState<'registry' | 'huggingface' | 'custom'>('registry');
  const [hfQuery, setHfQuery] = useState('');
  const [hfResults, setHfResults] = useState<string[]>([]);
  const [hfModelName, setHfModelName] = useState('');
  const [hfUsePeft, setHfUsePeft] = useState(false);
  const [hfPeftMethod, setHfPeftMethod] = useState('lora');
  const [customModelId, setCustomModelId] = useState('');
  const [customArchitecture, setCustomArchitecture] = useState('cnn');
  const [customModelType, setCustomModelType] = useState('vision');
  const [customDataset, setCustomDataset] = useState('MNIST');
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

  const searchHuggingFace = async () => {
    if (!hfQuery.trim()) {
      setHfResults([]);
      return;
    }
    try {
      const res = await fetch(`https://huggingface.co/api/models?search=${encodeURIComponent(hfQuery)}&limit=5`);
      if (res.ok) {
        const data = await res.json();
        const names = (data || []).map((item: { modelId?: string }) => item.modelId).filter(Boolean);
        setHfResults(names);
      }
    } catch (e) {
      console.error('Failed to search HuggingFace:', e);
    }
  };

  const registerHfModel = async (): Promise<string> => {
    const params = new URLSearchParams({
      model_name: hfModelName,
      use_peft: hfUsePeft ? 'true' : 'false',
      peft_method: hfPeftMethod
    });
    const res = await fetch(`${API_URL}/api/models/register/hf?${params.toString()}`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` }
    });
    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || 'Failed to register HuggingFace model');
    }
    const data = await res.json();
    return data.model?.model_id || data.model_id;
  };

  const registerCustomModel = async (): Promise<string> => {
    const params = new URLSearchParams({
      model_id: customModelId,
      model_type: customModelType,
      model_source: 'custom'
    });
    const config = {
      model: {
        type: customArchitecture,
        cnn: { name: 'simple_cnn' }
      },
      dataset: {
        name: customDataset
      }
    };
    const res = await fetch(`${API_URL}/api/models/register?${params.toString()}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify(config)
    });
    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || 'Failed to register custom model');
    }
    const data = await res.json();
    return data.model?.model_id || customModelId;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      let selectedModelId = form.model_id;
      if (modelChoice === 'huggingface') {
        selectedModelId = await registerHfModel();
      }
      if (modelChoice === 'custom') {
        selectedModelId = await registerCustomModel();
      }
      const res = await fetch(`${API_URL}/api/groups`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ ...form, model_id: selectedModelId })
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
            <button
              type="button"
              onClick={() => setModelChoice('huggingface')}
              className={`p-4 rounded-lg border text-left transition ${
                modelChoice === 'huggingface'
                  ? 'border-indigo-500 bg-indigo-900/20'
                  : 'border-gray-800 hover:border-gray-700 bg-gray-950'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs px-2 py-0.5 rounded bg-purple-900/50 text-purple-400">huggingface</span>
              </div>
              <p className="text-white font-medium text-sm">Search HuggingFace</p>
              <p className="text-gray-500 text-xs mt-1">Register a model by name</p>
            </button>

            <button
              type="button"
              onClick={() => setModelChoice('custom')}
              className={`p-4 rounded-lg border text-left transition ${
                modelChoice === 'custom'
                  ? 'border-indigo-500 bg-indigo-900/20'
                  : 'border-gray-800 hover:border-gray-700 bg-gray-950'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs px-2 py-0.5 rounded bg-green-900/50 text-green-400">custom</span>
              </div>
              <p className="text-white font-medium text-sm">Create From Scratch</p>
              <p className="text-gray-500 text-xs mt-1">Define a custom architecture</p>
            </button>

            {models.map((model) => (
              <button
                key={model.model_id}
                type="button"
                onClick={() => {
                  setModelChoice('registry');
                  setForm({ ...form, model_id: model.model_id });
                }}
                className={`p-4 rounded-lg border text-left transition ${
                  modelChoice === 'registry' && form.model_id === model.model_id
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

          {modelChoice === 'huggingface' && (
            <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-gray-400 text-sm mb-2">Search HuggingFace</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={hfQuery}
                    onChange={(e) => setHfQuery(e.target.value)}
                    className="w-full bg-gray-950 border border-gray-800 rounded-lg py-3 px-4 text-white focus:outline-none focus:border-indigo-500"
                    placeholder="e.g., openai/clip-vit-base-patch32"
                  />
                  <button
                    type="button"
                    onClick={searchHuggingFace}
                    className="px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded-lg text-white text-sm"
                  >
                    Search
                  </button>
                </div>
                {hfResults.length > 0 && (
                  <div className="mt-2 border border-gray-800 rounded-lg bg-gray-950">
                    {hfResults.map((name) => (
                      <button
                        key={name}
                        type="button"
                        onClick={() => setHfModelName(name)}
                        className={`w-full text-left px-3 py-2 text-sm border-b border-gray-800 last:border-b-0 transition ${
                          hfModelName === name ? 'bg-indigo-900/30 text-indigo-200' : 'text-gray-300 hover:bg-gray-900'
                        }`}
                      >
                        {name}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <div>
                <label className="block text-gray-400 text-sm mb-2">Selected Model</label>
                <input
                  type="text"
                  value={hfModelName}
                  onChange={(e) => setHfModelName(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-lg py-3 px-4 text-white focus:outline-none focus:border-indigo-500"
                  placeholder="Select from search or paste model name"
                />
                <div className="mt-3 flex items-center gap-3">
                  <button
                    type="button"
                    onClick={() => setHfUsePeft(!hfUsePeft)}
                    className={`px-3 py-2 rounded-lg border text-sm transition ${
                      hfUsePeft
                        ? 'bg-yellow-900/20 border-yellow-600 text-yellow-300'
                        : 'bg-gray-950 border-gray-800 text-gray-400'
                    }`}
                  >
                    PEFT: {hfUsePeft ? 'Enabled' : 'Disabled'}
                  </button>
                  <select
                    value={hfPeftMethod}
                    onChange={(e) => setHfPeftMethod(e.target.value)}
                    className="bg-gray-950 border border-gray-800 rounded-lg py-2 px-3 text-white text-sm"
                    disabled={!hfUsePeft}
                  >
                    <option value="lora">LoRA</option>
                  </select>
                </div>
              </div>
            </div>
          )}

          {modelChoice === 'custom' && (
            <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-gray-400 text-sm mb-2">Custom Model ID</label>
                <input
                  type="text"
                  value={customModelId}
                  onChange={(e) => setCustomModelId(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-lg py-3 px-4 text-white focus:outline-none focus:border-indigo-500"
                  placeholder="e.g., custom_cnn_v1"
                />
              </div>
              <div>
                <label className="block text-gray-400 text-sm mb-2">Architecture</label>
                <select
                  value={customArchitecture}
                  onChange={(e) => setCustomArchitecture(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-lg py-3 px-4 text-white focus:outline-none focus:border-indigo-500"
                >
                  <option value="cnn">CNN</option>
                  <option value="mlp">MLP</option>
                </select>
              </div>
              <div>
                <label className="block text-gray-400 text-sm mb-2">Model Type</label>
                <select
                  value={customModelType}
                  onChange={(e) => setCustomModelType(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-lg py-3 px-4 text-white focus:outline-none focus:border-indigo-500"
                >
                  <option value="vision">Vision</option>
                  <option value="text">Text</option>
                </select>
              </div>
              <div>
                <label className="block text-gray-400 text-sm mb-2">Dataset</label>
                <select
                  value={customDataset}
                  onChange={(e) => setCustomDataset(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-lg py-3 px-4 text-white focus:outline-none focus:border-indigo-500"
                >
                  <option value="MNIST">MNIST</option>
                  <option value="CIFAR10">CIFAR10</option>
                </select>
              </div>
            </div>
          )}
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
            disabled={
              loading ||
              !form.group_id ||
              (modelChoice === 'registry' && !form.model_id) ||
              (modelChoice === 'huggingface' && !hfModelName) ||
              (modelChoice === 'custom' && !customModelId)
            }
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
