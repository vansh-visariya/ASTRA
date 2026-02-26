'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/components/AuthContext';
import { Brain, Sparkles, Loader2, ExternalLink, Plus, Check } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Recommendation {
  id: string;
  model_type: string;
  model_size: string;
  estimated_params: number;
  expected_accuracy: number;
  reasoning: string;
  source: string;
  model_id: string;
  model_name: string;
}

interface ClientMetadata {
  dataset_size: number;
  num_classes: number;
  has_gpu: boolean;
  gpu_memory_mb?: number;
  cpu_cores?: number;
}

export default function ModelRecommender() {
  const { token } = useAuth();
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedRec, setSelectedRec] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<ClientMetadata>({
    dataset_size: 5000,
    num_classes: 10,
    has_gpu: false,
    cpu_cores: 4,
  });
  const [showHfForm, setShowHfForm] = useState(false);
  const [hfUrl, setHfUrl] = useState('');
  const [addingHf, setAddingHf] = useState(false);

  const fetchRecommendations = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/api/recommendations/unified`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(metadata)
      });
      if (res.ok) {
        const data = await res.json();
        setRecommendations(data.recommendations || []);
      }
    } catch (e) {
      console.error('Failed to fetch recommendations:', e);
    }
    setLoading(false);
  };

  const handleAddHuggingFace = async () => {
    if (!hfUrl.trim()) return;
    setAddingHf(true);
    try {
      const res = await fetch(`${API_URL}/api/recommendations/add-huggingface`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ model_url: hfUrl })
      });
      if (res.ok) {
        const data = await res.json();
        if (data.success) {
          setSelectedRec(data.model.model_id);
          setShowHfForm(false);
          setHfUrl('');
          fetchRecommendations();
        }
      }
    } catch (e) {
      console.error('Failed to add HF model:', e);
    }
    setAddingHf(false);
  };

  const formatParams = (num: number) => {
    if (num >= 1e6) return `${(num / 1e6).toFixed(1)}M`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(1)}K`;
    return num.toString();
  };

  const getSourceBadge = (source: string) => {
    switch (source) {
      case 'gemini':
        return <span className="text-xs px-2 py-0.5 rounded bg-purple-900/50 text-purple-400">AI</span>;
      case 'builtin':
        return <span className="text-xs px-2 py-0.5 rounded bg-blue-900/50 text-blue-400">Built-in</span>;
      case 'huggingface':
        return <span className="text-xs px-2 py-0.5 rounded bg-yellow-900/50 text-yellow-400">HF</span>;
      default:
        return null;
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Model Recommendations</h1>
        <p className="text-gray-400">Get AI-powered model recommendations based on your hardware</p>
      </div>

      {/* Metadata Input */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Your Device Profile</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-gray-400 text-sm mb-2">Dataset Size</label>
            <input
              type="number"
              value={metadata.dataset_size}
              onChange={(e) => setMetadata({ ...metadata, dataset_size: parseInt(e.target.value) })}
              className="w-full bg-gray-950 border border-gray-800 rounded-lg py-2 px-3 text-white"
            />
          </div>
          <div>
            <label className="block text-gray-400 text-sm mb-2">Num Classes</label>
            <input
              type="number"
              value={metadata.num_classes}
              onChange={(e) => setMetadata({ ...metadata, num_classes: parseInt(e.target.value) })}
              className="w-full bg-gray-950 border border-gray-800 rounded-lg py-2 px-3 text-white"
            />
          </div>
          <div>
            <label className="block text-gray-400 text-sm mb-2">Has GPU</label>
            <button
              type="button"
              onClick={() => setMetadata({ ...metadata, has_gpu: !metadata.has_gpu })}
              className={`w-full py-2 rounded-lg border transition ${
                metadata.has_gpu ? 'bg-green-900/30 border-green-600 text-green-400' : 'bg-gray-950 border-gray-800 text-gray-400'
              }`}
            >
              {metadata.has_gpu ? 'Yes' : 'No'}
            </button>
          </div>
          <div>
            <label className="block text-gray-400 text-sm mb-2">CPU Cores</label>
            <input
              type="number"
              value={metadata.cpu_cores || ''}
              onChange={(e) => setMetadata({ ...metadata, cpu_cores: parseInt(e.target.value) || undefined })}
              className="w-full bg-gray-950 border border-gray-800 rounded-lg py-2 px-3 text-white"
              placeholder="e.g., 4"
            />
          </div>
        </div>
        
        <button
          onClick={fetchRecommendations}
          disabled={loading}
          className="mt-4 px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:opacity-50 rounded-lg text-white flex items-center gap-2"
        >
          {loading ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
          Get Recommendations
        </button>
      </div>

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Brain size={20} className="text-purple-400" />
            Recommended Models
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {recommendations.map((rec) => (
              <div
                key={rec.id}
                className={`p-4 rounded-lg border ${
                  selectedRec === rec.model_id
                    ? 'border-purple-500 bg-purple-900/20'
                    : 'border-gray-800 bg-gray-950'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  {getSourceBadge(rec.source)}
                  <span className="text-xs text-gray-500">{rec.model_size}</span>
                </div>
                <p className="text-white font-medium">{rec.model_type}</p>
                <p className="text-gray-500 text-sm">{formatParams(rec.estimated_params)} params</p>
                <p className="text-purple-400 text-sm mt-1">Expected acc: {(rec.expected_accuracy * 100).toFixed(0)}%</p>
                <p className="text-gray-600 text-xs mt-2 line-clamp-2">{rec.reasoning}</p>
                
                {selectedRec === rec.model_id && (
                  <div className="mt-3 flex items-center gap-1 text-green-400 text-sm">
                    <Check size={14} /> Selected
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Add HuggingFace Model */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Add Custom Model</h2>
        
        {!showHfForm ? (
          <button
            onClick={() => setShowHfForm(true)}
            className="text-yellow-400 hover:text-yellow-300 flex items-center gap-1"
          >
            <ExternalLink size={14} />
            Add from HuggingFace
          </button>
        ) : (
          <div className="flex gap-2">
            <input
              type="text"
              value={hfUrl}
              onChange={(e) => setHfUrl(e.target.value)}
              placeholder="e.g., facebook/resnet-50"
              className="flex-1 bg-gray-950 border border-gray-800 rounded-lg py-2 px-3 text-white"
            />
            <button
              onClick={handleAddHuggingFace}
              disabled={addingHf || !hfUrl.trim()}
              className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 disabled:opacity-50 rounded-lg text-white flex items-center gap-2"
            >
              {addingHf ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />}
              Add
            </button>
            <button
              onClick={() => { setShowHfForm(false); setHfUrl(''); }}
              className="px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-gray-400"
            >
              Cancel
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
