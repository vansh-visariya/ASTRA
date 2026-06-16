'use client';

import React, { useState } from 'react';
import { Sparkles, Cpu, ExternalLink } from 'lucide-react';
import { getRecommendations, addHuggingFaceModel } from '@/lib/api/endpoints';
import type { Recommendation } from '@/lib/api/types';

interface RecommendationsProps {
  onModelRegistered: () => void;
}

export function Recommendations({ onModelRegistered }: RecommendationsProps) {
  const [metadata, setMetadata] = useState({
    dataset_size: '',
    num_classes: '',
    has_gpu: false,
    cpu_cores: '',
  });
  const [recs, setRecs] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [hfUrl, setHfUrl] = useState('');
  const [addingHf, setAddingHf] = useState(false);

  const fetchRecommendations = async () => {
    setLoading(true);
    try {
      const result = await getRecommendations({
        dataset_size: metadata.dataset_size ? parseInt(metadata.dataset_size) : undefined,
        num_classes: metadata.num_classes ? parseInt(metadata.num_classes) : undefined,
        has_gpu: metadata.has_gpu,
        cpu_cores: metadata.cpu_cores ? parseInt(metadata.cpu_cores) : undefined,
      });
      setRecs(result.recommendations || []);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  };

  const handleAddHf = async () => {
    if (!hfUrl.trim()) return;
    setAddingHf(true);
    try {
      await addHuggingFaceModel({ model_url: hfUrl, use_peft: false });
      onModelRegistered();
      setHfUrl('');
    } catch {
      // ignore
    } finally {
      setAddingHf(false);
    }
  };

  return (
    <div className="glass-card p-5 space-y-4">
      <div className="flex items-center gap-2">
        <Sparkles size={16} className="text-slate-500" />
        <h3 className="text-sm font-semibold text-white uppercase tracking-wider">AI Recommendations</h3>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <input
          value={metadata.dataset_size}
          onChange={(e) => setMetadata({ ...metadata, dataset_size: e.target.value })}
          placeholder="Dataset size"
          type="number"
          className="input-field"
        />
        <input
          value={metadata.num_classes}
          onChange={(e) => setMetadata({ ...metadata, num_classes: e.target.value })}
          placeholder="Num classes"
          type="number"
          className="input-field"
        />
        <label className="flex items-center gap-2 text-slate-400 text-xs">
          <input
            type="checkbox"
            checked={metadata.has_gpu}
            onChange={(e) => setMetadata({ ...metadata, has_gpu: e.target.checked })}
            className="w-4 h-4 rounded accent-white"
          />
          Has GPU
        </label>
        <input
          value={metadata.cpu_cores}
          onChange={(e) => setMetadata({ ...metadata, cpu_cores: e.target.value })}
          placeholder="CPU cores"
          type="number"
          className="input-field"
        />
      </div>

      <button onClick={fetchRecommendations} disabled={loading} className="btn-secondary w-full inline-flex items-center justify-center gap-2 !py-2.5">
        <Sparkles size={14} />
        {loading ? 'Analyzing...' : 'Get Recommendations'}
      </button>

      {recs.length > 0 && (
        <div className="space-y-2 mt-3">
          {recs.map((rec) => (
            <div key={rec.model_id} className="p-3 rounded-xl" style={{ background: 'rgba(15, 15, 15, 0.6)', border: '1px solid rgba(100, 100, 100, 0.15)' }}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-white text-sm font-medium">{rec.model_id}</span>
                <span className="text-slate-500 text-[10px] uppercase">{rec.source}</span>
              </div>
              <p className="text-slate-500 text-xs">{rec.reasoning}</p>
            </div>
          ))}
        </div>
      )}

      <div className="pt-3 border-t border-white/5">
        <p className="text-slate-500 text-xs mb-2">Quick add from HuggingFace:</p>
        <div className="flex gap-2">
          <input
            value={hfUrl}
            onChange={(e) => setHfUrl(e.target.value)}
            placeholder="https://huggingface.co/openai/clip-vit-base-patch32"
            className="input-field flex-1"
          />
          <button onClick={handleAddHf} disabled={addingHf || !hfUrl.trim()} className="btn-secondary !px-3 inline-flex items-center gap-1">
            <ExternalLink size={14} /> Add
          </button>
        </div>
      </div>
    </div>
  );
}
