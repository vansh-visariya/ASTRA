'use client';

import React, { useState } from 'react';
import { Search, Plus, Cpu, Box } from 'lucide-react';
import type { Model } from '@/lib/api/types';

interface ModelSelectorProps {
  models: Model[];
  modelChoice: 'registry' | 'huggingface' | 'external';
  selectedModelId: string;
  onSelectModel: (id: string) => void;
  onChoiceChange: (choice: 'registry' | 'huggingface' | 'external') => void;
  onRegisterHf: (modelName: string, peftRank?: number) => Promise<void>;
  onRegisterExternal: (modelId: string, architecturePath: string, config?: Record<string, unknown>) => Promise<void>;
  loading?: boolean;
}

export function ModelSelector({
  models,
  modelChoice,
  selectedModelId,
  onSelectModel,
  onChoiceChange,
  onRegisterHf,
  onRegisterExternal,
  loading,
}: ModelSelectorProps) {
  const [hfSearch, setHfSearch] = useState('');
  const [hfResults, setHfResults] = useState<Array<{ id: string }>>([]);
  const [searching, setSearching] = useState(false);
  const [usePeft, setUsePeft] = useState(false);

  const [extId, setExtId] = useState('');
  const [extPath, setExtPath] = useState('');

  const handleHfSearch = async () => {
    if (!hfSearch.trim()) return;
    setSearching(true);
    try {
      const res = await fetch(`https://huggingface.co/api/models?search=${encodeURIComponent(hfSearch)}&limit=10`);
      if (res.ok) setHfResults(await res.json());
    } catch {
      // ignore
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="glass-card p-5 space-y-4">
      <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Model Configuration</h3>

      <div className="flex gap-2 mb-4">
        {(['registry', 'huggingface', 'external'] as const).map((choice) => (
          <button
            key={choice}
            onClick={() => onChoiceChange(choice)}
            className={`px-4 py-2 rounded-xl text-xs font-medium transition-all ${
              modelChoice === choice
                ? 'bg-white/10 text-white border border-white/20'
                : 'text-slate-500 hover:text-white hover:bg-white/5'
            }`}
          >
            {choice === 'registry' && 'Registry'}
            {choice === 'huggingface' && 'HuggingFace'}
            {choice === 'external' && 'External'}
          </button>
        ))}
      </div>

      {modelChoice === 'registry' && (
        <div className="space-y-2">
          {loading ? (
            <p className="text-slate-500 text-xs">Loading models...</p>
          ) : models.length === 0 ? (
            <p className="text-slate-500 text-xs">No models registered yet.</p>
          ) : (
            models.map((m) => (
              <button
                key={m.model_id}
                onClick={() => onSelectModel(m.model_id)}
                className={`w-full p-3 rounded-xl text-left transition-all ${
                  selectedModelId === m.model_id
                    ? 'bg-white/10 border border-white/20'
                    : 'hover:bg-white/5 border border-transparent'
                }`}
              >
                <div className="flex items-center gap-3">
                  <Cpu size={16} className="text-slate-500" />
                  <div>
                    <p className="text-white text-sm font-medium">{m.model_id}</p>
                    <p className="text-slate-500 text-xs">{m.architecture || m.model_type} — {m.dataset || 'N/A'}</p>
                  </div>
                </div>
              </button>
            ))
          )}
        </div>
      )}

      {modelChoice === 'huggingface' && (
        <div className="space-y-3">
          <div className="flex gap-2">
            <input
              value={hfSearch}
              onChange={(e) => setHfSearch(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleHfSearch()}
              placeholder="Search HuggingFace models..."
              className="input-field flex-1"
            />
            <button onClick={handleHfSearch} className="btn-secondary !px-3" disabled={searching}>
              <Search size={16} />
            </button>
          </div>
          {hfResults.slice(0, 5).map((m) => (
            <button
              key={m.id}
              onClick={() => onRegisterHf(m.id, usePeft ? 8 : undefined)}
              className="w-full p-3 rounded-xl text-left hover:bg-white/5 transition-all flex items-center justify-between"
            >
              <span className="text-white text-sm">{m.id}</span>
              <Plus size={14} className="text-slate-600" />
            </button>
          ))}
          <label className="flex items-center gap-2 pt-2 text-xs text-slate-400 cursor-pointer">
            <input
              type="checkbox"
              checked={usePeft}
              onChange={(e) => setUsePeft(e.target.checked)}
              className="rounded"
            />
            Use LoRA/PEFT (train only adapters — faster, less bandwidth)
          </label>
        </div>
      )}

      {modelChoice === 'external' && (
        <div className="space-y-3">
          <div className="p-3 rounded-xl" style={{ background: 'rgba(30,41,59,0.4)' }}>
            <p className="text-slate-400 text-xs">Register any PyTorch model by its Python import path. Examples:</p>
            <ul className="text-slate-500 text-[11px] mt-1.5 space-y-0.5 font-mono">
              <li>torchvision.models.resnet18</li>
              <li>torchvision.models.efficientnet_b0</li>
              <li>torchvision.models.vit_b_16</li>
              <li>transformers.AutoModelForImageClassification</li>
            </ul>
          </div>
          <input
            value={extId}
            onChange={(e) => setExtId(e.target.value)}
            placeholder="Model ID (e.g., resnet18_v1)"
            className="input-field"
          />
          <input
            value={extPath}
            onChange={(e) => setExtPath(e.target.value)}
            placeholder="Python import path (e.g., torchvision.models.resnet18)"
            className="input-field font-mono text-xs"
          />
          <button
            onClick={() => onRegisterExternal(extId, extPath, {})}
            disabled={!extId.trim() || !extPath.trim()}
            className="btn-primary w-full inline-flex items-center justify-center gap-2 !py-3"
          >
            <Box size={16} />
            Register External Model
          </button>
        </div>
      )}
    </div>
  );
}
