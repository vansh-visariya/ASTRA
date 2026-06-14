'use client';

import React, { useState } from 'react';
import { Search, Plus, Cpu, Database } from 'lucide-react';
import type { Model } from '@/lib/api/types';

interface ModelSelectorProps {
  models: Model[];
  modelChoice: 'registry' | 'huggingface' | 'custom';
  selectedModelId: string;
  onSelectModel: (id: string) => void;
  onChoiceChange: (choice: 'registry' | 'huggingface' | 'custom') => void;
  onRegisterHf: (modelName: string, peftRank?: number) => Promise<void>;
  onRegisterCustom: (modelId: string, architecture: string, dataset: string) => Promise<void>;
  loading?: boolean;
}

export function ModelSelector({
  models,
  modelChoice,
  selectedModelId,
  onSelectModel,
  onChoiceChange,
  onRegisterHf,
  onRegisterCustom,
  loading,
}: ModelSelectorProps) {
  const [hfSearch, setHfSearch] = useState('');
  const [hfResults, setHfResults] = useState<Array<{ id: string }>>([]);
  const [searching, setSearching] = useState(false);

  // Custom model form
  const [customId, setCustomId] = useState('');
  const [architecture, setArchitecture] = useState('CNN');
  const [dataset, setDataset] = useState('MNIST');

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
        {(['registry', 'huggingface', 'custom'] as const).map((choice) => (
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
            {choice === 'custom' && 'Custom'}
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
                    <p className="text-slate-500 text-xs">{m.architecture || m.type} — {m.dataset || 'N/A'}</p>
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
              onClick={() => onRegisterHf(m.id)}
              className="w-full p-3 rounded-xl text-left hover:bg-white/5 transition-all flex items-center justify-between"
            >
              <span className="text-white text-sm">{m.id}</span>
              <Plus size={14} className="text-slate-600" />
            </button>
          ))}
        </div>
      )}

      {modelChoice === 'custom' && (
        <div className="space-y-3">
          <input
            value={customId}
            onChange={(e) => setCustomId(e.target.value)}
            placeholder="Model ID"
            className="input-field"
          />
          <select
            value={architecture}
            onChange={(e) => setArchitecture(e.target.value)}
            className="input-field"
          >
            <option value="CNN">CNN</option>
            <option value="MLP">MLP</option>
          </select>
          <select
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
            className="input-field"
          >
            <option value="MNIST">MNIST</option>
            <option value="CIFAR10">CIFAR10</option>
          </select>
          <button
            onClick={() => onRegisterCustom(customId, architecture, dataset)}
            disabled={!customId.trim()}
            className="btn-primary w-full inline-flex items-center justify-center gap-2 !py-3"
          >
            <Database size={16} />
            Register Custom Model
          </button>
        </div>
      )}
    </div>
  );
}
