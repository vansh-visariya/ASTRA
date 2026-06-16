'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/components/AuthContext';
import { Layers, ArrowLeft } from 'lucide-react';
import Link from 'next/link';
import { useWS } from '@/components/WebSocketProvider';
import { useModels } from '@/hooks';
import { createGroup, registerHfModel, registerArchitecture } from '@/lib/api/endpoints';
import { ModelSelector } from '@/components/create/ModelSelector';
import { TrainingConfig } from '@/components/create/TrainingConfig';
import { WindowConfig } from '@/components/create/WindowConfig';
import { Recommendations } from '@/components/create/Recommendations';
import type { Model } from '@/lib/api/types';

export default function CreateGroupPage() {
  const { token, user } = useAuth();
  const router = useRouter();
  const { isConnected } = useWS();
  const { data: modelsData, refetch: refetchModels } = useModels(!isConnected);

  const models: Model[] = (modelsData as any)?.models || [];

  const [modelChoice, setModelChoice] = useState<'registry' | 'huggingface' | 'external'>('registry');
  const [selectedModelId, setSelectedModelId] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [localEpochs, setLocalEpochs] = useState(2);
  const [batchSize, setBatchSize] = useState(32);
  const [learningRate, setLearningRate] = useState(0.01);
  const [dpEnabled, setDpEnabled] = useState(false);
  const [windowSize, setWindowSize] = useState(1);
  const [timeLimit, setTimeLimit] = useState(20);
  const [groupId, setGroupId] = useState('');
  const [aggregator, setAggregator] = useState('fedavg');

  const handleRegisterHf = async (modelName: string, peftRank?: number) => {
    const result = await registerHfModel({ model_name: modelName, use_peft: !!peftRank });
    setSelectedModelId(result.model_id);
    setModelChoice('registry');
    refetchModels();
  };

  const handleRegisterExternal = async (modelId: string, architecturePath: string, config?: Record<string, unknown>) => {
    await registerArchitecture({
      model_id: modelId,
      architecture_path: architecturePath,
      model_type: 'vision',
      config,
    });
    setSelectedModelId(modelId);
    setModelChoice('registry');
    refetchModels();
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    try {
      await createGroup({
        group_id: groupId,
        model_id: selectedModelId,
        window_size: windowSize,
        time_limit: timeLimit,
        local_epochs: localEpochs,
        batch_size: batchSize,
        lr: learningRate,
        dp_enabled: dpEnabled,
        aggregator,
      });
      router.push('/dashboard/groups');
    } catch {
      setSubmitting(false);
    }
  };

  if (!user || user.role !== 'admin') {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-slate-400">Access denied. Admins only.</p>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="flex items-center gap-4">
        <Link href="/dashboard/groups" className="p-2 hover:bg-gray-800 rounded-lg transition">
          <ArrowLeft size={20} className="text-gray-400" />
        </Link>
        <div>
          <h1 className="text-2xl font-bold text-white">Create Group</h1>
          <p className="text-slate-400 text-sm mt-1">Configure a new federated learning experiment</p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="glass-card p-5">
          <h2 className="text-sm font-semibold text-white mb-4 uppercase tracking-wider flex items-center gap-2">
            <Layers size={16} className="text-slate-400" /> Basic Information
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-slate-400 text-xs font-medium block mb-1.5">Group ID</label>
              <input
                type="text"
                value={groupId}
                onChange={(e) => setGroupId(e.target.value)}
                className="input-field"
                placeholder="e.g., experiment_001"
                required
              />
            </div>
            <div>
              <label className="text-slate-400 text-xs font-medium block mb-1.5">Aggregator</label>
              <select
                value={aggregator}
                onChange={(e) => setAggregator(e.target.value)}
                className="input-field"
              >
                <option value="fedavg">FedAvg</option>
                <option value="robust">Robust Aggregation</option>
                <option value="trimmed_mean">Trimmed Mean</option>
                <option value="median">Coordinate Median</option>
              </select>
            </div>
          </div>
        </div>

        <Recommendations onModelRegistered={refetchModels} />

        <ModelSelector
          models={models}
          modelChoice={modelChoice}
          selectedModelId={selectedModelId}
          onSelectModel={setSelectedModelId}
          onChoiceChange={setModelChoice}
          onRegisterHf={handleRegisterHf}
          onRegisterExternal={handleRegisterExternal}
        />

        <TrainingConfig
          localEpochs={localEpochs}
          batchSize={batchSize}
          learningRate={learningRate}
          dpEnabled={dpEnabled}
          onChange={(field, value) => {
            switch (field) {
              case 'local_epochs': setLocalEpochs(value as number); break;
              case 'batch_size': setBatchSize(value as number); break;
              case 'lr': setLearningRate(value as number); break;
              case 'dp_enabled': setDpEnabled(value as boolean); break;
            }
          }}
        />

        <WindowConfig
          windowSize={windowSize}
          timeLimit={timeLimit}
          onChange={(field, value) => {
            switch (field) {
              case 'window_size': setWindowSize(value); break;
              case 'time_limit': setTimeLimit(value); break;
            }
          }}
        />

        <button
          type="submit"
          disabled={submitting || !groupId.trim()}
          className="btn-primary w-full !py-4 font-semibold text-base disabled:opacity-50"
        >
          {submitting ? 'Creating...' : 'Create Federated Learning Group'}
        </button>
      </form>
    </div>
  );
}
