'use client';

import React from 'react';

interface TrainingConfigProps {
  learningRate: number;
  dpEnabled: boolean;
  localEpochs: number;
  batchSize: number;
  valDataset: string;
  expectedDeltaBytes: number;
  onChange: (field: string, value: number | boolean | string) => void;
}

export function TrainingConfig({
  learningRate,
  dpEnabled,
  localEpochs,
  batchSize,
  valDataset,
  expectedDeltaBytes,
  onChange,
}: TrainingConfigProps) {
  return (
    <div className="space-y-4">
      <div className="glass-card p-5 space-y-4">
        <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Server Configuration</h3>
        <p className="text-slate-500 text-xs">
          Clients train externally on their own data. The server only applies these settings when aggregating received deltas.
        </p>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-slate-400 text-xs font-medium block mb-1.5">Server Learning Rate</label>
            <input
              type="number"
              step="0.001"
              value={learningRate}
              onChange={(e) => onChange('lr', parseFloat(e.target.value) || 0.01)}
              className="input-field"
              min={0.0001}
              max={1}
            />
          </div>
          <div className="flex items-end">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={dpEnabled}
                onChange={(e) => onChange('dp_enabled', e.target.checked)}
                className="w-4 h-4 rounded accent-white"
              />
              <span className="text-slate-400 text-xs font-medium">Enable Differential Privacy (server-side)</span>
            </label>
          </div>
        </div>
      </div>

      <div className="glass-card p-5 space-y-4">
        <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Training Contract</h3>
        <p className="text-slate-500 text-xs">
          These parameters are sent to clients as a training contract. Clients see them before training and should follow them.
        </p>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-slate-400 text-xs font-medium block mb-1.5">Local Epochs</label>
            <input
              type="number"
              value={localEpochs}
              onChange={(e) => onChange('local_epochs', parseInt(e.target.value) || 2)}
              className="input-field"
              min={1}
              max={100}
            />
            <p className="text-slate-600 text-[10px] mt-1">Recommended epochs per client per round</p>
          </div>
          <div>
            <label className="text-slate-400 text-xs font-medium block mb-1.5">Batch Size</label>
            <input
              type="number"
              value={batchSize}
              onChange={(e) => onChange('batch_size', parseInt(e.target.value) || 32)}
              className="input-field"
              min={1}
              max={4096}
            />
            <p className="text-slate-600 text-[10px] mt-1">Recommended batch size for training</p>
          </div>
          <div>
            <label className="text-slate-400 text-xs font-medium block mb-1.5">Validation Dataset</label>
            <select
              value={valDataset}
              onChange={(e) => onChange('val_dataset', e.target.value)}
              className="input-field"
            >
              <option value="">None (no server evaluation)</option>
              <option value="mnist">MNIST</option>
              <option value="cifar10">CIFAR-10</option>
            </select>
            <p className="text-slate-600 text-[10px] mt-1">Server evaluates model on this dataset after each round</p>
          </div>
          <div>
            <label className="text-slate-400 text-xs font-medium block mb-1.5">Expected Delta Size (bytes)</label>
            <input
              type="number"
              value={expectedDeltaBytes}
              onChange={(e) => onChange('expected_delta_bytes', parseInt(e.target.value) || 0)}
              className="input-field"
              min={0}
            />
            <p className="text-slate-600 text-[10px] mt-1">Exact byte count for uploaded deltas (0 = skip check)</p>
          </div>
        </div>
      </div>
    </div>
  );
}
