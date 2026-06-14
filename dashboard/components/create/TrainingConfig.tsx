'use client';

import React from 'react';

interface TrainingConfigProps {
  localEpochs: number;
  batchSize: number;
  learningRate: number;
  dpEnabled: boolean;
  onChange: (field: string, value: number | boolean) => void;
}

export function TrainingConfig({
  localEpochs,
  batchSize,
  learningRate,
  dpEnabled,
  onChange,
}: TrainingConfigProps) {
  return (
    <div className="glass-card p-5 space-y-4">
      <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Training Configuration</h3>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="text-slate-400 text-xs font-medium block mb-1.5">Local Epochs</label>
          <input
            type="number"
            value={localEpochs}
            onChange={(e) => onChange('local_epochs', parseInt(e.target.value) || 1)}
            className="input-field"
            min={1}
            max={50}
          />
        </div>
        <div>
          <label className="text-slate-400 text-xs font-medium block mb-1.5">Batch Size</label>
          <input
            type="number"
            value={batchSize}
            onChange={(e) => onChange('batch_size', parseInt(e.target.value) || 32)}
            className="input-field"
            min={1}
            max={512}
          />
        </div>
        <div>
          <label className="text-slate-400 text-xs font-medium block mb-1.5">Learning Rate</label>
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
            <span className="text-slate-400 text-xs font-medium">Enable Differential Privacy</span>
          </label>
        </div>
      </div>
    </div>
  );
}
