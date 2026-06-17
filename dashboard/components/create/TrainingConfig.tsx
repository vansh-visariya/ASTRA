'use client';

import React from 'react';

interface TrainingConfigProps {
  learningRate: number;
  dpEnabled: boolean;
  onChange: (field: string, value: number | boolean) => void;
}

export function TrainingConfig({
  learningRate,
  dpEnabled,
  onChange,
}: TrainingConfigProps) {
  return (
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
  );
}
