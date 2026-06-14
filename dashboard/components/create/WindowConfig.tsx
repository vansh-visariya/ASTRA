'use client';

import React from 'react';

interface WindowConfigProps {
  windowSize: number;
  timeLimit: number;
  onChange: (field: string, value: number) => void;
}

export function WindowConfig({ windowSize, timeLimit, onChange }: WindowConfigProps) {
  return (
    <div className="glass-card p-5 space-y-4">
      <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Async Window Configuration</h3>
      <p className="text-slate-500 text-xs">
        Aggregation triggers when <em>either</em> N client updates arrive <em>or</em> the time limit expires.
      </p>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="text-slate-400 text-xs font-medium block mb-1.5">Window Size (N updates)</label>
          <input
            type="number"
            value={windowSize}
            onChange={(e) => onChange('window_size', parseInt(e.target.value) || 3)}
            className="input-field"
            min={1}
            max={100}
          />
        </div>
        <div>
          <label className="text-slate-400 text-xs font-medium block mb-1.5">Time Limit (seconds)</label>
          <input
            type="number"
            value={timeLimit}
            onChange={(e) => onChange('time_limit', parseInt(e.target.value) || 20)}
            className="input-field"
            min={1}
            max={3600}
          />
        </div>
      </div>
    </div>
  );
}
