import React from 'react';

interface MetricBarProps {
  value: number;
  max?: number;
  colorMode?: 'trust' | 'static';
  staticColor?: string;
  label?: string;
}

function getTrustClass(value: number): string {
  if (value < 0.35) return 'metric-bar__fill--trust-danger';
  if (value < 0.7) return 'metric-bar__fill--trust-low';
  if (value < 0.9) return 'metric-bar__fill--trust-good';
  return 'metric-bar__fill--trust-great';
}

export function MetricBar({
  value,
  max = 1,
  colorMode = 'trust',
  label,
}: MetricBarProps) {
  const pct = Math.min((value / max) * 100, 100);

  return (
    <div>
      {label && (
        <div className="flex justify-between mb-1">
          <span className="text-slate-500 text-[10px] uppercase tracking-wider">{label}</span>
          <span className="text-slate-400 text-xs font-mono">{(value * 100).toFixed(0)}%</span>
        </div>
      )}
      <div className="metric-bar">
        <div
          className={`metric-bar__fill ${colorMode === 'trust' ? getTrustClass(value) : ''}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
