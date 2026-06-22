import React from 'react';
import type { LucideIcon } from 'lucide-react';

const accentMap: Record<string, string> = {
  indigo: 'signal-cyan',
  emerald: 'signal-emerald',
  success: 'signal-emerald',
  blue: 'signal-blue',
  info: 'signal-blue',
  amber: 'signal-amber',
  warning: 'signal-amber',
  rose: 'signal-rose',
  error: 'signal-rose',
  violet: 'signal-violet',
  muted: 'signal-blue',
};

interface StatCardProps {
  label: string;
  value: string | number;
  icon: LucideIcon;
  accent?: string;
  isLoading?: boolean;
  delay?: number;
}

export function StatCard({
  label,
  value,
  icon: Icon,
  accent = 'signal-cyan',
  isLoading,
  delay = 0,
}: StatCardProps) {
  const signalClass = accentMap[accent] || accent;

  return (
    <div
      className={`stat-card ${signalClass} animate-fade-in`}
      style={{ animationDelay: `${delay}s`, opacity: 0 }}
    >
      <div className="flex items-center justify-between mb-3">
        <span className="section-label">
          {label}
        </span>
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ background: 'rgba(55,80,130,0.1)' }}
        >
          <Icon size={15} className="text-slate-400" />
        </div>
      </div>
      {isLoading ? (
        <div className="h-8 w-24 rounded bg-white/5 animate-pulse" />
      ) : (
        <p className="data-value text-2xl text-white">{value}</p>
      )}
    </div>
  );
}
