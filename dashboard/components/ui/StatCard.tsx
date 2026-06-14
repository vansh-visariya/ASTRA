import React from 'react';
import type { LucideIcon } from 'lucide-react';

interface StatCardProps {
  label: string;
  value: string | number;
  icon: LucideIcon;
  accent?: 'indigo' | 'emerald' | 'blue' | 'amber' | 'rose' | 'violet' | 'success' | 'error' | 'warning' | 'info' | 'muted';
  isLoading?: boolean;
  delay?: number;
}

export function StatCard({
  label,
  value,
  icon: Icon,
  accent = 'indigo',
  isLoading,
  delay = 0,
}: StatCardProps) {
  return (
    <div
      className={`stat-card accent-${accent} p-5 animate-fade-in`}
      style={{ animationDelay: `${delay}s`, opacity: 0 }}
    >
      <div className="flex items-center justify-between mb-4">
        <span className="text-slate-400 text-xs font-medium uppercase tracking-wider">
          {label}
        </span>
        <div
          className="w-9 h-9 rounded-xl flex items-center justify-center"
          style={{ background: 'rgba(255,255,255,0.06)' }}
        >
          <Icon size={17} className="text-gray-300" />
        </div>
      </div>
      {isLoading ? (
        <div className="h-8 w-20 rounded bg-white/5 animate-pulse" />
      ) : (
        <p className="text-3xl font-bold text-white tracking-tight">{value}</p>
      )}
    </div>
  );
}
