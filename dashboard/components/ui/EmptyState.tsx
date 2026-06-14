import React from 'react';
import type { LucideIcon } from 'lucide-react';

interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  message?: string;
  action?: React.ReactNode;
}

export function EmptyState({ icon: Icon, title, message, action }: EmptyStateProps) {
  return (
    <div className="flex items-center justify-center py-16">
      <div className="flex flex-col items-center gap-3 text-center max-w-sm">
        {Icon && <Icon size={32} className="text-slate-700" />}
        <h3 className="text-white font-semibold text-sm">{title}</h3>
        {message && <p className="text-slate-500 text-xs">{message}</p>}
        {action && <div className="mt-2">{action}</div>}
      </div>
    </div>
  );
}
