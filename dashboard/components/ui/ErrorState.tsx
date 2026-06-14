import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface ErrorStateProps {
  message?: string;
  onRetry?: () => void;
}

export function ErrorState({
  message = 'Something went wrong',
  onRetry,
}: ErrorStateProps) {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="flex flex-col items-center gap-3 text-center max-w-sm">
        <div className="w-12 h-12 rounded-full flex items-center justify-center" style={{ background: 'var(--color-error-bg)' }}>
          <AlertTriangle size={22} style={{ color: 'var(--color-error)' }} />
        </div>
        <p className="text-slate-300 text-sm">{message}</p>
        {onRetry && (
          <button
            onClick={onRetry}
            className="btn-secondary inline-flex items-center gap-2 mt-2"
          >
            <RefreshCw size={14} />
            Retry
          </button>
        )}
      </div>
    </div>
  );
}
