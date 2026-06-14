import React from 'react';

interface LoadingSpinnerProps {
  message?: string;
}

export function LoadingSpinner({ message = 'Loading...' }: LoadingSpinnerProps) {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="flex flex-col items-center gap-3">
        <div className="w-8 h-8 border-2 border-white/30 border-t-transparent rounded-full animate-spin" />
        <span className="text-slate-500 text-xs">{message}</span>
      </div>
    </div>
  );
}
