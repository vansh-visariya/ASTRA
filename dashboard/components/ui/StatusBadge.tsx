import React from 'react';

const STATUS_COLOR_MAP: Record<string, string> = {
  active: 'success',
  online: 'success',
  approved: 'success',
  completed: 'success',
  started: 'success',
  running: 'success',

  training: 'info',
  in_progress: 'info',

  pending: 'warning',
  paused: 'warning',
  idle: 'warning',

  offline: 'error',
  failed: 'error',
  rejected: 'error',
  quarantined: 'error',
  error: 'error',

  stopped: 'muted',
};

interface StatusBadgeProps {
  status: string;
  label?: string;
  size?: 'sm' | 'md' | 'lg';
}

export function StatusBadge({ status, label, size = 'sm' }: StatusBadgeProps) {
  const intent = STATUS_COLOR_MAP[status] || 'muted';
  const displayLabel = label || status.replace(/_/g, ' ');
  const sizeClass = size === 'sm' ? 'status-badge--sm' : size === 'lg' ? 'status-badge--lg' : '';

  return (
    <span className={`status-badge status-badge--${intent} ${sizeClass}`}>
      {displayLabel}
    </span>
  );
}
