import React, { useState } from 'react';
import { Play, Pause, RotateCw, Square } from 'lucide-react';

interface GroupControlBarProps {
  isTraining: boolean;
  status: string;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
}

export function GroupControlBar({
  isTraining,
  status,
  onStart,
  onPause,
  onResume,
  onStop,
}: GroupControlBarProps) {
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  const handleAction = async (action: string, fn: () => void) => {
    setActionLoading(action);
    fn();
    setActionLoading(null);
  };

  return (
    <div className="flex items-center gap-2">
      {isTraining || status === 'TRAINING' ? (
        <>
          <button
            onClick={() => handleAction('pause', onPause)}
            disabled={actionLoading !== null}
            className="btn-secondary inline-flex items-center gap-1.5 !px-3 !py-1.5 text-xs"
          >
            <Pause size={13} /> Pause
          </button>
          <button
            onClick={() => handleAction('stop', onStop)}
            disabled={actionLoading !== null}
            className="btn-destructive inline-flex items-center gap-1.5 !px-3 !py-1.5 text-xs"
          >
            <Square size={13} /> Stop
          </button>
        </>
      ) : status === 'PAUSED' ? (
        <button
          onClick={() => handleAction('resume', onResume)}
          disabled={actionLoading !== null}
          className="btn-success inline-flex items-center gap-1.5 !px-3 !py-1.5 text-xs"
        >
          <Play size={13} /> Resume
        </button>
      ) : (
        <button
          onClick={() => handleAction('start', onStart)}
          disabled={actionLoading !== null}
          className="btn-primary inline-flex items-center gap-1.5 !px-3 !py-1.5 text-xs"
        >
          <Play size={13} /> Start
        </button>
      )}
    </div>
  );
}
