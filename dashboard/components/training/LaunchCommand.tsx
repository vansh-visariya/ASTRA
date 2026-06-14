import React, { useState } from 'react';
import { Check, Copy } from 'lucide-react';

interface LaunchCommandProps {
  command: string;
}

export function LaunchCommand({ command }: LaunchCommandProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // clipboard unavailable
    }
  };

  return (
    <div className="launch-command flex items-center justify-between gap-3 group">
      <code className="truncate flex-1 select-all">{command}</code>
      <button
        onClick={handleCopy}
        className="shrink-0 p-1 rounded-md opacity-50 hover:opacity-100 transition-opacity"
        title="Copy command"
      >
        {copied ? (
          <Check size={14} style={{ color: 'var(--color-success)' }} />
        ) : (
          <Copy size={14} className="text-slate-400" />
        )}
      </button>
    </div>
  );
}
