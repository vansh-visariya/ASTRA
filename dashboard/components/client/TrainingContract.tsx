'use client';

import React from 'react';
import { ScrollText } from 'lucide-react';
import type { TrainingManifest } from '@/lib/api/types';

interface TrainingContractProps {
  manifest: TrainingManifest;
  compact?: boolean;
}

function Field({ label, value, mono = true }: { label: string; value: React.ReactNode; mono?: boolean }) {
  if (value === null || value === undefined || value === '') return null;
  return (
    <div>
      <p className="text-slate-500 text-[11px]">{label}</p>
      <p className={`text-white text-xs mt-0.5 ${mono ? 'font-mono' : ''}`}>{String(value)}</p>
    </div>
  );
}

export function TrainingContract({ manifest, compact = false }: TrainingContractProps) {
  if (!manifest) return null;

  return (
    <div className="p-3 rounded-xl" style={{ background: 'rgba(15,23,42,0.4)' }}>
      <div className="flex items-center gap-2 mb-3">
        <ScrollText size={14} className="text-cyan-400" />
        <span className="text-white text-xs font-medium">Training Contract</span>
        {manifest.contract_version != null && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 font-mono">
            v{manifest.contract_version}
          </span>
        )}
      </div>

      {manifest.data_description && (
        <p className="text-slate-400 text-[11px] mb-3 italic">{manifest.data_description}</p>
      )}

      {/* Architecture */}
      {(manifest.is_peft != null || manifest.target_modules || manifest.lora_rank != null) && (
        <div className="mb-2">
          <p className="text-slate-600 text-[10px] uppercase tracking-wider mb-1">Architecture</p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
            <Field label="PEFT" value={manifest.is_peft ? 'Yes' : 'No'} />
            {manifest.target_modules && (
              <Field label="Target modules" value={manifest.target_modules.join(', ')} />
            )}
            <Field label="LoRA rank" value={manifest.lora_rank} />
            <Field label="LoRA alpha" value={manifest.lora_alpha} />
            {manifest.expected_delta_bytes != null && (
              <Field label="Expected delta" value={`${manifest.expected_delta_bytes.toLocaleString()} bytes`} />
            )}
          </div>
        </div>
      )}

      {/* Training Protocol */}
      <div className="mb-2">
        <p className="text-slate-600 text-[10px] uppercase tracking-wider mb-1">Training Protocol</p>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          <Field label="Learning rate" value={manifest.lr} />
          <Field label="Optimizer" value={manifest.optimizer} />
          <Field label="Loss function" value={manifest.loss_function} />
          <Field label="Local epochs" value={manifest.local_epochs} />
          <Field label="Batch size" value={manifest.batch_size} />
          <Field label="Max grad norm" value={manifest.max_grad_norm} />
        </div>
      </div>

      {/* Data Schema */}
      {(manifest.input_shape || manifest.num_classes || manifest.label_type || manifest.preprocessing_steps) && (
        <div className="mb-2">
          <p className="text-slate-600 text-[10px] uppercase tracking-wider mb-1">Data Schema</p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
            <Field label="Input shape" value={manifest.input_shape?.join(', ')} />
            <Field label="Num classes" value={manifest.num_classes} />
            <Field label="Label type" value={manifest.label_type} />
            {manifest.preprocessing_steps && (
              <Field label="Preprocessing" value={manifest.preprocessing_steps.join(', ')} />
            )}
          </div>
        </div>
      )}

      {/* Verification */}
      {(manifest.val_dataset || manifest.val_metric) && (
        <div>
          <p className="text-slate-600 text-[10px] uppercase tracking-wider mb-1">Verification</p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
            <Field label="Validation dataset" value={manifest.val_dataset ? 'Uploaded' : 'None'} />
            <Field label="Validation metric" value={manifest.val_metric} />
          </div>
        </div>
      )}

      {!compact && (
        <p className="text-slate-600 text-[10px] mt-3">
          Training params are advisory. Your upload is validated for float32 format and NaN/Inf values.
        </p>
      )}
    </div>
  );
}
