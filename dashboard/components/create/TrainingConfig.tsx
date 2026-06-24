'use client';

import React from 'react';

interface TrainingConfigProps {
  learningRate: number;
  dpEnabled: boolean;
  localEpochs: number;
  batchSize: number;
  optimizer: string;
  lossFunction: string;
  maxGradNorm: string;
  inputShape: string;
  numClasses: string;
  labelType: string;
  dataDescription: string;
  preprocessingSteps: string;
  valMetric: string;
  onChange: (field: string, value: number | boolean | string) => void;
}

export function TrainingConfig({
  learningRate,
  dpEnabled,
  localEpochs,
  batchSize,
  optimizer,
  lossFunction,
  maxGradNorm,
  inputShape,
  numClasses,
  labelType,
  dataDescription,
  preprocessingSteps,
  valMetric,
  onChange,
}: TrainingConfigProps) {
  return (
    <div className="space-y-4">
      <div className="glass-card p-5 space-y-4">
        <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Server Configuration</h3>
        <p className="text-slate-500 text-xs">
          Clients train externally on their own data. The server only applies these settings when aggregating received deltas.
        </p>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-slate-400 text-xs font-medium block mb-1.5">Server Learning Rate</label>
            <input
              type="number"
              step="0.001"
              value={learningRate}
              onChange={(e) => onChange('lr', parseFloat(e.target.value) || 0.01)}
              className="input-field"
              min={0.0001}
              max={1}
            />
          </div>
          <div className="flex items-end">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={dpEnabled}
                onChange={(e) => onChange('dp_enabled', e.target.checked)}
                className="w-4 h-4 rounded accent-white"
              />
              <span className="text-slate-400 text-xs font-medium">Enable Differential Privacy (server-side)</span>
            </label>
          </div>
        </div>
      </div>

      <div className="glass-card p-5 space-y-4">
        <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Training Protocol</h3>
        <p className="text-slate-500 text-xs">
          These parameters are sent to clients as a training contract. Clients see them before training and should follow them.
        </p>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-slate-400 text-xs font-medium block mb-1.5">Optimizer</label>
            <select
              value={optimizer}
              onChange={(e) => onChange('optimizer', e.target.value)}
              className="input-field"
            >
              <option value="adamw">AdamW</option>
              <option value="adam">Adam</option>
              <option value="sgd">SGD</option>
              <option value="rmsprop">RMSprop</option>
            </select>
          </div>
          <div>
            <label className="text-slate-400 text-xs font-medium block mb-1.5">Loss Function</label>
            <select
              value={lossFunction}
              onChange={(e) => onChange('loss_function', e.target.value)}
              className="input-field"
            >
              <option value="cross_entropy">Cross Entropy</option>
              <option value="mse">MSE</option>
              <option value="nll">NLL</option>
              <option value="bce">Binary Cross Entropy</option>
            </select>
          </div>
          <div>
            <label className="text-slate-400 text-xs font-medium block mb-1.5">Local Epochs</label>
            <input
              type="number"
              value={localEpochs}
              onChange={(e) => onChange('local_epochs', parseInt(e.target.value) || 2)}
              className="input-field"
              min={1}
              max={100}
            />
            <p className="text-slate-600 text-[10px] mt-1">Recommended epochs per client per round</p>
          </div>
          <div>
            <label className="text-slate-400 text-xs font-medium block mb-1.5">Batch Size</label>
            <input
              type="number"
              value={batchSize}
              onChange={(e) => onChange('batch_size', parseInt(e.target.value) || 32)}
              className="input-field"
              min={1}
              max={4096}
            />
            <p className="text-slate-600 text-[10px] mt-1">Recommended batch size for training</p>
          </div>
          <div>
            <label className="text-slate-400 text-xs font-medium block mb-1.5">Max Gradient Norm</label>
            <input
              type="number"
              step="0.1"
              value={maxGradNorm}
              onChange={(e) => onChange('max_grad_norm', e.target.value)}
              className="input-field"
              placeholder="None (no clipping)"
            />
            <p className="text-slate-600 text-[10px] mt-1">Optional gradient clipping norm</p>
          </div>
        </div>
      </div>

      <div className="glass-card p-5 space-y-4">
        <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Data Schema</h3>
        <p className="text-slate-500 text-xs">
          Informational — describes expected data format for clients.
        </p>
        <div className="grid grid-cols-2 gap-4">
          <div className="col-span-2">
            <label className="text-slate-400 text-xs font-medium block mb-1.5">Data Description</label>
            <textarea
              value={dataDescription}
              onChange={(e) => onChange('data_description', e.target.value)}
              className="input-field w-full h-16 resize-none"
              placeholder="e.g., MNIST digits 0-9, 28x28 grayscale images flattened to 784-dim vectors"
            />
          </div>
          <div>
            <label className="text-slate-400 text-xs font-medium block mb-1.5">Input Shape</label>
            <input
              type="text"
              value={inputShape}
              onChange={(e) => onChange('input_shape', e.target.value)}
              className="input-field"
              placeholder="e.g., 784 or 3,224,224"
            />
            <p className="text-slate-600 text-[10px] mt-1">Comma-separated dimensions</p>
          </div>
          <div>
            <label className="text-slate-400 text-xs font-medium block mb-1.5">Number of Classes</label>
            <input
              type="number"
              value={numClasses}
              onChange={(e) => onChange('num_classes', e.target.value)}
              className="input-field"
              placeholder="e.g., 10"
              min={1}
            />
          </div>
          <div>
            <label className="text-slate-400 text-xs font-medium block mb-1.5">Label Type</label>
            <select
              value={labelType}
              onChange={(e) => onChange('label_type', e.target.value)}
              className="input-field"
            >
              <option value="">Not specified</option>
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
              <option value="causal_lm">Causal LM</option>
            </select>
          </div>
          <div>
            <label className="text-slate-400 text-xs font-medium block mb-1.5">Preprocessing Steps</label>
            <input
              type="text"
              value={preprocessingSteps}
              onChange={(e) => onChange('preprocessing_steps', e.target.value)}
              className="input-field"
              placeholder="e.g., normalize, tokenize"
            />
            <p className="text-slate-600 text-[10px] mt-1">Comma-separated steps</p>
          </div>
        </div>
      </div>

      <div className="glass-card p-5 space-y-4">
        <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Validation</h3>
        <p className="text-slate-500 text-xs">
          Server evaluates the global model after each aggregation round. Upload validation data after creating the group.
        </p>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-slate-400 text-xs font-medium block mb-1.5">Validation Metric</label>
            <select
              value={valMetric}
              onChange={(e) => onChange('val_metric', e.target.value)}
              className="input-field"
            >
              <option value="accuracy">Accuracy</option>
              <option value="f1">F1 Score</option>
              <option value="precision">Precision</option>
              <option value="recall">Recall</option>
              <option value="mse">MSE</option>
            </select>
          </div>
        </div>
        <p className="text-slate-600 text-[10px]">
          After creating the group, use the group detail page to upload a .pt validation dataset.
        </p>
      </div>
    </div>
  );
}
