'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import Link from 'next/link';
import {
  Upload, Download, Key, FileUp,
  RefreshCw, CheckCircle, AlertCircle, Copy, Layers,
} from 'lucide-react';
import { useAuth } from '@/components/AuthContext';
import { useGroups } from '@/hooks';
import { useWS } from '@/components/WebSocketProvider';
import { getMyJoinStatus, activateJoin } from '@/lib/api/endpoints';
import { api } from '@/lib/api/client';
import { API_URL } from '@/lib/config';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorState } from '@/components/ui/ErrorState';
import { EmptyState } from '@/components/ui/EmptyState';
import type { Group } from '@/lib/api/types';

const MAX_FILE_MB = 100;

export default function ClientUploadPage() {
  const { user, token } = useAuth();
  const { isConnected } = useWS();
  const { data: groupsData, loading, error, refetch } = useGroups(!isConnected);

  const [joinStatuses, setJoinStatuses] = useState<Record<string, string>>({});
  const [statusLoading, setStatusLoading] = useState(true);
  const [selectedGroup, setSelectedGroup] = useState<string>('');
  const [clientId, setClientId] = useState<string>('');
  const [file, setFile] = useState<File | null>(null);
  const [datasetSize, setDatasetSize] = useState<number>(1000);
  const [trainAccuracy, setTrainAccuracy] = useState<string>('');
  const [trainLoss, setTrainLoss] = useState<string>('');
  const [globalVersion, setGlobalVersion] = useState<number>(0);
  const [uploading, setUploading] = useState(false);
  const [lastResult, setLastResult] = useState<
    { ok: boolean; version?: number; error?: string } | null
  >(null);
  const [copied, setCopied] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const groups: Group[] = useMemo(() => (groupsData as any)?.groups || [], [groupsData]);
  const activatedGroups = groups.filter((g) => joinStatuses[g.group_id] === 'activated');

  // Default selection: first activated group
  useEffect(() => {
    if (!selectedGroup && activatedGroups.length > 0) {
      const g = activatedGroups[0];
      setSelectedGroup(g.group_id);
      setClientId(`${user?.username || 'client'}_${g.group_id}`);
    }
  }, [activatedGroups, selectedGroup, user]);

  // Sync join statuses
  useEffect(() => {
    let cancelled = false;
    const sync = async () => {
      const next: Record<string, string> = {};
      for (const g of groups) {
        try {
          const r = await getMyJoinStatus(g.group_id);
          next[g.group_id] = (r.status || 'none') as string;
        } catch {
          next[g.group_id] = 'none';
        }
      }
      if (!cancelled) {
        setJoinStatuses(next);
        setStatusLoading(false);
      }
    };
    sync();
    return () => { cancelled = true; };
  }, [groups]);

  // Fetch latest global version for the selected group
  useEffect(() => {
    if (!selectedGroup) return;
    let cancelled = false;
    const fetchVersion = async () => {
      try {
        const data = await api.get<{ group: Group }>(`/api/groups/${selectedGroup}`);
        if (!cancelled && data?.group) setGlobalVersion(data.group.model_version || 0);
      } catch {
        // ignore
      }
    };
    fetchVersion();
    return () => { cancelled = true; };
  }, [selectedGroup]);

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return setFile(null);
    if (f.size > MAX_FILE_MB * 1024 * 1024) {
      setLastResult({ ok: false, error: `File too large (${f.size} bytes). Limit is ${MAX_FILE_MB} MB.` });
      setFile(null);
      if (fileRef.current) fileRef.current.value = '';
      return;
    }
    setFile(f);
    setLastResult(null);
  };

  const onSubmit = async () => {
    if (!file || !clientId || !selectedGroup) return;
    setUploading(true);
    setLastResult(null);
    try {
      const buffer = await file.arrayBuffer();
      const bytes = new Uint8Array(buffer);
      let binary = '';
      for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
      const base64 = btoa(binary);

      const meta: Record<string, unknown> = {};
      if (trainAccuracy !== '') meta.train_accuracy = Number(trainAccuracy);
      if (trainLoss !== '') meta.train_loss = Number(trainLoss);
      meta.dataset_size = datasetSize;

      const result = await api.post<{ status: string; global_version: number }>(
        `/api/clients/${clientId}/delta`,
        {
          client_id: clientId,
          client_version: globalVersion,
          local_updates: base64,
          update_type: 'delta',
          local_dataset_size: datasetSize,
          meta,
        }
      );
      if (result.status === 'accepted') {
        setLastResult({ ok: true, version: result.global_version });
        setGlobalVersion(result.global_version);
        if (fileRef.current) fileRef.current.value = '';
        setFile(null);
      } else {
        setLastResult({ ok: false, error: result.status });
      }
    } catch (e: any) {
      setLastResult({ ok: false, error: e?.message || 'Upload failed' });
    } finally {
      setUploading(false);
    }
  };

  const copyToken = async () => {
    if (!token) return;
    try {
      await navigator.clipboard.writeText(token);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // ignore
    }
  };

  const activate = async (gid: string) => {
    try {
      await activateJoin(gid);
      setJoinStatuses((prev) => ({ ...prev, [gid]: 'activated' }));
    } catch {
      // ignore
    }
  };

  if (loading || statusLoading) return <LoadingSpinner message="Loading groups..." />;
  if (error) return <ErrorState message={error} onRetry={refetch} />;

  const group = groups.find((g) => g.group_id === selectedGroup);
  const hasActivated = activatedGroups.length > 0;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between animate-fade-in">
        <div>
          <h1 className="text-2xl font-bold text-white">Upload Delta</h1>
          <p className="text-slate-400 text-sm mt-1">
            Train externally, then submit your model delta to the server
          </p>
        </div>
        <button onClick={refetch} className="btn-secondary !px-3 !py-2.5 inline-flex items-center gap-1.5">
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      <div className="glass-card p-5 animate-fade-in">
        <div className="flex items-center gap-3 mb-3">
          <Key size={17} className="text-amber-400" />
          <h3 className="text-white font-semibold text-sm">Your Auth Token</h3>
        </div>
        <p className="text-slate-400 text-xs mb-3">
          Use this token from your external training script:
        </p>
        <div className="flex items-center gap-2">
          <code
            className="flex-1 px-3 py-2 rounded-lg text-xs font-mono text-slate-300 truncate"
            style={{ background: 'rgba(15,23,42,0.6)' }}
          >
            {token || 'Not signed in'}
          </code>
          <button
            onClick={copyToken}
            className="btn-secondary !px-3 !py-2 inline-flex items-center gap-1.5"
            disabled={!token}
          >
            {copied ? <CheckCircle size={14} className="text-emerald-400" /> : <Copy size={14} />}
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>
      </div>

      {!hasActivated && (
        <div className="glass-card p-5 animate-fade-in">
          <h3 className="text-white font-semibold text-sm mb-3">Activate a Group</h3>
          {groups.length === 0 ? (
            <EmptyState
              icon={Layers}
              title="No Groups Available"
              message="Ask an admin to create a group before you can upload deltas"
              action={
                <Link href="/client/groups" className="btn-emerald inline-flex text-white text-sm px-4 py-2 items-center gap-2">
                  Browse Groups
                </Link>
              }
            />
          ) : (
            <div className="space-y-2">
              {groups.map((g) => (
                <div
                  key={g.group_id}
                  className="flex items-center justify-between p-3 rounded-xl"
                  style={{ background: 'rgba(15,23,42,0.4)' }}
                >
                  <div className="flex items-center gap-3">
                    <Layers size={16} className="text-slate-400" />
                    <span className="text-white text-sm font-medium">{g.group_id}</span>
                    <span className="text-slate-500 text-xs">({g.model_id})</span>
                    <span className="text-slate-600 text-xs">· {joinStatuses[g.group_id]}</span>
                  </div>
                  <button
                    onClick={() => activate(g.group_id)}
                    className="btn-emerald text-white text-xs px-3 py-1.5"
                  >
                    Activate
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {hasActivated && (
        <>
          <div className="glass-card p-5 animate-fade-in">
            <div className="flex items-center gap-3 mb-4">
              <Download size={17} className="text-blue-400" />
              <h3 className="text-white font-semibold text-sm">Download Global Model</h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <a
                href={`${API_URL}/api/models/${selectedGroup}/download`}
                target="_blank"
                rel="noreferrer"
                className="btn-secondary inline-flex items-center justify-center gap-2 py-2.5 text-sm"
              >
                <Download size={14} /> Full Model (.pt)
              </a>
              <a
                href={`${API_URL}/api/models/${selectedGroup}/adapter`}
                target="_blank"
                rel="noreferrer"
                className="btn-secondary inline-flex items-center justify-center gap-2 py-2.5 text-sm"
              >
                <Download size={14} /> LoRA Adapter (.pt)
              </a>
            </div>
            <p className="text-slate-500 text-xs mt-3">
              Current global version: <span className="text-slate-300 font-mono">v{globalVersion}</span>
            </p>
          </div>

          <div className="glass-card p-5 animate-fade-in">
            <div className="flex items-center gap-3 mb-4">
              <Upload size={17} className="text-emerald-400" />
              <h3 className="text-white font-semibold text-sm">Upload Delta</h3>
            </div>

            <div className="space-y-3">
              <div>
                <label className="text-slate-400 text-xs block mb-1">Group</label>
                <select
                  value={selectedGroup}
                  onChange={(e) => {
                    setSelectedGroup(e.target.value);
                    setClientId(`${user?.username || 'client'}_${e.target.value}`);
                  }}
                  className="w-full px-3 py-2 rounded-lg text-sm bg-slate-900 border border-slate-700 text-white"
                >
                  {activatedGroups.map((g) => (
                    <option key={g.group_id} value={g.group_id}>
                      {g.group_id} ({g.model_id})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="text-slate-400 text-xs block mb-1">Client ID</label>
                <input
                  type="text"
                  value={clientId}
                  onChange={(e) => setClientId(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg text-sm bg-slate-900 border border-slate-700 text-white font-mono"
                />
              </div>

              <div>
                <label className="text-slate-400 text-xs block mb-1">Delta File (.pt / .npy / .bin)</label>
                <input
                  ref={fileRef}
                  type="file"
                  accept=".pt,.npy,.bin"
                  onChange={onFileChange}
                  className="block w-full text-sm text-slate-300 file:mr-3 file:py-2 file:px-3 file:rounded-lg file:border-0 file:text-sm file:bg-slate-800 file:text-white"
                />
                {file && (
                  <p className="text-slate-500 text-xs mt-1">
                    {file.name} — {(file.size / 1024).toFixed(1)} KB
                  </p>
                )}
              </div>

              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="text-slate-400 text-xs block mb-1">Dataset Size</label>
                  <input
                    type="number"
                    value={datasetSize}
                    onChange={(e) => setDatasetSize(Number(e.target.value))}
                    min={1}
                    className="w-full px-3 py-2 rounded-lg text-sm bg-slate-900 border border-slate-700 text-white"
                  />
                </div>
                <div>
                  <label className="text-slate-400 text-xs block mb-1">Train Acc (opt)</label>
                  <input
                    type="number"
                    step="0.0001"
                    value={trainAccuracy}
                    onChange={(e) => setTrainAccuracy(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg text-sm bg-slate-900 border border-slate-700 text-white"
                  />
                </div>
                <div>
                  <label className="text-slate-400 text-xs block mb-1">Train Loss (opt)</label>
                  <input
                    type="number"
                    step="0.0001"
                    value={trainLoss}
                    onChange={(e) => setTrainLoss(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg text-sm bg-slate-900 border border-slate-700 text-white"
                  />
                </div>
              </div>

              <button
                onClick={onSubmit}
                disabled={uploading || !file || !clientId}
                className="btn-emerald text-white text-sm px-5 py-2.5 inline-flex items-center gap-2 disabled:opacity-50"
              >
                <FileUp size={15} />
                {uploading ? 'Uploading…' : 'Upload Delta'}
              </button>

              {lastResult && (
                <div
                  className="flex items-center gap-2 p-3 rounded-xl text-sm"
                  style={{
                    background: lastResult.ok ? 'rgba(16,185,129,0.1)' : 'rgba(239,68,68,0.1)',
                  }}
                >
                  {lastResult.ok ? (
                    <>
                      <CheckCircle size={15} className="text-emerald-400" />
                      <span className="text-emerald-300">
                        Accepted — global model is now v{lastResult.version}
                      </span>
                    </>
                  ) : (
                    <>
                      <AlertCircle size={15} className="text-red-400" />
                      <span className="text-red-300">{lastResult.error || 'Upload failed'}</span>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        </>
      )}

      <div className="glass-card p-5 animate-fade-in">
        <h3 className="text-white font-semibold text-sm mb-3">How It Works</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          {[
            { step: '1', title: 'Train Externally', desc: 'Run your training script on your own hardware/data' },
            { step: '2', title: 'Download Global', desc: 'Pull the current global model from the group' },
            { step: '3', title: 'Compute Delta', desc: 'Subtract old weights from your new weights' },
            { step: '4', title: 'Upload Here', desc: 'Submit the delta bytes via this page or REST' },
          ].map((item) => (
            <div key={item.step} className="p-3 rounded-xl" style={{ background: 'rgba(15,23,42,0.4)' }}>
              <span className="w-6 h-6 rounded-lg flex items-center justify-center text-[11px] font-bold mb-2 inline-flex"
                style={{ color: 'var(--color-info)', background: 'rgba(59,130,246,0.12)' }}>
                {item.step}
              </span>
              <p className="text-white text-xs font-medium">{item.title}</p>
              <p className="text-slate-500 text-[11px] mt-0.5">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
