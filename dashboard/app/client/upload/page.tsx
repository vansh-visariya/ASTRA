'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import Link from 'next/link';
import {
  Upload, Download, Key, FileUp,
  RefreshCw, CheckCircle, AlertCircle, Copy, Layers, HardDrive, Package,
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

const INLINE_LIMIT_MB = 100;
const CHUNK_SIZE = 8 * 1024 * 1024;

async function sha256Hex(buffer: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

type UploadResult =
  | { ok: true; version: number; bytes: number; via: 'inline' | 'chunked' }
  | { ok: false; error: string };

interface DownloadInfo {
  group_id: string;
  model_id: string;
  is_peft: boolean;
  base_model: {
    available: boolean;
    formats: string[];
    sizes: Record<string, number>;
  };
  adapter: {
    available: boolean;
    versions: number[];
    latest_size: number;
  };
  global_model: {
    available: boolean;
    current_version: number;
  };
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

const STORAGE_KEY = 'astra_downloaded_bases';

function getDownloadedBases(): Record<string, boolean> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function markBaseDownloaded(groupId: string): void {
  const bases = getDownloadedBases();
  bases[groupId] = true;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(bases));
}

export default function ClientUploadPage() {
  const { user, token } = useAuth();
  const { isConnected, onMessage } = useWS();
  const { data: groupsData, loading, error, refetch } = useGroups(isConnected);

  const [joinStatuses, setJoinStatuses] = useState<Record<string, string>>({});
  const [statusLoading, setStatusLoading] = useState(true);
  const [selectedGroup, setSelectedGroup] = useState<string>('');
  const [clientId, setClientId] = useState<string>('');
  const [file, setFile] = useState<File | null>(null);
  const [datasetSize, setDatasetSize] = useState<number>(1000);
  const [globalVersion, setGlobalVersion] = useState<number>(0);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState<{ phase: string; pct: number } | null>(null);
  const [lastResult, setLastResult] = useState<UploadResult | null>(null);
  const [copied, setCopied] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [downloading, setDownloading] = useState<'pt' | 'raw' | 'safetensors' | 'base' | 'adapter' | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<
    { received: number; total: number; pct: number } | null
  >(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const downloadAbortRef = useRef<AbortController | null>(null);

  // Download info for the selected group
  const [downloadInfo, setDownloadInfo] = useState<DownloadInfo | null>(null);
  const [downloadInfoLoading, setDownloadInfoLoading] = useState(false);

  // Training manifest for the selected group
  const [manifest, setManifest] = useState<Record<string, any> | null>(null);

  // Track which base models have been downloaded locally
  const [downloadedBases, setDownloadedBases] = useState<Record<string, boolean>>({});

  // Fetch download info when group changes
  useEffect(() => {
    if (!selectedGroup) {
      setDownloadInfo(null);
      return;
    }
    let cancelled = false;
    const fetchInfo = async () => {
      setDownloadInfoLoading(true);
      try {
        const info = await api.get<DownloadInfo>(`/api/models/${selectedGroup}/download-info`);
        if (!cancelled) setDownloadInfo(info);
      } catch {
        if (!cancelled) setDownloadInfo(null);
      } finally {
        if (!cancelled) setDownloadInfoLoading(false);
      }
    };
    fetchInfo();
    return () => { cancelled = true; };
  }, [selectedGroup]);

  // Fetch training manifest when group changes
  useEffect(() => {
    if (!selectedGroup) {
      setManifest(null);
      return;
    }
    let cancelled = false;
    const fetchManifest = async () => {
      try {
        const resp = await api.get<{ manifest: Record<string, any> }>(`/api/groups/${selectedGroup}/manifest`);
        if (!cancelled) setManifest(resp?.manifest || null);
      } catch {
        if (!cancelled) setManifest(null);
      }
    };
    fetchManifest();
    return () => { cancelled = true; };
  }, [selectedGroup]);

  // Load downloaded bases from localStorage on mount
  useEffect(() => {
    setDownloadedBases(getDownloadedBases());
  }, []);

  const downloadModel = async (fmt: 'pt' | 'raw' | 'safetensors' | 'base' | 'adapter') => {
    if (!selectedGroup) return;
    setDownloading(fmt);
    setDownloadProgress({ received: 0, total: 0, pct: 0 });
    setDownloadError(null);
    try {
      downloadAbortRef.current = new AbortController();

      // Determine the download endpoint based on the type
      let initBody: {
        download_id: string;
        total_size: number;
        sha256: string;
        num_chunks: number;
        chunks: { index: number; url: string }[];
      };

      if (fmt === 'base') {
        // Download base model (one-time, large)
        initBody = await api.post<{
          download_id: string;
          total_size: number;
          sha256: string;
          num_chunks: number;
          chunks: { index: number; url: string }[];
        }>('/api/downloads/init', {
          group_id: selectedGroup,
          format: 'pt',
          download_type: 'base',
        });
      } else if (fmt === 'adapter') {
        // Download adapter (small, per-round)
        initBody = await api.post<{
          download_id: string;
          total_size: number;
          sha256: string;
          num_chunks: number;
          chunks: { index: number; url: string }[];
        }>('/api/downloads/init', {
          group_id: selectedGroup,
          format: 'pt',
          download_type: 'adapter',
        });
      } else {
        // Legacy: download full model (pt or raw)
        initBody = await api.post<{
          download_id: string;
          total_size: number;
          sha256: string;
          num_chunks: number;
          chunks: { index: number; url: string }[];
        }>('/api/downloads/init', {
          group_id: selectedGroup,
          format: fmt,
        });
      }

      const totalBytes = initBody.total_size;
      const chunks: Uint8Array[] = [];
      for (let i = 0; i < initBody.chunks.length; i++) {
        if (downloadAbortRef.current.signal.aborted) {
          setDownloadError('cancelled');
          return;
        }
        const r = await fetch(initBody.chunks[i].url, {
          signal: downloadAbortRef.current.signal,
        });
        if (!r.ok) {
          throw new Error(`chunk ${i}: HTTP ${r.status}`);
        }
        const buf = new Uint8Array(await r.arrayBuffer());
        chunks.push(buf);
        const received = chunks.reduce((s, c) => s + c.length, 0);
        setDownloadProgress({
          received: i + 1,
          total: initBody.chunks.length,
          pct: Math.round((received / totalBytes) * 100),
        });
      }

      // Reassemble + verify sha256 in-browser
      const reassembled = new Uint8Array(totalBytes);
      let offset = 0;
      for (const c of chunks) {
        reassembled.set(c, offset);
        offset += c.length;
      }
      const actualSha = await sha256Hex(reassembled.buffer);
      if (actualSha !== initBody.sha256) {
        throw new Error(
          `sha256 mismatch: expected ${initBody.sha256}, got ${actualSha}`,
        );
      }

      // Trigger download as a file in the browser
      const blob = new Blob([reassembled], { type: 'application/octet-stream' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      if (fmt === 'base') {
        a.download = `${selectedGroup}_base_model.pt`;
      } else if (fmt === 'adapter') {
        a.download = `${selectedGroup}_adapter.pt`;
      } else {
        a.download =
          fmt === 'raw'
            ? `${selectedGroup}_model.bin`
            : fmt === 'safetensors'
              ? `${selectedGroup}_model.safetensors`
              : `${selectedGroup}_model.pt`;
      }
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      // Mark base model as downloaded in localStorage
      if (fmt === 'base') {
        markBaseDownloaded(selectedGroup);
        setDownloadedBases(getDownloadedBases());
      }

      // Best-effort telemetry
      try {
        await api.post(`/api/downloads/${initBody.download_id}/complete`, {});
      } catch {
        // ignore — telemetry only
      }
    } catch (e: any) {
      if (e?.name !== 'AbortError') {
        setDownloadError(e?.message || 'Download failed');
      } else {
        setDownloadError('cancelled');
      }
    } finally {
      setDownloading(null);
      setDownloadProgress(null);
      downloadAbortRef.current = null;
    }
  };

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

  // Subscribe to WebSocket messages for real-time version updates
  useEffect(() => {
    if (!onMessage || !selectedGroup) return;
    const unsub = onMessage((msg) => {
      if (msg.type === 'aggregation_complete' && msg.group_id === selectedGroup) {
        const v = msg.version;
        if (typeof v === 'number') setGlobalVersion(v);
      }
      if (msg.type === 'model_update' && msg.group_id === selectedGroup) {
        const v = msg.version;
        if (typeof v === 'number') setGlobalVersion(v);
      }
    });
    return unsub;
  }, [onMessage, selectedGroup]);

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return setFile(null);
    setFile(f);
    setLastResult(null);
  };

  const cancelUpload = () => {
    abortRef.current?.abort();
  };

  const uploadInline = async (
    bytes: Uint8Array, base64: string
  ): Promise<UploadResult> => {
    const meta: Record<string, unknown> = {
      dataset_size: datasetSize,
    };

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
      return { ok: true, version: result.global_version, bytes: bytes.length, via: 'inline' };
    }
    const detail = (result as any).detail || (result as any).reason || result.status;
    return { ok: false, error: detail };
  };

  const uploadChunked = async (file: File): Promise<UploadResult> => {
    abortRef.current = new AbortController();

    // 1. Compute sha256 of the whole file
    setProgress({ phase: 'hashing', pct: 0 });
    const fileBuffer = await file.arrayBuffer();
    const sha = await sha256Hex(fileBuffer);

    // 2. Init upload — get presigned URL
    setProgress({ phase: 'init', pct: 5 });
    const initBody = await api.post<{
      upload_id: string;
      upload_url: string;
      expires_at: number;
      chunk_size: number;
    }>('/api/uploads/init', {
      client_id: clientId,
      group_id: selectedGroup,
      content_length: file.size,
      sha256: sha,
      filename: file.name,
    });

    const chunkSize = initBody.chunk_size || CHUNK_SIZE;
    const total = Math.ceil(file.size / chunkSize);

    // 3. PUT all chunks (single PUT for now — chunked PUT via Content-Range
    //    is a future enhancement once the server endpoint supports Range)
    let uploaded = 0;
    setProgress({ phase: 'uploading', pct: 10 });
    for (let i = 0; i < total; i++) {
      if (abortRef.current.signal.aborted) {
        return { ok: false, error: 'cancelled' };
      }
      const start = i * chunkSize;
      const end = Math.min(start + chunkSize, file.size);
      const chunk = fileBuffer.slice(start, end);
      const putRes = await fetch(initBody.upload_url, {
        method: 'PUT',
        body: chunk,
        signal: abortRef.current.signal,
      });
      if (!putRes.ok) {
        return { ok: false, error: `PUT chunk ${i} failed: ${putRes.status}` };
      }
      uploaded += end - start;
      const pct = 10 + Math.round((uploaded / file.size) * 75);
      setProgress({ phase: 'uploading', pct });
    }

    // 4. Complete — server verifies sha256 + dispatches into the FLServer
    setProgress({ phase: 'completing', pct: 90 });
    const completeBody = await api.post<{
      status: string;
      sha256: string;
      size: number;
      global_version: number;
      message?: string;
    }>(`/api/uploads/${initBody.upload_id}/complete`, {
      sha256: sha,
      client_version: globalVersion,
      local_dataset_size: datasetSize,
      meta: {
        dataset_size: datasetSize,
      },
    });

    if (completeBody.status !== 'completed') {
      return { ok: false, error: completeBody.message || `status=${completeBody.status}` };
    }
    setProgress({ phase: 'done', pct: 100 });
    return {
      ok: true,
      version: completeBody.global_version,
      bytes: completeBody.size,
      via: 'chunked',
    };
  };

  const onSubmit = async () => {
    if (!file || !clientId || !selectedGroup) return;
    setUploading(true);
    setLastResult(null);
    try {
      const isLarge = file.size > INLINE_LIMIT_MB * 1024 * 1024;
      let result: UploadResult;
      if (isLarge) {
        result = await uploadChunked(file);
      } else {
        setProgress({ phase: 'encoding', pct: 0 });
        const buffer = await file.arrayBuffer();
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
        const base64 = btoa(binary);
        setProgress({ phase: 'uploading', pct: 50 });
        result = await uploadInline(bytes, base64);
      }

      if (result.ok) {
        setLastResult(result);
        setGlobalVersion(result.version);
        if (fileRef.current) fileRef.current.value = '';
        setFile(null);
      } else {
        setLastResult({ ok: false, error: 'error' in result ? result.error : 'upload failed' });
      }
    } catch (e: any) {
      setLastResult({ ok: false, error: e?.message || 'Upload failed' });
    } finally {
      setUploading(false);
      setProgress(null);
      abortRef.current = null;
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
  const fileSizeMB = file ? (file.size / 1024 / 1024).toFixed(2) : null;
  const isLarge = file && file.size > INLINE_LIMIT_MB * 1024 * 1024;

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
              <h3 className="text-white font-semibold text-sm">Download Model</h3>
              {downloadInfo?.is_peft && (
                <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-300">
                  PEFT Group
                </span>
              )}
            </div>

            {downloadInfoLoading ? (
              <p className="text-slate-500 text-xs">Loading model info...</p>
            ) : downloadInfo?.is_peft ? (
              /* PEFT group: show base + adapter download */
              <div className="space-y-3">
                <div className="p-3 rounded-xl text-xs text-slate-400" style={{ background: 'rgba(15,23,42,0.4)' }}>
                  <strong className="text-slate-200">PEFT workflow:</strong> Download the base model once,
                  then download only the small adapter for each training round.
                  Upload only adapter weights (not the full model).
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {/* Base model download */}
                  <div className="p-3 rounded-xl" style={{ background: 'rgba(15,23,42,0.4)' }}>
                    <div className="flex items-center gap-2 mb-2">
                      <HardDrive size={14} className="text-blue-400" />
                      <span className="text-white text-xs font-medium">Base Model (one-time)</span>
                      {downloadedBases[selectedGroup] && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300">
                          Cached locally
                        </span>
                      )}
                    </div>
                    <p className="text-slate-500 text-[11px] mb-2">
                      Frozen backbone weights. Download once, reuse for all rounds.
                    </p>
                    <div className="flex gap-2">
                      <button
                        onClick={() => downloadModel('base')}
                        disabled={downloading !== null || !downloadInfo?.base_model.available}
                        className="btn-secondary flex-1 inline-flex items-center justify-center gap-1.5 py-2 text-xs disabled:opacity-50"
                      >
                        <Download size={12} />
                        {downloading === 'base' ? 'Downloading...' : 'Download .pt'}
                      </button>
                      {downloadInfo?.base_model.formats.includes('safetensors') && (
                        <button
                          onClick={() => {
                            // Download safetensors directly via the base endpoint
                            window.open(`${API_URL}/api/models/${selectedGroup}/base?format=safetensors`);
                          }}
                          disabled={downloading !== null}
                          className="btn-secondary inline-flex items-center justify-center gap-1.5 py-2 text-xs disabled:opacity-50"
                        >
                          <Download size={12} />
                          .safetensors
                        </button>
                      )}
                    </div>
                    {downloadInfo?.base_model.available && (
                      <p className="text-slate-600 text-[10px] mt-1.5">
                        {downloadInfo.base_model.sizes.pt
                          ? `Size: ${formatBytes(downloadInfo.base_model.sizes.pt)}`
                          : 'Available on server'}
                      </p>
                    )}
                  </div>

                  {/* Adapter download */}
                  <div className="p-3 rounded-xl" style={{ background: 'rgba(15,23,42,0.4)' }}>
                    <div className="flex items-center gap-2 mb-2">
                      <Package size={14} className="text-emerald-400" />
                      <span className="text-white text-xs font-medium">Adapter (per round)</span>
                    </div>
                    <p className="text-slate-500 text-[11px] mb-2">
                      LoRA adapter weights. Download after each aggregation round.
                    </p>
                    <button
                      onClick={() => downloadModel('adapter')}
                      disabled={downloading !== null || !downloadInfo?.adapter.available}
                      className="btn-secondary w-full inline-flex items-center justify-center gap-1.5 py-2 text-xs disabled:opacity-50"
                    >
                      <Download size={12} />
                      {downloading === 'adapter' ? 'Downloading...' : 'Download Adapter'}
                    </button>
                    {downloadInfo?.adapter.available && (
                      <p className="text-slate-600 text-[10px] mt-1.5">
                        {downloadInfo.adapter.latest_size > 0
                          ? `Size: ${formatBytes(downloadInfo.adapter.latest_size)} | v${downloadInfo.global_model.current_version}`
                          : `Available | v${downloadInfo.global_model.current_version}`}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              /* Non-PEFT group: show standard download buttons */
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <button
                  onClick={() => downloadModel('raw')}
                  disabled={downloading !== null}
                  className="btn-secondary inline-flex items-center justify-center gap-2 py-2.5 text-sm disabled:opacity-50"
                >
                  <Download size={14} />
                  {downloading === 'raw' ? 'Downloading...' : 'Weights (.bin, raw float32)'}
                </button>
                <button
                  onClick={() => downloadModel('pt')}
                  disabled={downloading !== null}
                  className="btn-secondary inline-flex items-center justify-center gap-2 py-2.5 text-sm disabled:opacity-50"
                >
                  <Download size={14} />
                  {downloading === 'pt' ? 'Downloading...' : 'Full Model (.pt checkpoint)'}
                </button>
              </div>
            )}

            {downloadProgress && (
              <div className="mt-3 space-y-1">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span>
                    Downloading ({downloadProgress.received}/{downloadProgress.total} chunks)
                  </span>
                  <span className="font-mono">{downloadProgress.pct}%</span>
                </div>
                <div className="h-2 rounded-full overflow-hidden" style={{ background: 'rgba(15,23,42,0.6)' }}>
                  <div
                    className="h-full bg-blue-500 transition-all"
                    style={{ width: `${downloadProgress.pct}%` }}
                  />
                </div>
              </div>
            )}
            {downloadError && (
              <div className="mt-3 flex items-center gap-2 p-2 rounded-xl text-sm" style={{ background: 'rgba(239,68,68,0.1)' }}>
                <AlertCircle size={14} className="text-red-400" />
                <span className="text-red-300 text-xs">{downloadError}</span>
              </div>
            )}
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

              {manifest && (
                <div className="p-3 rounded-xl" style={{ background: 'rgba(15,23,42,0.4)' }}>
                  <div className="flex items-center gap-2 mb-2">
                    <FileUp size={14} className="text-cyan-400" />
                    <span className="text-white text-xs font-medium">Training Contract</span>
                  </div>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
                    {manifest.expected_delta_bytes != null && (
                      <div className="col-span-2">
                        <span className="text-slate-500">Expected delta:</span>{' '}
                        <span className="text-slate-300 font-mono">{formatBytes(manifest.expected_delta_bytes)}</span>
                      </div>
                    )}
                    {manifest.is_peft != null && (
                      <div>
                        <span className="text-slate-500">PEFT:</span>{' '}
                        <span className={manifest.is_peft ? 'text-purple-300' : 'text-slate-300'}>
                          {manifest.is_peft ? 'Yes' : 'No'}
                        </span>
                      </div>
                    )}
                    {manifest.lr != null && (
                      <div>
                        <span className="text-slate-500">Learning rate:</span>{' '}
                        <span className="text-slate-300 font-mono">{manifest.lr}</span>
                      </div>
                    )}
                    {manifest.target_modules != null && (
                      <div className="col-span-2">
                        <span className="text-slate-500">Target modules:</span>{' '}
                        <span className="text-slate-300 font-mono">{manifest.target_modules.join(', ')}</span>
                      </div>
                    )}
                    {manifest.local_epochs != null && (
                      <div>
                        <span className="text-slate-500">Local epochs:</span>{' '}
                        <span className="text-slate-300 font-mono">{manifest.local_epochs}</span>
                      </div>
                    )}
                    {manifest.batch_size != null && (
                      <div>
                        <span className="text-slate-500">Batch size:</span>{' '}
                        <span className="text-slate-300 font-mono">{manifest.batch_size}</span>
                      </div>
                    )}
                    {manifest.val_dataset != null && (
                      <div className="col-span-2">
                        <span className="text-slate-500">Val dataset:</span>{' '}
                        <span className="text-slate-300 font-mono">{manifest.val_dataset}</span>
                      </div>
                    )}
                    {manifest.lora_rank != null && (
                      <div>
                        <span className="text-slate-500">LoRA rank:</span>{' '}
                        <span className="text-slate-300 font-mono">{manifest.lora_rank}</span>
                      </div>
                    )}
                    {manifest.lora_alpha != null && (
                      <div>
                        <span className="text-slate-500">LoRA alpha:</span>{' '}
                        <span className="text-slate-300 font-mono">{manifest.lora_alpha}</span>
                      </div>
                    )}
                  </div>
                  <p className="text-slate-600 text-[10px] mt-2">
                    Your upload must match the expected delta size. Training params are advisory.
                  </p>
                </div>
              )}

              <div>
                <label className="text-slate-400 text-xs block mb-1">
                  Delta File (.pt / .npy / .bin / .safetensors) — {downloadInfo?.is_peft ? 'adapter weights only' : 'raw float32 weight bytes'}
                </label>
                <input
                  ref={fileRef}
                  type="file"
                  accept=".pt,.npy,.bin,.raw,.safetensors"
                  onChange={onFileChange}
                  className="block w-full text-sm text-slate-300 file:mr-3 file:py-2 file:px-3 file:rounded-lg file:border-0 file:text-sm file:bg-slate-800 file:text-white"
                />
                {file && (
                  <p className="text-slate-500 text-xs mt-1 flex items-center gap-2">
                    <span>{file.name} — {fileSizeMB} MB</span>
                    {isLarge && (
                      <span className="text-amber-400 font-medium">
                        (large file — chunked upload to /api/uploads)
                      </span>
                    )}
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
              </div>

              {progress && (
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span>{progress.phase}</span>
                    <span className="font-mono">{progress.pct}%</span>
                  </div>
                  <div className="h-2 rounded-full overflow-hidden" style={{ background: 'rgba(15,23,42,0.6)' }}>
                    <div
                      className="h-full bg-emerald-500 transition-all"
                      style={{ width: `${progress.pct}%` }}
                    />
                  </div>
                </div>
              )}

              <div className="flex items-center gap-2">
                <button
                  onClick={onSubmit}
                  disabled={uploading || !file || !clientId}
                  className="btn-emerald text-white text-sm px-5 py-2.5 inline-flex items-center gap-2 disabled:opacity-50"
                >
                  <FileUp size={15} />
                  {uploading ? 'Uploading…' : 'Upload Delta'}
                </button>
                {uploading && (
                  <button
                    onClick={cancelUpload}
                    className="btn-secondary text-sm px-4 py-2.5"
                  >
                    Cancel
                  </button>
                )}
              </div>

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
                        Accepted via {lastResult.via} — global model is now v{lastResult.version}
                        {' '}({(lastResult.bytes / 1024 / 1024).toFixed(2)} MB)
                      </span>
                    </>
                  ) : (
                    <>
                      <AlertCircle size={15} className="text-red-400" />
                      <span className="text-red-300">{'error' in lastResult ? lastResult.error || 'Upload failed' : 'Upload failed'}</span>
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
          {(downloadInfo?.is_peft ? [
            { step: '1', title: 'Download Base Model', desc: 'Pull the frozen backbone once (.pt or .safetensors)' },
            { step: '2', title: 'Fine-tune Locally', desc: 'Apply LoRA adapters on your data, train only adapter weights' },
            { step: '3', title: 'Download Adapter', desc: 'Pull the latest global adapter weights each round' },
            { step: '4', title: 'Upload Adapter', desc: 'Submit only your adapter delta (not the full model)' },
          ] : [
            { step: '1', title: 'Train Externally', desc: 'Run your training script on your own hardware/data' },
            { step: '2', title: 'Download Global', desc: 'Pull the current global model from the group' },
            { step: '3', title: 'Compute Delta', desc: 'Subtract old weights from your new weights' },
            { step: '4', title: 'Upload Here', desc: 'Submit the delta bytes via this page or REST' },
          ]).map((item) => (
            <div key={item.step} className="p-3 rounded-xl" style={{ background: 'rgba(15,23,42,0.4)' }}>
              <span
                className="w-6 h-6 rounded-lg flex items-center justify-center text-[11px] font-bold mb-2 inline-flex"
                style={{ color: 'var(--color-info)', background: 'rgba(59,130,246,0.12)' }}
              >
                {item.step}
              </span>
              <p className="text-white text-xs font-medium">{item.title}</p>
              <p className="text-slate-500 text-[11px] mt-0.5">{item.desc}</p>
            </div>
          ))}
        </div>
        <div className="mt-4 p-3 rounded-xl text-xs text-slate-400" style={{ background: 'rgba(15,23,42,0.4)' }}>
          <strong className="text-slate-200">Upload format:</strong>{' '}
          {downloadInfo?.is_peft ? (
            <>Adapter-only delta (LoRA weights). Use <code className="font-mono">flatten_peft_params()</code> from <code className="font-mono">astra.core.models.model_zoo</code> to extract adapter parameters.</>
          ) : (
            <>Raw float32 little-endian weight delta,
            byte count = <code className="font-mono">total_params × 4</code>. Files ≤ {INLINE_LIMIT_MB} MB are sent inline;
            larger files are uploaded via the presigned-URL flow (chunked, resumable, sha256-verified).</>
          )}
        </div>
      </div>
    </div>
  );
}
