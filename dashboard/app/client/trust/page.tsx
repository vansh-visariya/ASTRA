'use client';

import { Shield, CheckCircle, Activity, TrendingUp, AlertTriangle, Info } from 'lucide-react';
import { useWS } from '@/components/WebSocketProvider';
import { useTrustScores } from '@/hooks';
import { MetricBar } from '@/components/ui/MetricBar';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorState } from '@/components/ui/ErrorState';
import { useAuth } from '@/components/AuthContext';
import type { TrustData } from '@/lib/api/types';

function getTrustLabel(score: number): string {
  if (score >= 0.8) return 'Excellent';
  if (score >= 0.6) return 'Good';
  if (score >= 0.4) return 'Fair';
  return 'Low';
}

function getTrustBarColor(score: number): string {
  if (score >= 0.8) return 'var(--color-success)';
  if (score >= 0.6) return 'var(--color-success)';
  if (score >= 0.4) return 'var(--color-warning)';
  return 'var(--color-error)';
}

export default function ClientTrustPage() {
  const { user } = useAuth();
  const { isConnected } = useWS();
  const { data: trustData, loading, error } = useTrustScores(isConnected, user?.id);

  const score = (trustData as TrustData | null)?.score ?? 1.0;
  const quarantined = score < 0.35;

  if (loading) return <LoadingSpinner message="Loading trust score..." />;
  if (error) return <ErrorState message={error} />;

  return (
    <div className="space-y-6">
      <div className="animate-fade-in">
        <h1 className="text-2xl font-bold text-white">Trust Score</h1>
        <p className="text-slate-400 text-sm mt-1">Your reliability rating in the federated network</p>
      </div>

      <div className="instrument-card p-6 animate-fade-in" style={{ animationDelay: '0.05s', opacity: 0 }}>
        <div className="flex items-center gap-6 flex-wrap">
          {/* Conic gauge */}
          <div
            className="w-28 h-28 rounded-full shrink-0 flex items-center justify-center"
            style={{
              background: `conic-gradient(${getTrustBarColor(score)} ${(score * 360).toFixed(0)}deg, rgba(30,50,80,0.5) 0deg)`,
            }}
          >
            <div className="w-[90px] h-[90px] rounded-full flex flex-col items-center justify-center" style={{ background: 'var(--bg-card)' }}>
              <span className="data-value text-2xl text-white">{(score * 100).toFixed(0)}</span>
              <span className="font-mono text-xs" style={{ color: 'var(--text-muted)' }}>%</span>
            </div>
          </div>

          <div className="flex-1 min-w-[200px]">
            <div className="flex items-center gap-2 mb-1">
              <h2 className="text-xl font-bold" style={{ color: getTrustBarColor(score) }}>
                {getTrustLabel(score)}
              </h2>
              {quarantined && (
                <span className="status-badge status-badge--error">
                  <AlertTriangle size={11} /> QUARANTINED
                </span>
              )}
            </div>
            <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>Based on your update submissions and participation</p>
            <div className="flex items-center gap-2 mt-3">
              <Shield size={14} style={{ color: 'var(--signal-cyan)' }} />
              <span className="font-mono text-[11px]" style={{ color: 'var(--text-muted)' }}>TRUSTED PARTICIPANT</span>
            </div>
            <div className="mt-4">
              <MetricBar value={score} max={1} colorMode="trust" />
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 animate-fade-in" style={{ animationDelay: '0.15s', opacity: 0 }}>
        <div className="signal-card p-4" style={{ borderLeftColor: 'var(--signal-emerald)' }}>
          <div className="flex items-center gap-3 mb-2">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'var(--color-success-bg)' }}>
              <CheckCircle size={16} style={{ color: 'var(--color-success)' }} />
            </div>
            <h3 className="text-white font-medium text-sm">Participation</h3>
          </div>
          <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>Consistent participation in training rounds improves your trust score.</p>
        </div>
        <div className="signal-card p-4" style={{ borderLeftColor: 'var(--signal-cyan)' }}>
          <div className="flex items-center gap-3 mb-2">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'var(--signal-cyan-bg)' }}>
              <Activity size={16} style={{ color: 'var(--signal-cyan)' }} />
            </div>
            <h3 className="text-white font-medium text-sm">Update Quality</h3>
          </div>
          <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>Submitting high-quality updates with accurate metrics boosts your score.</p>
        </div>
        <div className="signal-card p-4" style={{ borderLeftColor: 'var(--signal-amber)' }}>
          <div className="flex items-center gap-3 mb-2">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'var(--signal-amber-bg)' }}>
              <TrendingUp size={16} style={{ color: 'var(--signal-amber)' }} />
            </div>
            <h3 className="text-white font-medium text-sm">Aggregation</h3>
          </div>
          <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>Trust score influences weighted aggregation of your model updates.</p>
        </div>
      </div>

      {quarantined && (
        <div className="signal-card p-4 animate-fade-in" style={{ animationDelay: '0.2s', opacity: 0, borderLeftColor: 'var(--color-error)', borderColor: 'var(--color-error-border)' }}>
          <div className="flex items-start gap-3">
            <AlertTriangle size={20} style={{ color: 'var(--color-error)' }} />
            <div>
              <h3 className="text-white font-semibold text-sm mb-1">Account Quarantined</h3>
              <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                Your account has been flagged due to low trust score. Please ensure consistent, high-quality updates to restore your score.
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="instrument-card p-5 animate-fade-in" style={{ animationDelay: '0.25s', opacity: 0 }}>
        <h3 className="section-label mb-4 flex items-center gap-2">
          <Info size={14} style={{ color: 'var(--text-muted)' }} /> HOW TRUST SCORES WORK
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-slate-400">
          <div>
            <h4 className="text-white font-medium mb-2">Score Calculation</h4>
            <ul className="space-y-1.5">
              <li>• Based on cosine similarity of your updates to the global model</li>
              <li>• Updated after each aggregation round</li>
              <li>• Range: 0.0 to 1.0 (higher is better)</li>
            </ul>
          </div>
          <div>
            <h4 className="text-white font-medium mb-2">Impact</h4>
            <ul className="space-y-1.5">
              <li>• Higher scores = more weight in aggregation</li>
              <li>• Scores below 0.35 trigger quarantine</li>
              <li>• Quarantined accounts can recover by improving quality</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
