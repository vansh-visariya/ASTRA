'use client';

import { Shield, CheckCircle, Activity, TrendingUp, AlertTriangle, Info } from 'lucide-react';
import { useWS } from '@/components/WebSocketProvider';
import { useTrustScores } from '@/hooks';
import { MetricBar } from '@/components/ui/MetricBar';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
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
  const { data: trustData, loading } = useTrustScores(!isConnected, user?.id);

  const score = (trustData as TrustData | null)?.score ?? 1.0;
  const quarantined = score < 0.35;

  if (loading) return <LoadingSpinner message="Loading trust score..." />;

  return (
    <div className="space-y-6">
      <div className="animate-fade-in">
        <h1 className="text-2xl font-bold text-white">Trust Score</h1>
        <p className="text-slate-400 text-sm mt-1">Your reliability rating in the federated network</p>
      </div>

      <div className="glass-card p-8 animate-fade-in" style={{ animationDelay: '0.05s', opacity: 0 }}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-6">
            <div
              className="w-32 h-32 rounded-full flex items-center justify-center"
              style={{
                background: `conic-gradient(${getTrustBarColor(score)} ${(score * 360).toFixed(0)}deg, rgba(30,41,59,0.6) 0deg)`,
              }}
            >
              <div className="w-[108px] h-[108px] rounded-full flex flex-col items-center justify-center text-center" style={{ background: 'var(--bg-card)' }}>
                <span className="text-4xl font-bold text-white">{(score * 100).toFixed(0)}</span>
                <span className="text-xl text-white/70">%</span>
              </div>
            </div>

            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-2xl font-bold" style={{ color: getTrustBarColor(score) }}>
                  {getTrustLabel(score)}
                </h2>
                {quarantined && (
                  <span className="status-badge status-badge--error">
                    <AlertTriangle size={12} /> Quarantined
                  </span>
                )}
              </div>
              <p className="text-slate-400 text-sm mt-1">Based on your update submissions and participation</p>
            </div>
          </div>

          <div className="text-right">
            <div className="flex items-center gap-2 text-slate-400 text-sm">
              <Shield size={16} style={{ color: 'var(--color-success)' }} />
              <span>Trusted Participant</span>
            </div>
          </div>
        </div>

        <div className="mt-8">
          <MetricBar value={score} max={1} colorMode="trust" />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 animate-fade-in" style={{ animationDelay: '0.15s', opacity: 0 }}>
        <div className="glass-card p-5">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{ background: 'var(--color-success-bg)' }}>
              <CheckCircle size={20} style={{ color: 'var(--color-success)' }} />
            </div>
            <h3 className="text-white font-medium text-sm">Participation</h3>
          </div>
          <p className="text-slate-400 text-xs">Consistent participation in training rounds improves your trust score.</p>
        </div>
        <div className="glass-card p-5">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{ background: 'var(--color-info-bg)' }}>
              <Activity size={20} style={{ color: 'var(--color-info)' }} />
            </div>
            <h3 className="text-white font-medium text-sm">Update Quality</h3>
          </div>
          <p className="text-slate-400 text-xs">Submitting high-quality updates with accurate metrics boosts your score.</p>
        </div>
        <div className="glass-card p-5">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{ background: 'var(--color-warning-bg)' }}>
              <TrendingUp size={20} style={{ color: 'var(--color-warning)' }} />
            </div>
            <h3 className="text-white font-medium text-sm">Aggregation</h3>
          </div>
          <p className="text-slate-400 text-xs">Trust score influences weighted aggregation of your model updates.</p>
        </div>
      </div>

      {quarantined && (
        <div className="glass-card p-5 animate-fade-in" style={{ animationDelay: '0.2s', opacity: 0, borderColor: 'var(--color-error-border)' }}>
          <div className="flex items-start gap-4">
            <AlertTriangle size={24} style={{ color: 'var(--color-error)' }} />
            <div>
              <h3 className="text-white font-semibold mb-2">Account Quarantined</h3>
              <p className="text-slate-400 text-sm">
                Your account has been flagged due to low trust score. Please ensure consistent, high-quality updates to restore your score.
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="glass-card p-5 animate-fade-in" style={{ animationDelay: '0.25s', opacity: 0 }}>
        <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
          <Info size={16} className="text-slate-500" /> How Trust Scores Work
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
