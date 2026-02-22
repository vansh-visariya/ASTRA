'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/components/AuthContext';
import { Shield, Activity, TrendingUp, AlertTriangle, CheckCircle, XCircle, Info } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface TrustData {
  score: number;
  quarantined: boolean;
  group_id: string;
}

export default function ClientTrustPage() {
  const { token, user } = useAuth();
  const [trustData, setTrustData] = useState<TrustData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchTrustScore = async () => {
      if (!token || !user) return;
      
      try {
        const res = await fetch(`${API_URL}/api/trust/scores/${user.id}`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        const data = await res.json();
        setTrustData({
          score: data.score || 1.0,
          quarantined: data.score < 0.35,
          group_id: data.group_id || 'default'
        });
      } catch (e) {
        console.error('Failed to fetch trust score:', e);
        setTrustData({ score: 1.0, quarantined: false, group_id: 'default' });
      } finally {
        setLoading(false);
      }
    };
    
    fetchTrustScore();
  }, [token, user]);

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-400';
    if (score >= 0.6) return 'text-yellow-400';
    if (score >= 0.4) return 'text-orange-400';
    return 'text-red-400';
  };

  const getScoreBg = (score: number) => {
    if (score >= 0.8) return 'from-green-600 to-emerald-600';
    if (score >= 0.6) return 'from-yellow-600 to-orange-600';
    if (score >= 0.4) return 'from-orange-600 to-red-600';
    return 'from-red-600 to-red-800';
  };

  const getScoreLabel = (score: number) => {
    if (score >= 0.8) return 'Excellent';
    if (score >= 0.6) return 'Good';
    if (score >= 0.4) return 'Fair';
    return 'Low';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Trust Score</h1>
        <p className="text-gray-400">Your reliability rating in the federated network</p>
      </div>

      {/* Main Score Card */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-6">
            <div className={`w-32 h-32 rounded-full bg-gradient-to-br ${getScoreBg(trustData?.score || 1)} flex items-center justify-center`}>
              <div className="text-center">
                <span className="text-4xl font-bold text-white">
                  {((trustData?.score || 1) * 100).toFixed(0)}
                </span>
                <span className="text-xl text-white/70">%</span>
              </div>
            </div>
            
            <div>
              <div className="flex items-center gap-2">
                <h2 className={`text-2xl font-bold ${getScoreColor(trustData?.score || 1)}`}>
                  {getScoreLabel(trustData?.score || 1)}
                </h2>
                {trustData?.quarantined && (
                  <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-red-500/20 text-red-400 text-xs">
                    <AlertTriangle size={12} /> Quarantined
                  </span>
                )}
              </div>
              <p className="text-gray-400 mt-1">
                Based on your update submissions and participation
              </p>
            </div>
          </div>
          
          <div className="text-right">
            <div className="flex items-center gap-2 text-gray-400 text-sm">
              <Shield size={16} />
              <span>Trusted Participant</span>
            </div>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mt-8">
          <div className="flex justify-between text-sm text-gray-400 mb-2">
            <span>0%</span>
            <span>Trust Score</span>
            <span>100%</span>
          </div>
          <div className="h-4 bg-gray-800 rounded-full overflow-hidden">
            <div 
              className={`h-full bg-gradient-to-r ${getScoreBg(trustData?.score || 1)} transition-all duration-500`}
              style={{ width: `${(trustData?.score || 1) * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
              <CheckCircle className="text-green-400" size={20} />
            </div>
            <h3 className="text-white font-medium">Participation</h3>
          </div>
          <p className="text-gray-400 text-sm">
            Consistent participation in training rounds improves your trust score.
          </p>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
              <Activity className="text-blue-400" size={20} />
            </div>
            <h3 className="text-white font-medium">Update Quality</h3>
          </div>
          <p className="text-gray-400 text-sm">
            Submitting high-quality updates with accurate metrics boosts your score.
          </p>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-yellow-500/20 rounded-lg flex items-center justify-center">
              <TrendingUp className="text-yellow-400" size={20} />
            </div>
            <h3 className="text-white font-medium">Aggregation</h3>
          </div>
          <p className="text-gray-400 text-sm">
            Trust score influences weighted aggregation of your model updates.
          </p>
        </div>
      </div>

      {/* Quarantine Warning */}
      {trustData?.quarantined && (
        <div className="bg-red-900/20 border border-red-800 rounded-xl p-6">
          <div className="flex items-start gap-4">
            <AlertTriangle className="text-red-400 shrink-0" size={24} />
            <div>
              <h3 className="text-white font-semibold mb-2">Account Quarantined</h3>
              <p className="text-gray-400 text-sm">
                Your account has been flagged due to low trust score. This may limit your participation 
                in training rounds. Please ensure consistent, high-quality updates to restore your score.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* How it Works */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Info size={20} className="text-gray-400" />
          How Trust Scores Work
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-gray-400">
          <div>
            <h4 className="text-white font-medium mb-2">Score Calculation</h4>
            <ul className="space-y-2">
              <li>• Based on cosine similarity of your updates to the global model</li>
              <li>• Updated after each aggregation round</li>
              <li>• Range: 0.0 to 1.0 (higher is better)</li>
            </ul>
          </div>
          <div>
            <h4 className="text-white font-medium mb-2">Impact</h4>
            <ul className="space-y-2">
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
