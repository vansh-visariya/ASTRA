'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/components/AuthContext';
import { TrainFront, Play, Pause, Square, Download, Upload, Activity, CheckCircle, XCircle, Loader2 } from 'lucide-react';

interface TrainingSession {
  group_id: string;
  status: 'idle' | 'training' | 'paused' | 'completed';
  current_round: number;
  accuracy: number;
  loss: number;
}

export default function ClientTrainingPage() {
  const { token, user } = useAuth();
  const [session, setSession] = useState<TrainingSession | null>(null);
  const [training, setTraining] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch current training session
    const fetchSession = async () => {
      if (!token) return;
      
      try {
        // This would normally fetch from API
        setSession({
          group_id: 'group_a',
          status: 'idle',
          current_round: 0,
          accuracy: 0,
          loss: 0
        });
      } catch (e) {
        console.error('Failed to fetch session:', e);
      } finally {
        setLoading(false);
      }
    };
    
    fetchSession();
  }, [token]);

  const startTraining = async () => {
    if (!token || !session) return;
    
    setTraining(true);
    setSession(prev => prev ? { ...prev, status: 'training' } : null);
    addLog('Starting local training...');
    
    // Simulate training
    for (let round = 1; round <= 5; round++) {
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      setSession(prev => prev ? {
        ...prev,
        current_round: round,
        accuracy: Math.random() * 0.3 + 0.6,
        loss: Math.random() * 0.5 + 0.1
      } : null);
      
      addLog(`Round ${round}: accuracy=${(Math.random() * 0.3 + 0.6).toFixed(4)}, loss=${(Math.random() * 0.5 + 0.1).toFixed(4)}`);
    }
    
    setTraining(false);
    setSession(prev => prev ? { ...prev, status: 'completed' } : null);
    addLog('Training completed. Uploading update to server...');
  };

  const pauseTraining = () => {
    setTraining(false);
    setSession(prev => prev ? { ...prev, status: 'paused' } : null);
    addLog('Training paused');
  };

  const stopTraining = () => {
    setTraining(false);
    setSession(prev => prev ? { ...prev, status: 'idle', current_round: 0 } : null);
    addLog('Training stopped');
  };

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev.slice(-99), `[${timestamp}] ${message}`]);
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
        <h1 className="text-2xl font-bold text-white">Training</h1>
        <p className="text-gray-400">Local model training and update submission</p>
      </div>

      {/* Status Card */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-lg font-semibold text-white">
              {session?.group_id || 'No Active Group'}
            </h2>
            <p className="text-gray-400 text-sm">
              Status: <span className="capitalize text-emerald-400">{session?.status || 'idle'}</span>
            </p>
          </div>
          
          <div className="flex gap-2">
            {!training && session?.status === 'idle' && (
              <button
                onClick={startTraining}
                className="flex items-center gap-2 px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition"
              >
                <Play size={16} /> Start Training
              </button>
            )}
            
            {training && (
              <button
                onClick={pauseTraining}
                className="flex items-center gap-2 px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 transition"
              >
                <Pause size={16} /> Pause
              </button>
            )}
            
            {session?.status === 'paused' && (
              <>
                <button
                  onClick={startTraining}
                  className="flex items-center gap-2 px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition"
                >
                  <Play size={16} /> Resume
                </button>
                <button
                  onClick={stopTraining}
                  className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition"
                >
                  <Square size={16} /> Stop
                </button>
              </>
            )}
          </div>
        </div>

        {/* Training Progress */}
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-gray-400 text-sm">Current Round</p>
            <p className="text-2xl font-bold text-white mt-1">
              {session?.current_round || 0}
            </p>
          </div>
          
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-gray-400 text-sm">Local Accuracy</p>
            <p className="text-2xl font-bold text-white mt-1">
              {session?.accuracy ? `${(session.accuracy * 100).toFixed(1)}%` : '-'}
            </p>
          </div>
          
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-gray-400 text-sm">Local Loss</p>
            <p className="text-2xl font-bold text-white mt-1">
              {session?.loss ? session.loss.toFixed(4) : '-'}
            </p>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mt-6">
          <div className="flex justify-between text-sm text-gray-400 mb-2">
            <span>Progress</span>
            <span>{session?.current_round || 0} / 5 rounds</span>
          </div>
          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
            <div 
              className="h-full bg-emerald-500 transition-all duration-300"
              style={{ width: `${((session?.current_round || 0) / 5) * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
              <Download className="text-blue-400" size={20} />
            </div>
            <div>
              <h3 className="text-white font-medium">Download Global Model</h3>
              <p className="text-gray-400 text-sm">Get latest model weights</p>
            </div>
          </div>
          <button className="w-full px-4 py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-700 transition">
            Download Model
          </button>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
              <Upload className="text-purple-400" size={20} />
            </div>
            <div>
              <h3 className="text-white font-medium">Submit Update</h3>
              <p className="text-gray-400 text-sm">Upload your model delta</p>
            </div>
          </div>
          <button className="w-full px-4 py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-700 transition">
            Submit Update
          </button>
        </div>
      </div>

      {/* Training Logs */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h3 className="text-white font-medium mb-4 flex items-center gap-2">
          <Activity size={18} className="text-gray-400" />
          Training Logs
        </h3>
        
        <div className="bg-gray-950 rounded-lg p-4 h-64 overflow-auto font-mono text-sm">
          {logs.length === 0 ? (
            <p className="text-gray-600">No training logs yet</p>
          ) : (
            logs.map((log, i) => (
              <div key={i} className="text-gray-300">
                {log}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
