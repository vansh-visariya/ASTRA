'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/components/AuthContext';
import { Layers, Clock, RefreshCw, Filter } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://10.146.11.202:8000';

interface Log {
  timestamp: number;
  type: string;
  message: string;
  group_id: string | null;
  details: Record<string, any>;
}

export default function LogsPage() {
  const { token } = useAuth();
  const [logs, setLogs] = useState<Log[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<string | null>(null);

  const fetchLogs = async () => {
    try {
      const url = filter 
        ? `${API_URL}/api/logs?event_type=${filter}` 
        : `${API_URL}/api/logs`;
      const res = await fetch(url, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setLogs(data.logs || []);
      }
    } catch (e) {
      console.error('Failed to fetch logs:', e);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchLogs();
    const interval = setInterval(fetchLogs, 2000);
    return () => clearInterval(interval);
  }, [token, filter]);

  const formatTime = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleTimeString();
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'training_started': return 'text-green-400';
      case 'aggregation': return 'text-blue-400';
      case 'client_joined': return 'text-purple-400';
      case 'client_rejected': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getTypeBg = (type: string) => {
    switch (type) {
      case 'training_started': return 'bg-green-900/20 border-green-800';
      case 'aggregation': return 'bg-blue-900/20 border-blue-800';
      case 'client_joined': return 'bg-purple-900/20 border-purple-800';
      case 'client_rejected': return 'bg-red-900/20 border-red-800';
      default: return 'bg-gray-900/20 border-gray-800';
    }
  };

  const eventTypes = ['training_started', 'aggregation', 'client_joined', 'client_rejected'];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Event Logs</h1>
          <p className="text-gray-400">Server events and training history</p>
        </div>
        <div className="flex gap-3">
          <select
            value={filter || ''}
            onChange={(e) => setFilter(e.target.value || null)}
            className="bg-gray-900 border border-gray-800 rounded-lg px-4 py-2 text-white"
          >
            <option value="">All Events</option>
            {eventTypes.map(type => (
              <option key={type} value={type}>{type.replace('_', ' ')}</option>
            ))}
          </select>
          <button 
            onClick={fetchLogs} 
            className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition"
          >
            <RefreshCw size={18} className="text-gray-400" />
          </button>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : logs.length === 0 ? (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-12 text-center">
          <Clock size={48} className="mx-auto text-gray-600 mb-4" />
          <h3 className="text-white font-semibold mb-2">No logs yet</h3>
          <p className="text-gray-400">Events will appear here when training starts</p>
        </div>
      ) : (
        <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
          <div className="max-h-[600px] overflow-y-auto">
            {logs.map((log, idx) => (
              <div 
                key={idx} 
                className={`p-4 border-b border-gray-800 ${getTypeBg(log.type)}`}
              >
                <div className="flex items-start gap-4">
                  <div className="text-gray-500 text-sm font-mono min-w-[80px]">
                    {formatTime(log.timestamp)}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`text-xs font-medium uppercase ${getTypeColor(log.type)}`}>
                        {log.type.replace('_', ' ')}
                      </span>
                      {log.group_id && (
                        <span className="text-xs text-gray-500">
                          {log.group_id}
                        </span>
                      )}
                    </div>
                    <p className="text-white">{log.message}</p>
                    {log.details && Object.keys(log.details).length > 0 && (
                      <pre className="text-gray-500 text-xs mt-1 font-mono">
                        {JSON.stringify(log.details, null, 2)}
                      </pre>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
