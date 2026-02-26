'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/components/AuthContext';
import { Bell, Check, CheckCheck, Trash2, AlertCircle, Info, CheckCircle, AlertTriangle } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Notification {
  id: number;
  type: string;
  priority: string;
  title: string;
  message: string;
  group_id: string | null;
  data: Record<string, any>;
  created_at: string;
  read: boolean;
}

export default function ClientNotificationsPage() {
  const { token } = useAuth();
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'unread'>('all');

  const fetchNotifications = async () => {
    if (!token) return;
    setLoading(true);
    
    try {
      const res = await fetch(`${API_URL}/api/notifications?limit=50&unread_only=${filter === 'unread'}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const data = await res.json();
      setNotifications(data.notifications || []);
    } catch (e) {
      console.error('Failed to fetch notifications:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchNotifications();
  }, [token, filter]);

  const markAsRead = async (id: number) => {
    if (!token) return;
    
    try {
      await fetch(`${API_URL}/api/notifications/${id}/read`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      
      setNotifications(prev => 
        prev.map(n => n.id === id ? { ...n, read: true } : n)
      );
    } catch (e) {
      console.error('Failed to mark as read:', e);
    }
  };

  const markAllAsRead = async () => {
    if (!token) return;
    
    try {
      await fetch(`${API_URL}/api/notifications/read-all`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      
      setNotifications(prev => prev.map(n => ({ ...n, read: true })));
    } catch (e) {
      console.error('Failed to mark all as read:', e);
    }
  };

  const getIcon = (priority: string, type: string) => {
    switch (priority) {
      case 'error':
        return <AlertCircle className="text-red-400" size={20} />;
      case 'warning':
        return <AlertTriangle className="text-yellow-400" size={20} />;
      case 'success':
        return <CheckCircle className="text-green-400" size={20} />;
      default:
        return <Info className="text-blue-400" size={20} />;
    }
  };

  const getPriorityBg = (priority: string) => {
    switch (priority) {
      case 'error':
        return 'bg-red-900/20 border-red-800';
      case 'warning':
        return 'bg-yellow-900/20 border-yellow-800';
      case 'success':
        return 'bg-green-900/20 border-green-800';
      default:
        return 'bg-blue-900/20 border-blue-800';
    }
  };

  const unreadCount = notifications.filter(n => !n.read).length;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Notifications</h1>
          <p className="text-gray-400">
            {unreadCount > 0 ? `${unreadCount} unread notifications` : 'All caught up!'}
          </p>
        </div>
        <div className="flex items-center gap-3">
          {unreadCount > 0 && (
            <button
              onClick={markAllAsRead}
              className="flex items-center gap-2 px-4 py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-700 transition"
            >
              <CheckCheck size={16} /> Mark all as read
            </button>
          )}
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex gap-2 border-b border-gray-800 pb-2">
        <button
          onClick={() => setFilter('all')}
          className={`px-4 py-2 rounded-t-lg text-sm font-medium transition ${
            filter === 'all'
              ? 'bg-emerald-600 text-white'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          All
        </button>
        <button
          onClick={() => setFilter('unread')}
          className={`px-4 py-2 rounded-t-lg text-sm font-medium transition ${
            filter === 'unread'
              ? 'bg-emerald-600 text-white'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          Unread
          {unreadCount > 0 && (
            <span className="ml-2 px-2 py-0.5 bg-red-500 text-white text-xs rounded-full">
              {unreadCount}
            </span>
          )}
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : notifications.length === 0 ? (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-12 text-center">
          <Bell className="mx-auto mb-4 text-gray-600" size={48} />
          <h3 className="text-lg font-medium text-white mb-2">No Notifications</h3>
          <p className="text-gray-400">
            {filter === 'unread' 
              ? "You've read all your notifications" 
              : "No notifications yet - check back later"}
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {notifications.map((notification) => (
            <div
              key={notification.id}
              className={`relative p-4 rounded-xl border ${
                notification.read 
                  ? 'bg-gray-900/50 border-gray-800' 
                  : getPriorityBg(notification.priority)
              }`}
            >
              <div className="flex items-start gap-4">
                <div className="shrink-0 mt-1">
                  {getIcon(notification.priority, notification.type)}
                </div>
                
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <h3 className="text-white font-medium">{notification.title}</h3>
                      <p className="text-gray-400 text-sm mt-1">{notification.message}</p>
                    </div>
                    
                    {!notification.read && (
                      <button
                        onClick={() => markAsRead(notification.id)}
                        className="shrink-0 p-2 text-gray-500 hover:text-white hover:bg-gray-800 rounded-lg transition"
                        title="Mark as read"
                      >
                        <Check size={16} />
                      </button>
                    )}
                  </div>
                  
                  <div className="flex items-center gap-4 mt-3 text-xs text-gray-500">
                    <span>{new Date(notification.created_at).toLocaleString()}</span>
                    {notification.group_id && (
                      <span className="text-emerald-400">Group: {notification.group_id}</span>
                    )}
                    <span className="uppercase">{notification.type}</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
