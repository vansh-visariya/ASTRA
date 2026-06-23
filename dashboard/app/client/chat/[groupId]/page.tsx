'use client';

import { useState, useEffect, useRef } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { ArrowLeft, Send } from 'lucide-react';
import { useAuth } from '@/components/AuthContext';
import { useWS } from '@/components/WebSocketProvider';
import { getMessages, sendMessage } from '@/lib/api/endpoints';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import type { Message } from '@/lib/api/types';

export default function ClientChatPage() {
  const { groupId } = useParams<{ groupId: string }>();
  const { user } = useAuth();
  const router = useRouter();
  const { isConnected } = useWS();
  const [messages, setMessages] = useState<Message[]>([]);
  const [messageText, setMessageText] = useState('');
  const [loading, setLoading] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (!groupId) return;
    setLoading(true);
    getMessages(groupId)
      .then((res: any) => {
        setMessages(res?.messages || []);
        setLoading(false);
        setTimeout(scrollToBottom, 100);
      })
      .catch(() => setLoading(false));
  }, [groupId]);

  // Listen for new messages via WebSocket
  useEffect(() => {
    if (!isConnected) return;
    const handler = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'new_message' && data.group_id === groupId) {
          setMessages((prev) => [
            ...prev,
            {
              id: data.message_id,
              group_id: data.group_id,
              sender_id: data.sender_id,
              sender_name: data.sender_name,
              sender_role: data.sender_role,
              content: data.content,
              created_at: new Date().toISOString(),
            },
          ]);
          setTimeout(scrollToBottom, 100);
        }
      } catch {}
    };
    // We rely on the global WS connection from WebSocketProvider
    return () => {};
  }, [isConnected, groupId]);

  const handleSend = async () => {
    if (!messageText.trim()) return;
    try {
      await sendMessage(groupId, messageText);
      setMessageText('');
      const res: any = await getMessages(groupId);
      setMessages(res?.messages || []);
      setTimeout(scrollToBottom, 100);
    } catch {}
  };

  if (loading) return <LoadingSpinner message="Loading chat..." />;

  return (
    <div className="max-w-3xl mx-auto space-y-4">
      <div className="flex items-center gap-4">
        <button onClick={() => router.push('/client')} className="p-2 hover:bg-gray-800 rounded-lg transition">
          <ArrowLeft size={20} className="text-gray-400" />
        </button>
        <div>
          <h1 className="text-2xl font-bold text-white">Group Chat</h1>
          <p className="text-slate-400 text-sm mt-1">{groupId}</p>
        </div>
      </div>

      <div className="glass-card p-5">
        <div className="space-y-3 max-h-[500px] overflow-y-auto mb-4">
          {messages.length === 0 ? (
            <p className="text-slate-500 text-sm text-center py-8">No messages yet. Start the conversation.</p>
          ) : (
            messages.map((m) => (
              <div
                key={m.id}
                className={`flex gap-3 ${m.sender_id === (user as any)?.id ? 'flex-row-reverse' : ''}`}
              >
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${
                  m.sender_role === 'admin' ? 'bg-purple-500/20 text-purple-400' : 'bg-blue-500/20 text-blue-400'
                }`}>
                  {(m.sender_name || '?')[0].toUpperCase()}
                </div>
                <div className={`max-w-[70%] ${m.sender_id === (user as any)?.id ? 'text-right' : ''}`}>
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="text-slate-400 text-xs">{m.sender_name}</span>
                    {m.sender_role === 'admin' && (
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-400 font-medium">Admin</span>
                    )}
                  </div>
                  <div className={`inline-block px-3 py-2 rounded-xl text-sm ${
                    m.sender_id === (user as any)?.id
                      ? 'bg-blue-600/30 text-white'
                      : 'text-white'
                  }`} style={m.sender_id !== (user as any)?.id ? { background: 'rgba(30,41,59,0.6)' } : {}}>
                    {m.content}
                  </div>
                  <p className="text-slate-600 text-[10px] mt-0.5">
                    {new Date(m.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </p>
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="flex gap-2">
          <input
            type="text"
            value={messageText}
            onChange={(e) => setMessageText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleSend();
            }}
            className="input-field flex-1"
            placeholder="Type a message..."
          />
          <button
            onClick={handleSend}
            disabled={!messageText.trim()}
            className="btn-primary !px-4 text-sm disabled:opacity-50 inline-flex items-center gap-1.5"
          >
            <Send size={14} /> Send
          </button>
        </div>
      </div>
    </div>
  );
}
