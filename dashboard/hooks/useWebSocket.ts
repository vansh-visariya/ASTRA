import { useState, useEffect, useRef, useCallback } from 'react';
import { WS_URL } from '@/lib/config';
import { getToken } from '@/lib/api/client';

type WsMessageHandler = (message: Record<string, unknown>) => void;

interface UseWebSocketResult {
  isConnected: boolean;
  lastMessage: Record<string, unknown> | null;
  onMessage: (handler: WsMessageHandler) => () => void;
}

export function useWebSocket(): UseWebSocketResult {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<Record<string, unknown> | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const handlersRef = useRef<Set<WsMessageHandler>>(new Set());
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectDelayRef = useRef(1000);
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;
    const token = getToken();
    if (!token) return;

    const url = `${WS_URL}?token=${encodeURIComponent(token)}`;
    const ws = new WebSocket(url);

    ws.onopen = () => {
      if (mountedRef.current) {
        setIsConnected(true);
        reconnectDelayRef.current = 1000;
      }
    };

    ws.onmessage = (event) => {
      if (!mountedRef.current) return;
      try {
        const data = JSON.parse(event.data);
        setLastMessage(data);
        handlersRef.current.forEach((handler) => handler(data));
      } catch {
        // ignore malformed messages
      }
    };

    ws.onclose = () => {
      if (mountedRef.current) {
        setIsConnected(false);
        scheduleReconnect();
      }
    };

    ws.onerror = () => {
      ws.close();
    };

    wsRef.current = ws;
  }, []);

  const scheduleReconnect = useCallback(() => {
    if (!mountedRef.current) return;
    const delay = reconnectDelayRef.current;
    reconnectDelayRef.current = Math.min(delay * 2, 30000);
    reconnectTimeoutRef.current = setTimeout(() => {
      if (mountedRef.current) connect();
    }, delay);
  }, [connect]);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  const onMessage = useCallback((handler: WsMessageHandler) => {
    handlersRef.current.add(handler);
    return () => {
      handlersRef.current.delete(handler);
    };
  }, []);

  return { isConnected, lastMessage, onMessage };
}
