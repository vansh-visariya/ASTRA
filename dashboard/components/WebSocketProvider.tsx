'use client';

import { createContext, useContext, ReactNode } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';

type WsMessageHandler = (message: Record<string, unknown>) => void;

interface WsContextValue {
  isConnected: boolean;
  lastMessage: Record<string, unknown> | null;
  onMessage: (handler: WsMessageHandler) => () => void;
  send: (message: Record<string, unknown>) => void;
}

const WsContext = createContext<WsContextValue>({
  isConnected: false,
  lastMessage: null,
  onMessage: () => () => {},
  send: () => {},
});

export function WebSocketProvider({ children }: { children: ReactNode }) {
  const ws = useWebSocket();

  return <WsContext.Provider value={ws}>{children}</WsContext.Provider>;
}

export function useWS() {
  return useContext(WsContext);
}
