import { useState, useEffect, useCallback } from 'react';
import { wsService, WebSocketMessage } from '../services/websocket';

export interface WebSocketState {
  isConnected: boolean;
  lastMessage: WebSocketMessage | null;
  error: Error | null;
}

export function useWebSocket() {
  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    lastMessage: null,
    error: null
  });

  useEffect(() => {
    const handleConnected = () => setState(prev => ({ ...prev, isConnected: true, error: null }));
    const handleDisconnected = () => setState(prev => ({ ...prev, isConnected: false }));
    const handleError = (error: Error) => setState(prev => ({ ...prev, error }));
    const handleMessage = (message: WebSocketMessage) => setState(prev => ({ ...prev, lastMessage: message }));

    wsService.on('connected', handleConnected);
    wsService.on('disconnected', handleDisconnected);
    wsService.on('error', handleError);
    wsService.on('message', handleMessage);

    wsService.connect();

    return () => {
      wsService.off('connected', handleConnected);
      wsService.off('disconnected', handleDisconnected);
      wsService.off('error', handleError);
      wsService.off('message', handleMessage);
      wsService.disconnect();
    };
  }, []);

  const sendMessage = useCallback((message: WebSocketMessage) => {
    wsService.send(message);
  }, []);

  return {
    ...state,
    sendMessage
  };
} 