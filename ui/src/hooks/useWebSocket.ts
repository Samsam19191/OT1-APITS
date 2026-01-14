import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  ClientMessage,
  ServerEvent,
  ConnectionStatus,
  TypingSessionState,
} from '../types/websocket';

const WS_URL = 'ws://localhost:8000/ws/session';
const RECONNECT_DELAY_MS = 2000;

interface UseWebSocketReturn {
  status: ConnectionStatus;
  sessionStatus: ConnectionStatus;
  sessionId: string | null;
  sessionState: TypingSessionState;
  generatedText: string;
  metrics: { ttft_ms: number; total_time_ms: number } | null;
  connect: () => void;
  disconnect: () => void;
  startSession: (basePrompt: string) => void;
  sendTextUpdate: (text: string) => void;
  submit: () => void;
}

export function useWebSocket(
  onEvent?: (event: ServerEvent) => void
): UseWebSocketReturn {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const [sessionStatus, setSessionStatus] = useState<ConnectionStatus>('disconnected');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionState, setSessionState] = useState<TypingSessionState>({
    currentText: '',
    confirmedText: '',
    pendingText: '',
    isGenerating: false,
  });
  const [generatedText, setGeneratedText] = useState('');
  const [metrics, setMetrics] = useState<{ ttft_ms: number; total_time_ms: number } | null>(null);

  const send = useCallback((message: ClientMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  const handleMessage = useCallback(
    (event: MessageEvent) => {
      const data = JSON.parse(event.data) as ServerEvent;
      console.log(data);
      onEvent?.(data);

      switch (data.event) {
        case 'connected':
          setSessionId(data.session_id);
          break;

        case 'session_started':
          setSessionStatus('connected');
          setSessionId(data.session_id);
          break;

        case 'text_update':
          setSessionState((prev) => ({
            ...prev,
            currentText: data.current_text,
            confirmedText: data.confirmed_text,
            pendingText: data.pending_text,
          }));
          break;

        case 'submit_start':
          setSessionState((prev) => ({ ...prev, isGenerating: true }));
          setGeneratedText('');
          break;

        case 'generation_start':
          setMetrics({ ttft_ms: data.time_to_first_token_ms, total_time_ms: 0 });
          break;

        case 'generation_token':
          setGeneratedText((prev) => prev + data.token);
          break;

        case 'generation_complete':
          setGeneratedText(data.generated_text);
          setMetrics((prev) => ({
            ttft_ms: prev?.ttft_ms ?? 0,
            total_time_ms: data.total_time_ms,
          }));
          setSessionState((prev) => ({ ...prev, isGenerating: false }));
          break;

        case 'error':
          console.error('Server error:', data.message);
          break;
      }
    },
    [onEvent]
  );

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setStatus('connecting');
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      setStatus('connected');
    };

    ws.onmessage = handleMessage;

    ws.onclose = () => {
      setStatus('disconnected');
      wsRef.current = null;
      reconnectTimeoutRef.current = window.setTimeout(connect, RECONNECT_DELAY_MS);
    };

    ws.onerror = () => {
      setStatus('error');
    };

    wsRef.current = ws;
  }, [handleMessage]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    wsRef.current?.close();
    wsRef.current = null;
    setStatus('disconnected');
    setSessionStatus('disconnected');
  }, []);

  const startSession = useCallback(
    (basePrompt: string) => {
      setSessionState({
        currentText: '',
        confirmedText: '',
        pendingText: '',
        isGenerating: false,
      });
      setGeneratedText('');
      setMetrics(null);
      send({ type: 'start_session', base_prompt: basePrompt });
      setSessionStatus('connecting');
    },
    [send]
  );

  const sendTextUpdate = useCallback(
    (text: string) => {
      send({ type: 'text_update', full_text: text });
    },
    [send]
  );

  const submit = useCallback(() => {
    send({ type: 'submit' });
  }, [send]);

  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    status,
    sessionStatus,
    sessionId,
    sessionState,
    generatedText,
    metrics,
    connect,
    disconnect,
    startSession,
    sendTextUpdate,
    submit,
  };
}
