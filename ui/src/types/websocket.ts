// Client → Server messages

export interface StartSessionMessage {
  type: 'start_session';
  base_prompt: string;
}

export interface KeystrokeMessage {
  type: 'keystroke';
  char: string;
}

export interface DeleteMessage {
  type: 'delete';
}

export interface SubmitMessage {
  type: 'submit';
}

export type ClientMessage =
  | StartSessionMessage
  | KeystrokeMessage
  | DeleteMessage
  | SubmitMessage;

// Server → Client events

export interface ConnectedEvent {
  event: 'connected';
  session_id: string;
  message: string;
}

export interface SessionStartedEvent {
  event: 'session_started';
  session_id: string;
  base_prompt_length: number;
}

export interface KeystrokeEvent {
  event: 'keystroke';
  current_text: string;
  confirmed_text: string;
  pending_text: string;
}

export interface FlushEvent {
  event: 'flush';
  confirmed_text: string;
  delta: string;
  cache_extended: boolean;
}

export interface DeleteEvent {
  event: 'delete';
  current_text: string;
  confirmed_text: string;
  rollback: boolean;
}

export interface SubmitStartEvent {
  event: 'submit_start';
  final_text: string;
  cache_ready: boolean;
}

export interface GenerationStartEvent {
  event: 'generation_start';
  time_to_first_token_ms: number;
}

export interface GenerationTokenEvent {
  event: 'generation_token';
  token: string;
}

export interface GenerationCompleteEvent {
  event: 'generation_complete';
  generated_text: string;
  total_time_ms: number;
}

export interface ErrorEvent {
  event: 'error';
  message: string;
}

export type ServerEvent =
  | ConnectedEvent
  | SessionStartedEvent
  | KeystrokeEvent
  | FlushEvent
  | DeleteEvent
  | SubmitStartEvent
  | GenerationStartEvent
  | GenerationTokenEvent
  | GenerationCompleteEvent
  | ErrorEvent;

// Utility types

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  metrics?: {
    ttft_ms?: number;
    total_time_ms?: number;
  };
}

export interface TypingSessionState {
  currentText: string;
  confirmedText: string;
  pendingText: string;
  isGenerating: boolean;
  keystrokeCount: number;
}
