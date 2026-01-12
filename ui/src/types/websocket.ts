// Client → Server messages

export interface StartSessionMessage {
  type: 'start_session';
  base_prompt: string;
}

export interface TextUpdateMessage {
  type: 'text_update';
  full_text: string;
}

export interface SubmitMessage {
  type: 'submit';
}

export type ClientMessage = StartSessionMessage | TextUpdateMessage | SubmitMessage;

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

export interface TextUpdateEvent {
  event: 'text_update';
  current_text: string;
  confirmed_text: string;
  pending_text: string;
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
  | TextUpdateEvent
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
}
