import { useEffect, useState, useRef } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import type { ChatMessage } from '../types/websocket';
import './Chat.css';

const DEFAULT_BASE_PROMPT = 'You are a SQL assistant. Convert natural language to SQL.';

export function Chat() {
  const {
    status,
    sessionState,
    generatedText,
    metrics,
    connect,
    startSession,
    sendKeystroke,
    sendDelete,
    submit,
  } = useWebSocket();

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [showDebug, setShowDebug] = useState(true);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const prevInputRef = useRef('');

  // Connect on mount
  useEffect(() => {
    connect();
  }, [connect]);

  // Start session when connected
  useEffect(() => {
    if (status === 'connected') {
      startSession(DEFAULT_BASE_PROMPT);
    }
  }, [status, startSession]);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, generatedText]);

  // Handle input change - send keystrokes
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    const prevValue = prevInputRef.current;

    if (newValue.length > prevValue.length) {
      // Character added
      const newChars = newValue.slice(prevValue.length);
      for (const char of newChars) {
        sendKeystroke(char);
      }
    } else if (newValue.length < prevValue.length) {
      // Characters deleted
      const deletedCount = prevValue.length - newValue.length;
      for (let i = 0; i < deletedCount; i++) {
        sendDelete();
      }
    }

    prevInputRef.current = newValue;
    setInputValue(newValue);
  };

  // Handle submit
  const handleSubmit = () => {
    if (!inputValue.trim() || sessionState.isGenerating) return;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: inputValue,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);

    submit();
    setInputValue('');
    prevInputRef.current = '';
  };

  // Handle Enter key
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // Add assistant message when generation completes
  useEffect(() => {
    if (!sessionState.isGenerating && generatedText && messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      if (lastMessage.role === 'user') {
        const assistantMessage: ChatMessage = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: generatedText,
          timestamp: new Date(),
          metrics: metrics ?? undefined,
        };
        setMessages((prev) => [...prev, assistantMessage]);
        startSession(DEFAULT_BASE_PROMPT); // Reset for next query
      }
    }
  }, [sessionState.isGenerating, generatedText, messages, metrics, startSession]);

  return (
    <div className="chat-container">
      <header className="chat-header">
        <h1>Text-to-SQL</h1>
        <div className="header-controls">
          <span className={`status-badge ${status}`}>{status}</span>
          <button
            className="debug-toggle"
            onClick={() => setShowDebug(!showDebug)}
          >
            {showDebug ? 'Hide Debug' : 'Show Debug'}
          </button>
        </div>
      </header>

      <div className="chat-body">
        <main className="messages-container">
          {messages.length === 0 && (
            <div className="empty-state">
              <p>Ask a question in natural language to generate SQL</p>
              <p className="example">Example: "Show all patients older than 50"</p>
            </div>
          )}

          {messages.map((msg) => (
            <div key={msg.id} className={`message ${msg.role}`}>
              <div className="message-content">
                {msg.role === 'assistant' ? (
                  <pre><code>{msg.content}</code></pre>
                ) : (
                  <p>{msg.content}</p>
                )}
              </div>
              {msg.metrics && (
                <div className="message-metrics">
                  TTFT: {msg.metrics.ttft_ms?.toFixed(0)}ms | Total: {msg.metrics.total_time_ms?.toFixed(0)}ms
                </div>
              )}
            </div>
          ))}

          {sessionState.isGenerating && (
            <div className="message assistant generating">
              <div className="message-content">
                <pre><code>{generatedText || 'Generating...'}</code></pre>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </main>

        {showDebug && (
          <aside className="debug-panel">
            <h3>Debug</h3>
            <div className="debug-item">
              <label>Keystrokes:</label>
              <span>{sessionState.keystrokeCount}</span>
            </div>
            <div className="debug-item">
              <label>Current text:</label>
              <span className="mono">{sessionState.currentText || '(empty)'}</span>
            </div>
            <div className="debug-item">
              <label>Confirmed:</label>
              <span className="mono confirmed">{sessionState.confirmedText || '(none)'}</span>
            </div>
            <div className="debug-item">
              <label>Pending:</label>
              <span className="mono pending">{sessionState.pendingText || '(none)'}</span>
            </div>
            {metrics && (
              <>
                <div className="debug-item">
                  <label>TTFT:</label>
                  <span>{metrics.ttft_ms.toFixed(0)}ms</span>
                </div>
                <div className="debug-item">
                  <label>Total time:</label>
                  <span>{metrics.total_time_ms.toFixed(0)}ms</span>
                </div>
              </>
            )}
          </aside>
        )}
      </div>

      <footer className="chat-footer">
        <textarea
          ref={inputRef}
          value={inputValue}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder="Describe your SQL query in natural language..."
          disabled={status !== 'connected'}
          rows={2}
        />
        <button
          onClick={handleSubmit}
          disabled={!inputValue.trim() || sessionState.isGenerating || status !== 'connected'}
        >
          {sessionState.isGenerating ? 'Generating...' : 'Send'}
        </button>
      </footer>
    </div>
  );
}
