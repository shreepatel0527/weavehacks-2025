import React, { useState, useRef, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  visualization?: string;
  code?: string;
  isStreaming?: boolean;
}

interface ChatInterfaceProps {
  onVisualizationGenerated?: (vizId: string) => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ onVisualizationGenerated }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Use WebSocket for real-time communication
  const { connected, sendMessage, lastMessage, streaming, setStreaming } = useWebSocket('ws://localhost:3001/ws');

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingContent]);

  // Handle WebSocket messages
  useEffect(() => {
    if (!lastMessage) return;

    switch (lastMessage.type) {
      case 'connected':
        console.log('Connected to WebSocket:', lastMessage.sessionId);
        break;

      case 'chat_response':
        const assistantMessage: Message = {
          role: 'assistant',
          content: lastMessage.response,
          timestamp: new Date(),
          visualization: lastMessage.visualization,
          code: lastMessage.code
        };
        setMessages(prev => [...prev, assistantMessage]);
        setLoading(false);
        
        if (lastMessage.visualization && onVisualizationGenerated) {
          onVisualizationGenerated(lastMessage.visualization);
        }
        break;

      case 'stream_chunk':
        setStreamingContent(prev => prev + lastMessage.chunk);
        break;

      case 'stream_complete':
        if (streamingContent) {
          const streamedMessage: Message = {
            role: 'assistant',
            content: streamingContent,
            timestamp: new Date(),
            visualization: lastMessage.visualization,
            code: lastMessage.code
          };
          setMessages(prev => [...prev, streamedMessage]);
          setStreamingContent('');
          setLoading(false);
          
          if (lastMessage.visualization && onVisualizationGenerated) {
            onVisualizationGenerated(lastMessage.visualization);
          }
        }
        break;

      case 'typing':
        // Handle typing indicator if needed
        break;

      case 'error':
        const errorMessage: Message = {
          role: 'assistant',
          content: `Error: ${lastMessage.error}`,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
        setLoading(false);
        break;
    }
  }, [lastMessage, streamingContent, onVisualizationGenerated]);

  const handleSendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    setStreamingContent('');

    if (connected) {
      // Use WebSocket for real-time communication
      sendMessage({
        type: 'chat',
        message: userMessage.content,
        context: {
          previousMessages: messages.slice(-5).map(m => ({
            role: m.role,
            content: m.content
          }))
        }
      });
    } else {
      // Fallback to HTTP API
      try {
        const response = await fetch('http://localhost:3001/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            message: userMessage.content,
            context: { 
              previousMessages: messages.slice(-5) 
            }
          })
        });

        const data = await response.json();

        if (data.success) {
          const assistantMessage: Message = {
            role: 'assistant',
            content: data.response || 'I processed your request.',
            timestamp: new Date()
          };
          setMessages(prev => [...prev, assistantMessage]);

          // Check if a visualization was mentioned in the response
          const vizMatch = data.response.match(/viz_([a-f0-9-]+)/);
          if (vizMatch && onVisualizationGenerated) {
            onVisualizationGenerated(vizMatch[1]);
          }
        } else {
          throw new Error(data.error || 'Failed to get response');
        }
      } catch (error) {
        const errorMessage: Message = {
          role: 'assistant',
          content: `Error: ${error instanceof Error ? error.message : 'Something went wrong'}`,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
      } finally {
        setLoading(false);
      }
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Render code blocks nicely
  const renderContent = (content: string) => {
    const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
    const parts = content.split(codeBlockRegex);
    
    return parts.map((part, index) => {
      if (index % 3 === 2) {
        // This is code content
        const language = parts[index - 1] || 'plaintext';
        return (
          <pre key={index} style={{ 
            backgroundColor: '#f4f4f4', 
            padding: '10px', 
            borderRadius: '5px',
            overflow: 'auto'
          }}>
            <code>{part}</code>
          </pre>
        );
      } else if (index % 3 === 0) {
        // This is regular text
        return <span key={index}>{part}</span>;
      }
      return null;
    });
  };

  return (
    <div className="chat-container">
      {!connected && (
        <div className="connection-status" style={{
          backgroundColor: '#fff3cd',
          color: '#856404',
          padding: '10px',
          borderRadius: '5px',
          marginBottom: '10px'
        }}>
          Connecting to WebSocket... (using HTTP fallback)
        </div>
      )}
      <div className="messages-container">
        {messages.length === 0 && (
          <div className="message assistant">
            <div className="message-content">
              Hello! I'm Claude-Flow. I can help you with:
              <ul style={{ textAlign: 'left', marginTop: '10px' }}>
                <li>Answering questions</li>
                <li>Processing data you've uploaded</li>
                <li>Generating visualizations</li>
                <li>Writing and executing Python code</li>
              </ul>
              Try asking me to analyze some data or create a visualization!
            </div>
          </div>
        )}
        
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.role}`}>
            <div className="message-role">
              {message.role === 'user' ? 'You' : 'Claude-Flow'}
            </div>
            <div className="message-content">
              {renderContent(message.content)}
              {message.visualization && (
                <div style={{ marginTop: '10px' }}>
                  <img 
                    src={`http://localhost:3001/api/visualizations/${message.visualization}`}
                    alt="Generated visualization"
                    style={{ maxWidth: '100%', borderRadius: '5px' }}
                  />
                </div>
              )}
            </div>
          </div>
        ))}
        
        {(loading || streamingContent) && (
          <div className="message assistant">
            <div className="message-content">
              {streamingContent || (
                <span className="loading">Claude-Flow is thinking...</span>
              )}
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      <div className="input-container">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message..."
          disabled={loading}
        />
        <button onClick={handleSendMessage} disabled={loading || !input.trim()}>
          Send
        </button>
        {connected && (
          <button 
            onClick={() => setStreaming(!streaming)}
            style={{ marginLeft: '10px', fontSize: '12px' }}
          >
            {streaming ? '🔴 Streaming' : '⚡ Stream'}
          </button>
        )}
      </div>
    </div>
  );
};

export default ChatInterface;