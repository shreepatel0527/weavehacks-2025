const WebSocket = require('ws');
const claudeFlow = require('./claude-flow');
const { v4: uuidv4 } = require('uuid');

class WebSocketHandler {
  constructor(server) {
    this.wss = new WebSocket.Server({ server, path: '/ws' });
    this.clients = new Map();
    
    this.setupWebSocket();
  }

  setupWebSocket() {
    this.wss.on('connection', (ws, req) => {
      const clientId = uuidv4();
      const client = {
        id: clientId,
        ws,
        sessionId: uuidv4(),
        streaming: false
      };
      
      this.clients.set(clientId, client);
      console.log(`WebSocket client connected: ${clientId}`);
      
      // Send welcome message
      this.sendToClient(clientId, {
        type: 'connected',
        clientId,
        sessionId: client.sessionId
      });
      
      // Handle messages
      ws.on('message', async (message) => {
        try {
          const data = JSON.parse(message);
          await this.handleMessage(clientId, data);
        } catch (error) {
          console.error('WebSocket message error:', error);
          this.sendToClient(clientId, {
            type: 'error',
            error: error.message
          });
        }
      });
      
      // Handle disconnect
      ws.on('close', () => {
        console.log(`WebSocket client disconnected: ${clientId}`);
        this.clients.delete(clientId);
      });
      
      // Handle errors
      ws.on('error', (error) => {
        console.error(`WebSocket error for client ${clientId}:`, error);
      });
    });
  }

  async handleMessage(clientId, data) {
    const client = this.clients.get(clientId);
    if (!client) return;

    switch (data.type) {
      case 'chat':
        await this.handleChatMessage(clientId, data);
        break;
        
      case 'execute':
        await this.handleCodeExecution(clientId, data);
        break;
        
      case 'voice':
        await this.handleVoiceMessage(clientId, data);
        break;
        
      case 'stream_start':
        client.streaming = true;
        break;
        
      case 'stream_stop':
        client.streaming = false;
        break;
        
      default:
        this.sendToClient(clientId, {
          type: 'error',
          error: `Unknown message type: ${data.type}`
        });
    }
  }

  async handleChatMessage(clientId, data) {
    const client = this.clients.get(clientId);
    
    try {
      // Send typing indicator
      this.sendToClient(clientId, {
        type: 'typing',
        status: 'start'
      });

      // Process with claude-flow
      const result = await claudeFlow.processMessage(data.message, {
        sessionId: client.sessionId,
        ...data.context
      });

      // Send response
      if (client.streaming && result.response.length > 100) {
        // Stream response in chunks
        await this.streamResponse(clientId, result);
      } else {
        // Send complete response
        this.sendToClient(clientId, {
          type: 'chat_response',
          response: result.response,
          context: result.context,
          visualization: result.visualization,
          code: result.code
        });
      }

      // Send typing indicator stop
      this.sendToClient(clientId, {
        type: 'typing',
        status: 'stop'
      });
    } catch (error) {
      this.sendToClient(clientId, {
        type: 'error',
        error: error.message
      });
    }
  }

  async streamResponse(clientId, result) {
    const chunkSize = 50; // characters per chunk
    const text = result.response;
    
    for (let i = 0; i < text.length; i += chunkSize) {
      const chunk = text.slice(i, i + chunkSize);
      
      this.sendToClient(clientId, {
        type: 'stream_chunk',
        chunk,
        progress: (i + chunkSize) / text.length,
        isComplete: i + chunkSize >= text.length
      });
      
      // Small delay between chunks for effect
      await new Promise(resolve => setTimeout(resolve, 50));
    }

    // Send final complete message with any additional data
    this.sendToClient(clientId, {
      type: 'stream_complete',
      visualization: result.visualization,
      code: result.code,
      context: result.context
    });
  }

  async handleCodeExecution(clientId, data) {
    try {
      this.sendToClient(clientId, {
        type: 'execution_start',
        language: data.language || 'python'
      });

      const result = await claudeFlow.executeCode(data.code, data.language);

      this.sendToClient(clientId, {
        type: 'execution_result',
        output: result.output,
        error: result.error,
        success: result.success
      });
    } catch (error) {
      this.sendToClient(clientId, {
        type: 'execution_error',
        error: error.message
      });
    }
  }

  async handleVoiceMessage(clientId, data) {
    // Placeholder for voice handling
    // This will be implemented when voice features are added
    this.sendToClient(clientId, {
      type: 'voice_response',
      message: 'Voice features coming soon!'
    });
  }

  sendToClient(clientId, data) {
    const client = this.clients.get(clientId);
    if (client && client.ws.readyState === WebSocket.OPEN) {
      client.ws.send(JSON.stringify(data));
    }
  }

  broadcast(data) {
    this.clients.forEach((client, clientId) => {
      this.sendToClient(clientId, data);
    });
  }
}

module.exports = WebSocketHandler;