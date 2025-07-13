const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;
const NativeClaudeFlow = require('./native');
const ClaudeCLI = require('./claude-cli');

class ClaudeFlowInterface {
  constructor() {
    this.scratchDir = path.join(__dirname, '../../scratch');
    this.visualizationsDir = path.join(__dirname, '../../visualizations');
    
    // Use claude CLI if configured (recommended)
    if (process.env.USE_CLAUDE_CLI === 'true') {
      console.log('Using Claude CLI for interactions');
      this.claudeCLI = new ClaudeCLI();
    }
    // Use native claude-flow if configured
    else if (process.env.CLAUDE_FLOW_MODE === 'native') {
      console.log('Using claude-flow native mode');
      this.nativeFlow = new NativeClaudeFlow();
    }
  }

  async processMessage(message, context = {}) {
    try {
      // Use Claude CLI if available (preferred)
      if (this.claudeCLI) {
        return await this.claudeCLI.processMessage(message, context);
      }
      // Use native claude-flow if available
      else if (this.nativeFlow) {
        // Add available data files to context
        const dataFiles = await this.listDataFiles();
        context.dataFiles = dataFiles;
        
        return await this.nativeFlow.processMessage(message, context);
      }
      
      // Fallback to simulation mode
      let response = '';
      
      const lowerMessage = message.toLowerCase();
      
      if (lowerMessage.includes('hello') || lowerMessage.includes('hi')) {
        response = "Hello! I'm Claude-Flow. I can help you analyze data, create visualizations, and execute Python code. Try uploading some data or asking me to create a visualization!";
      } else if (lowerMessage.includes('analyze') && lowerMessage.includes('data')) {
        const files = await this.listDataFiles();
        if (files.length > 0) {
          response = `I found ${files.length} data files in the system. I can analyze them for you. Which file would you like me to analyze?`;
        } else {
          response = "I don't see any data files yet. Please upload some data using the Data Ingestion tab first.";
        }
      } else if (lowerMessage.includes('visualization') || lowerMessage.includes('plot') || lowerMessage.includes('chart')) {
        response = "I can create visualizations from your data! Upload a JSON file with your data, then ask me to create a specific type of chart (bar, line, scatter) or let me choose the best visualization for your data.";
      } else if (lowerMessage.includes('python') || lowerMessage.includes('code')) {
        response = "I can execute Python code for data analysis and visualization. Just provide the code you'd like me to run, or ask me to write code for a specific task.";
      } else {
        response = `I understand you said: "${message}". I can help with data analysis, visualizations, and Python code execution. What would you like me to do?`;
      }
      
      return {
        success: true,
        response,
        context: { ...context, lastMessage: message }
      };
    } catch (error) {
      throw new Error(`Failed to process message: ${error.message}`);
    }
  }

  async executeCode(code, language = 'python') {
    const filename = `temp_${Date.now()}.${language === 'python' ? 'py' : 'js'}`;
    const filepath = path.join(this.scratchDir, filename);
    
    try {
      await fs.writeFile(filepath, code);
      
      const executor = language === 'python' ? 'python3' : 'node';
      const execProcess = spawn(executor, [filepath], {
        cwd: this.scratchDir
      });

      return new Promise((resolve, reject) => {
        let output = '';
        let error = '';

        execProcess.stdout.on('data', (data) => {
          output += data.toString();
        });

        execProcess.stderr.on('data', (data) => {
          error += data.toString();
        });

        execProcess.on('close', async (code) => {
          await fs.unlink(filepath).catch(() => {});
          
          if (code === 0) {
            resolve({ success: true, output, error });
          } else {
            reject(new Error(`Code execution failed: ${error}`));
          }
        });
      });
    } catch (error) {
      throw new Error(`Failed to execute code: ${error.message}`);
    }
  }

  async listDataFiles() {
    try {
      const uploadsDir = path.join(__dirname, '../../uploads');
      const files = await fs.readdir(uploadsDir);
      return files.filter(f => f.endsWith('.json'));
    } catch (error) {
      return [];
    }
  }

  async listScratchFiles() {
    try {
      const files = await fs.readdir(this.scratchDir);
      return files.filter(f => !f.startsWith('temp_'));
    } catch (error) {
      return [];
    }
  }
}

module.exports = new ClaudeFlowInterface();