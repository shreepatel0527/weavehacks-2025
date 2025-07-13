const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;
const { v4: uuidv4 } = require('uuid');

class NativeClaudeFlow {
  constructor() {
    this.scratchDir = path.join(__dirname, '../../scratch');
    this.visualizationsDir = path.join(__dirname, '../../visualizations');
    this.sessions = new Map();
  }

  async processMessage(message, context = {}) {
    const sessionId = context.sessionId || uuidv4();
    
    try {
      // Construct the claude-flow command
      const prompt = this.constructPrompt(message, context);
      
      // Execute claude-flow with the user's existing credentials
      const result = await this.executeClaudeFlow(prompt, sessionId);
      
      // Parse and process the response
      const response = await this.processResponse(result, context);
      
      return {
        success: true,
        response: response.text,
        code: response.code,
        visualization: response.visualization,
        context: { 
          ...context, 
          sessionId,
          lastMessage: message,
          hasCode: !!response.code,
          hasVisualization: !!response.visualization
        }
      };
    } catch (error) {
      console.error('Claude-Flow error:', error);
      throw new Error(`Failed to process message: ${error.message}`);
    }
  }

  constructPrompt(message, context) {
    let prompt = message;
    
    // Add context about available data files
    if (context.dataFiles && context.dataFiles.length > 0) {
      prompt += `\n\nAvailable data files:\n${context.dataFiles.map(f => `- ${f}`).join('\n')}`;
    }
    
    // Add context about code execution capability
    if (message.toLowerCase().includes('code') || message.toLowerCase().includes('python')) {
      prompt += '\n\nNote: You can write and execute Python code. Any generated visualizations will be saved and displayed in the UI.';
    }
    
    // Add instruction for structured responses
    prompt += '\n\nPlease provide clear, actionable responses. If you write code, make sure it\'s complete and executable.';
    
    return prompt;
  }

  async executeClaudeFlow(prompt, sessionId) {
    return new Promise((resolve, reject) => {
      // Try different command options
      let command, args;
      
      // Option 1: Use npx claude-flow@alpha with headless mode
      if (process.env.USE_CLAUDE_FLOW === 'true' || process.env.CLAUDE_FLOW_MODE === 'flow') {
        command = 'npx';
        args = ['claude-flow@alpha', '-p', prompt, '--output-format', 'stream-json'];
      }
      // Option 2: Use claude CLI directly (recommended for simple interactions)
      else if (process.env.USE_CLAUDE_CLI === 'true' || process.env.CLAUDE_FLOW_MODE === 'claude') {
        command = 'claude';
        args = [prompt];  // claude CLI takes prompt as direct argument
      }
      // Option 3: Try npx claude (Claude Code)
      else {
        command = 'npx';
        args = ['@anthropic-ai/claude-code', '-p', prompt];
      }

      console.log(`Executing: ${command} ${args[0]}...`);
      
      const claudeProcess = spawn(command, args, {
        env: { 
          ...process.env,
          // Use system's existing Claude credentials
          HOME: process.env.HOME,
          ANTHROPIC_API_KEY: process.env.ANTHROPIC_API_KEY || '',
          // Ensure npx can find packages
          PATH: process.env.PATH
        },
        cwd: this.scratchDir,
        shell: true
      });

      let output = '';
      let error = '';
      let jsonOutput = '';

      claudeProcess.stdout.on('data', (data) => {
        const chunk = data.toString();
        output += chunk;
        
        // Try to extract JSON if using claude-flow
        if (command === 'npx' && args[0] === 'claude-flow@alpha') {
          const lines = chunk.split('\n');
          for (const line of lines) {
            if (line.trim().startsWith('{') || line.trim().startsWith('[')) {
              jsonOutput += line;
            }
          }
        }
      });

      claudeProcess.stderr.on('data', (data) => {
        error += data.toString();
      });

      claudeProcess.on('error', (err) => {
        reject(new Error(`Failed to spawn process: ${err.message}`));
      });

      claudeProcess.on('close', (code) => {
        // Try to parse JSON output if available
        let finalOutput = output;
        if (jsonOutput) {
          try {
            const parsed = JSON.parse(jsonOutput);
            if (parsed.response || parsed.content) {
              finalOutput = parsed.response || parsed.content;
            }
          } catch (e) {
            // Use raw output if JSON parsing fails
          }
        }
        
        if (code === 0 || finalOutput.length > 0) {
          resolve({
            success: true,
            output: finalOutput,
            error,
            code
          });
        } else {
          reject(new Error(`Command exited with code ${code}: ${error}`));
        }
      });
    });
  }

  async processResponse(result, context) {
    const response = {
      text: result.output,
      code: null,
      visualization: null
    };

    // Extract code blocks
    const codeMatch = result.output.match(/```python\n([\s\S]*?)```/);
    if (codeMatch) {
      response.code = codeMatch[1];
      
      // If code contains visualization commands, execute it
      if (response.code.includes('plt.') || response.code.includes('plotly') || response.code.includes('seaborn')) {
        const vizResult = await this.executeVisualizationCode(response.code);
        if (vizResult.success) {
          response.visualization = vizResult.vizId;
        }
      }
    }

    return response;
  }

  async executeVisualizationCode(code) {
    const vizId = uuidv4();
    const outputPath = path.join(this.visualizationsDir, `viz_${vizId}.png`);
    
    // Ensure matplotlib uses the right backend
    const fullCode = `
import matplotlib
matplotlib.use('Agg')
${code}

# Save the current figure if one exists
import matplotlib.pyplot as plt
if plt.get_fignums():
    plt.savefig('${outputPath}', dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"Visualization saved: ${vizId}")
`;

    try {
      await this.executeCode(fullCode, 'python');
      
      // Check if file was created
      const exists = await fs.access(outputPath).then(() => true).catch(() => false);
      
      return {
        success: exists,
        vizId: exists ? vizId : null
      };
    } catch (error) {
      console.error('Visualization execution error:', error);
      return { success: false, vizId: null };
    }
  }

  async executeCode(code, language = 'python') {
    const filename = `temp_${Date.now()}.${language === 'python' ? 'py' : 'js'}`;
    const filepath = path.join(this.scratchDir, filename);
    
    try {
      await fs.writeFile(filepath, code);
      
      const executor = language === 'python' ? 'python3' : 'node';
      const execProcess = spawn(executor, [filepath], {
        cwd: this.scratchDir,
        env: { ...process.env, PYTHONUNBUFFERED: '1' }
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
}

module.exports = NativeClaudeFlow;