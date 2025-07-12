const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;
const { v4: uuidv4 } = require('uuid');

class ClaudeCLI {
  constructor() {
    this.scratchDir = path.join(__dirname, '../../scratch');
    this.visualizationsDir = path.join(__dirname, '../../visualizations');
    this.dataDir = path.join(__dirname, '../../uploads');
  }

  async processMessage(message, context = {}) {
    const sessionId = context.sessionId || uuidv4();
    
    try {
      // Construct enhanced prompt with context
      const prompt = await this.constructPrompt(message, context);
      
      // Execute claude CLI
      const result = await this.executeClaude(prompt);
      
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
      console.error('Claude CLI error:', error);
      throw new Error(`Failed to process message: ${error.message}`);
    }
  }

  async constructPrompt(message, context) {
    let prompt = message;
    
    // Add working directory context
    prompt = `Current working directory: ${this.scratchDir}\n\n${prompt}`;
    
    // Add available data files context
    const dataFiles = await this.listDataFiles();
    if (dataFiles.length > 0) {
      prompt += `\n\nAvailable data files in ${this.dataDir}:\n${dataFiles.map(f => `- ${f}`).join('\n')}`;
    }
    
    // Add visualization context
    if (message.toLowerCase().includes('visualiz') || message.toLowerCase().includes('plot') || message.toLowerCase().includes('chart')) {
      prompt += `\n\nWhen creating visualizations:
- Use matplotlib with backend 'Agg' for headless operation
- Save plots to: ${this.visualizationsDir}/viz_[unique_id].png
- Include the visualization ID in your response`;
    }
    
    // Add code execution context
    if (message.toLowerCase().includes('code') || message.toLowerCase().includes('python')) {
      prompt += '\n\nYou can write and execute Python code. Make sure code is complete and executable.';
    }
    
    return prompt;
  }

  async executeClaude(prompt) {
    return new Promise((resolve, reject) => {
      console.log('Executing claude CLI...');
      
      // Use claude CLI with the prompt
      const claudeProcess = spawn('claude', [prompt], {
        env: { 
          ...process.env,
          // Ensure HOME is set for credential access
          HOME: process.env.HOME,
          // Set non-interactive mode
          CLAUDE_INTERACTIVE: 'false'
        },
        cwd: this.scratchDir
      });

      let output = '';
      let error = '';

      claudeProcess.stdout.on('data', (data) => {
        const chunk = data.toString();
        output += chunk;
        // Log chunks for debugging
        if (process.env.NODE_ENV === 'development') {
          console.log('Claude output chunk:', chunk.substring(0, 100) + '...');
        }
      });

      claudeProcess.stderr.on('data', (data) => {
        error += data.toString();
      });

      claudeProcess.on('error', (err) => {
        console.error('Failed to spawn claude process:', err);
        reject(new Error(`Failed to spawn claude process: ${err.message}`));
      });

      claudeProcess.on('close', (code) => {
        console.log(`Claude process exited with code ${code}`);
        
        if (code === 0 && output.length > 0) {
          resolve({
            success: true,
            output: output.trim(),
            error,
            code
          });
        } else if (output.length > 0) {
          // Sometimes claude returns non-zero but still has output
          resolve({
            success: true,
            output: output.trim(),
            error,
            code
          });
        } else {
          reject(new Error(`Claude exited with code ${code}: ${error || 'No output'}`));
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

    // Extract Python code blocks
    const codeMatches = result.output.matchAll(/```python\n([\s\S]*?)```/g);
    const codeBlocks = Array.from(codeMatches);
    
    if (codeBlocks.length > 0) {
      // Combine all code blocks
      response.code = codeBlocks.map(match => match[1]).join('\n\n');
      
      // Check if code contains visualization commands
      if (this.hasVisualizationCode(response.code)) {
        const vizResult = await this.executeVisualizationCode(response.code);
        if (vizResult.success) {
          response.visualization = vizResult.vizId;
          // Add visualization ID to response text if not already there
          if (!response.text.includes(vizResult.vizId)) {
            response.text += `\n\nVisualization generated: viz_${vizResult.vizId}`;
          }
        }
      }
    }

    return response;
  }

  hasVisualizationCode(code) {
    const vizPatterns = [
      'plt.', 'pyplot', 'matplotlib',
      'seaborn', 'sns.',
      'plotly', 'px.', 'go.',
      '.plot(', '.scatter(', '.bar(', '.hist(',
      'savefig('
    ];
    
    return vizPatterns.some(pattern => code.includes(pattern));
  }

  async executeVisualizationCode(code) {
    const vizId = uuidv4();
    const outputPath = path.join(this.visualizationsDir, `viz_${vizId}.png`);
    
    // Prepare code with proper imports and save logic
    const fullCode = `
import sys
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for headless operation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import commonly used libraries
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except ImportError:
    pass

# Change to data directory for file access
import os
os.chdir('${this.dataDir}')

# User code
${code}

# Save any open figures
import matplotlib.pyplot as plt
if plt.get_fignums():
    plt.savefig('${outputPath}', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close('all')
    print(f"\\nVisualization saved: viz_${vizId}")
else:
    print("\\nNo visualization was created.")
`;

    try {
      const result = await this.executeCode(fullCode, 'python');
      
      // Check if file was created
      const exists = await fs.access(outputPath).then(() => true).catch(() => false);
      
      if (exists) {
        console.log(`Visualization saved: ${outputPath}`);
      }
      
      return {
        success: exists,
        vizId: exists ? vizId : null,
        output: result.output
      };
    } catch (error) {
      console.error('Visualization execution error:', error);
      return { success: false, vizId: null, error: error.message };
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
          const errorStr = data.toString();
          // Filter out common warnings
          if (!errorStr.includes('UserWarning') && !errorStr.includes('FutureWarning')) {
            error += errorStr;
          }
        });

        execProcess.on('close', async (code) => {
          // Clean up temp file
          await fs.unlink(filepath).catch(() => {});
          
          if (code === 0) {
            resolve({ success: true, output, error });
          } else {
            reject(new Error(`Code execution failed with code ${code}: ${error || output}`));
          }
        });
      });
    } catch (error) {
      throw new Error(`Failed to execute code: ${error.message}`);
    }
  }

  async listDataFiles() {
    try {
      const files = await fs.readdir(this.dataDir);
      return files.filter(f => f.endsWith('.json') || f.endsWith('.csv'));
    } catch (error) {
      return [];
    }
  }
}

module.exports = ClaudeCLI;