const express = require('express');
const router = express.Router();
const path = require('path');
const fs = require('fs').promises;
const { v4: uuidv4 } = require('uuid');
const claudeFlow = require('../claude-flow');

router.post('/generate', async (req, res) => {
  try {
    const { dataFileId, visualizationType, pythonCode } = req.body;
    
    if (!dataFileId) {
      return res.status(400).json({ error: 'Data file ID is required' });
    }

    const vizId = uuidv4();
    const outputPath = path.join(__dirname, '../../visualizations', `viz_${vizId}.png`);
    
    let code;
    if (pythonCode) {
      code = pythonCode;
    } else {
      const dataPath = path.join(__dirname, '../../uploads', `data_${dataFileId}.json`);
      
      code = `
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
with open('${dataPath}', 'r') as f:
    data = json.load(f)

# Generate visualization based on type
viz_type = '${visualizationType || 'auto'}'

if isinstance(data, list) and len(data) > 0:
    df = pd.DataFrame(data)
    
    if viz_type == 'bar' or (viz_type == 'auto' and len(df.columns) >= 2):
        plt.figure(figsize=(10, 6))
        if df.select_dtypes(include=[np.number]).shape[1] >= 1:
            df.select_dtypes(include=[np.number]).iloc[:20].plot(kind='bar')
        plt.title('Data Visualization')
        plt.tight_layout()
    elif viz_type == 'line':
        plt.figure(figsize=(10, 6))
        df.select_dtypes(include=[np.number]).plot(kind='line')
        plt.title('Line Chart')
        plt.tight_layout()
    elif viz_type == 'scatter' and df.select_dtypes(include=[np.number]).shape[1] >= 2:
        plt.figure(figsize=(10, 6))
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]])
        plt.xlabel(numeric_cols[0])
        plt.ylabel(numeric_cols[1])
        plt.title('Scatter Plot')
    else:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f'Data shape: {df.shape}\\nColumns: {list(df.columns)}', 
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Data Summary')
else:
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, f'Data type: {type(data).__name__}\\nContent: {str(data)[:100]}...', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Data Preview')

plt.savefig('${outputPath}', dpi=150, bbox_inches='tight')
plt.close()
print(f"Visualization saved to ${outputPath}")
`;
    }

    const result = await claudeFlow.executeCode(code, 'python');
    
    const exists = await fs.access(outputPath).then(() => true).catch(() => false);
    
    if (exists) {
      res.json({
        success: true,
        vizId,
        message: 'Visualization generated successfully',
        output: result.output
      });
    } else {
      throw new Error('Visualization file was not created');
    }
  } catch (error) {
    console.error('Visualization error:', error);
    res.status(500).json({ 
      error: 'Failed to generate visualization',
      details: error.message 
    });
  }
});

router.get('/:vizId', async (req, res) => {
  try {
    const { vizId } = req.params;
    const filepath = path.join(__dirname, '../../visualizations', `viz_${vizId}.png`);
    
    const exists = await fs.access(filepath).then(() => true).catch(() => false);
    
    if (exists) {
      res.sendFile(filepath);
    } else {
      res.status(404).json({ error: 'Visualization not found' });
    }
  } catch (error) {
    res.status(500).json({ 
      error: 'Failed to retrieve visualization',
      details: error.message 
    });
  }
});

router.get('/', async (req, res) => {
  try {
    const vizDir = path.join(__dirname, '../../visualizations');
    const files = await fs.readdir(vizDir);
    
    const vizFiles = files
      .filter(f => f.startsWith('viz_') && f.endsWith('.png'))
      .map(f => ({
        vizId: f.replace('viz_', '').replace('.png', ''),
        filename: f
      }));
    
    res.json({
      success: true,
      visualizations: vizFiles
    });
  } catch (error) {
    res.status(500).json({ 
      error: 'Failed to list visualizations',
      details: error.message 
    });
  }
});

module.exports = router;