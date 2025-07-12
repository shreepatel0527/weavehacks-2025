const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fs = require('fs').promises;
const { v4: uuidv4 } = require('uuid');

const upload = multer({
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'application/json') {
      cb(null, true);
    } else {
      cb(new Error('Only JSON files are allowed'));
    }
  }
});

router.post('/', upload.single('data'), async (req, res) => {
  try {
    let jsonData;
    
    if (req.file) {
      const content = req.file.buffer.toString();
      jsonData = JSON.parse(content);
    } else if (req.body.data) {
      jsonData = typeof req.body.data === 'string' 
        ? JSON.parse(req.body.data) 
        : req.body.data;
    } else {
      return res.status(400).json({ error: 'No data provided' });
    }

    const fileId = uuidv4();
    const filename = `data_${fileId}.json`;
    const filepath = path.join(__dirname, '../../uploads', filename);
    
    await fs.writeFile(filepath, JSON.stringify(jsonData, null, 2));
    
    res.json({
      success: true,
      fileId,
      filename,
      message: 'Data ingested successfully',
      dataSize: JSON.stringify(jsonData).length
    });
  } catch (error) {
    console.error('Ingestion error:', error);
    res.status(500).json({ 
      error: 'Failed to ingest data',
      details: error.message 
    });
  }
});

router.get('/:fileId', async (req, res) => {
  try {
    const { fileId } = req.params;
    const filename = `data_${fileId}.json`;
    const filepath = path.join(__dirname, '../../uploads', filename);
    
    const data = await fs.readFile(filepath, 'utf-8');
    res.json({
      success: true,
      data: JSON.parse(data),
      fileId,
      filename
    });
  } catch (error) {
    if (error.code === 'ENOENT') {
      res.status(404).json({ error: 'File not found' });
    } else {
      res.status(500).json({ 
        error: 'Failed to retrieve data',
        details: error.message 
      });
    }
  }
});

router.get('/', async (req, res) => {
  try {
    const uploadsDir = path.join(__dirname, '../../uploads');
    const files = await fs.readdir(uploadsDir);
    
    const dataFiles = files
      .filter(f => f.startsWith('data_') && f.endsWith('.json'))
      .map(f => ({
        fileId: f.replace('data_', '').replace('.json', ''),
        filename: f
      }));
    
    res.json({
      success: true,
      files: dataFiles
    });
  } catch (error) {
    res.status(500).json({ 
      error: 'Failed to list data files',
      details: error.message 
    });
  }
});

module.exports = router;