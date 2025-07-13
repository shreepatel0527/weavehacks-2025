const express = require('express');
const router = express.Router();
const claudeFlow = require('../claude-flow');

router.post('/', async (req, res) => {
  try {
    const { message, context } = req.body;
    
    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }

    const result = await claudeFlow.processMessage(message, context);
    
    res.json({
      success: true,
      response: result.response,
      context: result.context
    });
  } catch (error) {
    console.error('Chat error:', error);
    res.status(500).json({ 
      error: 'Failed to process message',
      details: error.message 
    });
  }
});

router.post('/execute', async (req, res) => {
  try {
    const { code, language = 'python' } = req.body;
    
    if (!code) {
      return res.status(400).json({ error: 'Code is required' });
    }

    const result = await claudeFlow.executeCode(code, language);
    
    res.json({
      success: true,
      output: result.output,
      error: result.error
    });
  } catch (error) {
    console.error('Code execution error:', error);
    res.status(500).json({ 
      error: 'Failed to execute code',
      details: error.message 
    });
  }
});

module.exports = router;