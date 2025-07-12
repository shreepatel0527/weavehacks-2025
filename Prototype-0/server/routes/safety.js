const express = require('express');
const router = express.Router();
const { spawn } = require('child_process');
const path = require('path');

class SafetyMonitoringService {
  constructor() {
    this.monitoringProcess = null;
    this.alerts = [];
    this.isMonitoring = false;
    this.statusCallbacks = [];
  }

  startMonitoring() {
    if (this.isMonitoring) {
      throw new Error('Safety monitoring is already running');
    }

    const pythonScript = path.join(__dirname, '../../../safety_monitoring/advanced_safety_agent.py');
    
    this.monitoringProcess = spawn('python3', [pythonScript], {
      cwd: path.join(__dirname, '../../../safety_monitoring'),
      stdio: ['pipe', 'pipe', 'pipe']
    });

    this.monitoringProcess.stdout.on('data', (data) => {
      const output = data.toString();
      console.log('Safety Monitor:', output);
      
      // Parse safety alerts from output
      if (output.includes('SAFETY ALERT:') || output.includes('EMERGENCY SHUTDOWN:')) {
        const alert = this.parseAlertFromOutput(output);
        if (alert) {
          this.alerts.push(alert);
          this.notifyStatusCallbacks({
            type: 'alert',
            data: alert
          });
        }
      }
    });

    this.monitoringProcess.stderr.on('data', (data) => {
      console.error('Safety Monitor Error:', data.toString());
    });

    this.monitoringProcess.on('close', (code) => {
      console.log(`Safety monitoring process exited with code ${code}`);
      this.isMonitoring = false;
      this.notifyStatusCallbacks({
        type: 'monitoring_stopped',
        data: { exitCode: code }
      });
    });

    this.isMonitoring = true;
    console.log('Safety monitoring started');
  }

  stopMonitoring() {
    if (this.monitoringProcess && this.isMonitoring) {
      this.monitoringProcess.kill('SIGTERM');
      this.isMonitoring = false;
      console.log('Safety monitoring stopped');
    }
  }

  parseAlertFromOutput(output) {
    try {
      // Extract JSON-like data from log output
      const lines = output.split('\n');
      for (const line of lines) {
        if (line.includes('SAFETY ALERT:') || line.includes('EMERGENCY SHUTDOWN:')) {
          const timestamp = new Date().toISOString();
          const alertData = {
            timestamp,
            message: line.trim(),
            level: this.extractAlertLevel(line),
            raw: output
          };
          return alertData;
        }
      }
    } catch (error) {
      console.error('Failed to parse alert:', error);
    }
    return null;
  }

  extractAlertLevel(line) {
    if (line.includes('EMERGENCY')) return 'emergency';
    if (line.includes('CRITICAL')) return 'critical';
    if (line.includes('WARNING')) return 'warning';
    return 'info';
  }

  getStatus() {
    return {
      isMonitoring: this.isMonitoring,
      alertCount: this.alerts.length,
      recentAlerts: this.alerts.slice(-10),
      processId: this.monitoringProcess ? this.monitoringProcess.pid : null
    };
  }

  clearAlerts() {
    this.alerts = [];
  }

  addStatusCallback(callback) {
    this.statusCallbacks.push(callback);
  }

  notifyStatusCallbacks(event) {
    this.statusCallbacks.forEach(callback => {
      try {
        callback(event);
      } catch (error) {
        console.error('Status callback error:', error);
      }
    });
  }
}

// Global safety monitoring service instance
const safetyService = new SafetyMonitoringService();

// Routes
router.post('/start', async (req, res) => {
  try {
    if (safetyService.isMonitoring) {
      return res.status(409).json({ 
        error: 'Safety monitoring is already running',
        status: safetyService.getStatus()
      });
    }

    safetyService.startMonitoring();
    
    res.json({
      success: true,
      message: 'Safety monitoring started',
      status: safetyService.getStatus()
    });
  } catch (error) {
    console.error('Failed to start safety monitoring:', error);
    res.status(500).json({ 
      error: 'Failed to start safety monitoring',
      details: error.message 
    });
  }
});

router.post('/stop', async (req, res) => {
  try {
    safetyService.stopMonitoring();
    
    res.json({
      success: true,
      message: 'Safety monitoring stopped',
      status: safetyService.getStatus()
    });
  } catch (error) {
    console.error('Failed to stop safety monitoring:', error);
    res.status(500).json({ 
      error: 'Failed to stop safety monitoring',
      details: error.message 
    });
  }
});

router.get('/status', (req, res) => {
  res.json({
    success: true,
    status: safetyService.getStatus()
  });
});

router.get('/alerts', (req, res) => {
  const { limit = 50, level } = req.query;
  let alerts = safetyService.alerts;
  
  if (level) {
    alerts = alerts.filter(alert => alert.level === level);
  }
  
  alerts = alerts.slice(-parseInt(limit));
  
  res.json({
    success: true,
    alerts,
    total: safetyService.alerts.length
  });
});

router.delete('/alerts', (req, res) => {
  safetyService.clearAlerts();
  res.json({
    success: true,
    message: 'Alerts cleared'
  });
});

router.post('/test-alert', (req, res) => {
  const { level = 'warning', message = 'Test alert' } = req.body;
  
  const testAlert = {
    timestamp: new Date().toISOString(),
    level,
    message: `TEST: ${message}`,
    raw: 'Test alert generated from API'
  };
  
  safetyService.alerts.push(testAlert);
  safetyService.notifyStatusCallbacks({
    type: 'alert',
    data: testAlert
  });
  
  res.json({
    success: true,
    message: 'Test alert generated',
    alert: testAlert
  });
});

// WebSocket integration
router.setupWebSocket = (wsHandler) => {
  safetyService.addStatusCallback((event) => {
    wsHandler.broadcast('safety-event', event);
  });
  
  // Send periodic status updates
  setInterval(() => {
    if (safetyService.isMonitoring) {
      wsHandler.broadcast('safety-status', safetyService.getStatus());
    }
  }, 5000); // Every 5 seconds
};

module.exports = { router, safetyService };