import React, { useState, useEffect, useRef } from 'react';
import './LabDashboard.css';

interface SafetyAlert {
  timestamp: string;
  level: 'warning' | 'critical' | 'emergency';
  message: string;
  raw?: string;
}

interface SafetyStatus {
  isMonitoring: boolean;
  alertCount: number;
  recentAlerts: SafetyAlert[];
  processId?: number;
}

interface Experiment {
  id: string;
  protocol: string;
  researcher: string;
  startTime: string;
  status: 'active' | 'completed' | 'paused';
  phase: string;
  reagents: Record<string, any>;
  observations: any[];
  dataPoints: any[];
}

interface VoiceRecording {
  isRecording: boolean;
  audioBlob?: Blob;
  transcript?: string;
}

const LabDashboard: React.FC = () => {
  const [safetyStatus, setSafetyStatus] = useState<SafetyStatus>({
    isMonitoring: false,
    alertCount: 0,
    recentAlerts: []
  });
  
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [currentExperiment, setCurrentExperiment] = useState<Experiment | null>(null);
  const [voiceRecording, setVoiceRecording] = useState<VoiceRecording>({ isRecording: false });
  const [textInput, setTextInput] = useState('');
  const [activeTab, setActiveTab] = useState<'dashboard' | 'safety' | 'experiments' | 'voice'>('dashboard');
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Initialize WebSocket connection
    const ws = new WebSocket(`ws://localhost:3001/ws`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('Connected to WebSocket');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'safety-event') {
        handleSafetyEvent(data.data);
      } else if (data.type === 'safety-status') {
        setSafetyStatus(data.data);
      }
    };

    ws.onclose = () => {
      console.log('Disconnected from WebSocket');
    };

    // Fetch initial data
    fetchSafetyStatus();
    fetchExperiments();

    return () => {
      ws.close();
    };
  }, []);

  const handleSafetyEvent = (eventData: any) => {
    if (eventData.type === 'alert') {
      setSafetyStatus(prev => ({
        ...prev,
        alertCount: prev.alertCount + 1,
        recentAlerts: [eventData.data, ...prev.recentAlerts.slice(0, 9)]
      }));
      
      // Show browser notification for critical alerts
      if (eventData.data.level === 'critical' || eventData.data.level === 'emergency') {
        if (Notification.permission === 'granted') {
          new Notification(`🚨 Lab Safety Alert: ${eventData.data.level.toUpperCase()}`, {
            body: eventData.data.message,
            icon: '/favicon.ico'
          });
        }
      }
    }
  };

  const fetchSafetyStatus = async () => {
    try {
      const response = await fetch('/api/safety/status');
      const data = await response.json();
      if (data.success) {
        setSafetyStatus(data.status);
      }
    } catch (error) {
      console.error('Failed to fetch safety status:', error);
    }
  };

  const fetchExperiments = async () => {
    try {
      const response = await fetch('/api/data-collection/experiments');
      const data = await response.json();
      if (data.success) {
        setExperiments(data.experiments);
        if (data.experiments.length > 0) {
          setCurrentExperiment(data.experiments[0]);
        }
      }
    } catch (error) {
      console.error('Failed to fetch experiments:', error);
    }
  };

  const startSafetyMonitoring = async () => {
    try {
      const response = await fetch('/api/safety/start', { method: 'POST' });
      const data = await response.json();
      if (data.success) {
        setSafetyStatus(data.status);
      }
    } catch (error) {
      console.error('Failed to start safety monitoring:', error);
    }
  };

  const stopSafetyMonitoring = async () => {
    try {
      const response = await fetch('/api/safety/stop', { method: 'POST' });
      const data = await response.json();
      if (data.success) {
        setSafetyStatus(data.status);
      }
    } catch (error) {
      console.error('Failed to stop safety monitoring:', error);
    }
  };

  const createExperiment = async () => {
    const protocol = prompt('Enter experiment protocol:');
    const researcher = prompt('Enter researcher name:');
    
    if (!protocol || !researcher) return;

    try {
      const response = await fetch('/api/data-collection/experiment/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ protocol, researcher })
      });
      
      const data = await response.json();
      if (data.success) {
        setCurrentExperiment(data.experiment);
        fetchExperiments();
      }
    } catch (error) {
      console.error('Failed to create experiment:', error);
    }
  };

  const startVoiceRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        setVoiceRecording(prev => ({ ...prev, audioBlob, isRecording: false }));
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setVoiceRecording(prev => ({ ...prev, isRecording: true }));
    } catch (error) {
      console.error('Failed to start recording:', error);
    }
  };

  const stopVoiceRecording = () => {
    if (mediaRecorderRef.current && voiceRecording.isRecording) {
      mediaRecorderRef.current.stop();
    }
  };

  const processVoiceInput = async () => {
    if (!voiceRecording.audioBlob || !currentExperiment) return;

    const formData = new FormData();
    formData.append('audio', voiceRecording.audioBlob, 'recording.wav');
    formData.append('experimentId', currentExperiment.id);
    
    // For demo, include a sample transcript
    formData.append('transcript', 'The mass of the gold compound is 0.1598 grams');

    try {
      const response = await fetch('/api/data-collection/voice/upload', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      if (data.success) {
        setVoiceRecording({ isRecording: false });
        fetchExperiments(); // Refresh to show updated experiment data
      }
    } catch (error) {
      console.error('Failed to process voice input:', error);
    }
  };

  const processTextInput = async () => {
    if (!textInput.trim() || !currentExperiment) return;

    try {
      const response = await fetch('/api/data-collection/text/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: textInput,
          experimentId: currentExperiment.id
        })
      });
      
      const data = await response.json();
      if (data.success) {
        setTextInput('');
        fetchExperiments(); // Refresh to show updated experiment data
      }
    } catch (error) {
      console.error('Failed to process text input:', error);
    }
  };

  const getAlertColor = (level: string) => {
    switch (level) {
      case 'emergency': return '#dc2626';
      case 'critical': return '#f59e0b';
      case 'warning': return '#eab308';
      default: return '#6b7280';
    }
  };

  // Request notification permission
  useEffect(() => {
    if (Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }, []);

  return (
    <div className="lab-dashboard">
      <header className="dashboard-header">
        <h1>🧬 WeaveHacks 2025 Lab Automation Dashboard</h1>
        <div className="status-indicators">
          <div className={`status-indicator ${safetyStatus.isMonitoring ? 'active' : 'inactive'}`}>
            🛡️ Safety: {safetyStatus.isMonitoring ? 'Active' : 'Inactive'}
          </div>
          <div className="status-indicator">
            🧪 Experiments: {experiments.length}
          </div>
          <div className="status-indicator">
            🚨 Alerts: {safetyStatus.alertCount}
          </div>
        </div>
      </header>

      <nav className="dashboard-nav">
        <button 
          className={activeTab === 'dashboard' ? 'active' : ''}
          onClick={() => setActiveTab('dashboard')}
        >
          📊 Dashboard
        </button>
        <button 
          className={activeTab === 'safety' ? 'active' : ''}
          onClick={() => setActiveTab('safety')}
        >
          🛡️ Safety
        </button>
        <button 
          className={activeTab === 'experiments' ? 'active' : ''}
          onClick={() => setActiveTab('experiments')}
        >
          🧪 Experiments
        </button>
        <button 
          className={activeTab === 'voice' ? 'active' : ''}
          onClick={() => setActiveTab('voice')}
        >
          🎙️ Voice Input
        </button>
      </nav>

      <main className="dashboard-content">
        {activeTab === 'dashboard' && (
          <div className="dashboard-overview">
            <div className="overview-grid">
              <div className="overview-card">
                <h3>🛡️ Safety Status</h3>
                <div className="card-content">
                  <p>Monitoring: <span className={safetyStatus.isMonitoring ? 'status-active' : 'status-inactive'}>
                    {safetyStatus.isMonitoring ? 'ACTIVE' : 'INACTIVE'}
                  </span></p>
                  <p>Recent Alerts: {safetyStatus.recentAlerts.length}</p>
                  <div className="action-buttons">
                    <button onClick={startSafetyMonitoring} disabled={safetyStatus.isMonitoring}>
                      Start Monitoring
                    </button>
                    <button onClick={stopSafetyMonitoring} disabled={!safetyStatus.isMonitoring}>
                      Stop Monitoring
                    </button>
                  </div>
                </div>
              </div>

              <div className="overview-card">
                <h3>🧪 Current Experiment</h3>
                <div className="card-content">
                  {currentExperiment ? (
                    <>
                      <p><strong>ID:</strong> {currentExperiment.id}</p>
                      <p><strong>Researcher:</strong> {currentExperiment.researcher}</p>
                      <p><strong>Protocol:</strong> {currentExperiment.protocol}</p>
                      <p><strong>Status:</strong> {currentExperiment.status}</p>
                      <p><strong>Phase:</strong> {currentExperiment.phase}</p>
                    </>
                  ) : (
                    <p>No active experiment</p>
                  )}
                  <button onClick={createExperiment}>Create New Experiment</button>
                </div>
              </div>

              <div className="overview-card">
                <h3>📊 Recent Activity</h3>
                <div className="card-content">
                  {safetyStatus.recentAlerts.slice(0, 3).map((alert, index) => (
                    <div key={index} className="activity-item" style={{ borderLeftColor: getAlertColor(alert.level) }}>
                      <span className="activity-time">
                        {new Date(alert.timestamp).toLocaleTimeString()}
                      </span>
                      <span className="activity-message">{alert.message}</span>
                    </div>
                  ))}
                  {safetyStatus.recentAlerts.length === 0 && <p>No recent activity</p>}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'safety' && (
          <div className="safety-panel">
            <div className="safety-controls">
              <h2>Safety Monitoring Controls</h2>
              <div className="control-buttons">
                <button 
                  className={safetyStatus.isMonitoring ? 'btn-stop' : 'btn-start'}
                  onClick={safetyStatus.isMonitoring ? stopSafetyMonitoring : startSafetyMonitoring}
                >
                  {safetyStatus.isMonitoring ? '⏹️ Stop Monitoring' : '▶️ Start Monitoring'}
                </button>
                <button onClick={() => fetch('/api/safety/test-alert', { method: 'POST' })}>
                  🧪 Test Alert
                </button>
              </div>
            </div>

            <div className="alerts-panel">
              <h3>Recent Safety Alerts</h3>
              <div className="alerts-list">
                {safetyStatus.recentAlerts.map((alert, index) => (
                  <div key={index} className={`alert-item alert-${alert.level}`}>
                    <div className="alert-header">
                      <span className="alert-level">{alert.level.toUpperCase()}</span>
                      <span className="alert-time">
                        {new Date(alert.timestamp).toLocaleString()}
                      </span>
                    </div>
                    <div className="alert-message">{alert.message}</div>
                  </div>
                ))}
                {safetyStatus.recentAlerts.length === 0 && (
                  <p className="no-alerts">No safety alerts</p>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'experiments' && (
          <div className="experiments-panel">
            <div className="experiments-header">
              <h2>Experiment Management</h2>
              <button onClick={createExperiment}>+ New Experiment</button>
            </div>
            
            <div className="experiments-list">
              {experiments.map(exp => (
                <div key={exp.id} className={`experiment-item ${exp.id === currentExperiment?.id ? 'active' : ''}`}>
                  <div className="experiment-header">
                    <h3>{exp.protocol}</h3>
                    <span className={`status status-${exp.status}`}>{exp.status}</span>
                  </div>
                  <div className="experiment-details">
                    <p><strong>Researcher:</strong> {exp.researcher}</p>
                    <p><strong>Started:</strong> {new Date(exp.startTime).toLocaleString()}</p>
                    <p><strong>Observations:</strong> {exp.observations.length}</p>
                    <p><strong>Data Points:</strong> {exp.dataPoints.length}</p>
                    <p><strong>Reagents:</strong> {Object.keys(exp.reagents).length}</p>
                  </div>
                  <button onClick={() => setCurrentExperiment(exp)}>
                    Select
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'voice' && (
          <div className="voice-panel">
            <h2>Voice Data Collection</h2>
            
            {!currentExperiment && (
              <div className="warning">
                ⚠️ Please select an active experiment before using voice input
              </div>
            )}

            <div className="voice-controls">
              <div className="recording-section">
                <h3>🎙️ Voice Recording</h3>
                <div className="recording-controls">
                  <button 
                    className={voiceRecording.isRecording ? 'btn-recording' : 'btn-record'}
                    onClick={voiceRecording.isRecording ? stopVoiceRecording : startVoiceRecording}
                    disabled={!currentExperiment}
                  >
                    {voiceRecording.isRecording ? '⏹️ Stop Recording' : '🎙️ Start Recording'}
                  </button>
                  
                  {voiceRecording.audioBlob && (
                    <button onClick={processVoiceInput}>
                      📤 Process Recording
                    </button>
                  )}
                </div>
                
                {voiceRecording.isRecording && (
                  <div className="recording-indicator">
                    🔴 Recording... Speak your observations or measurements
                  </div>
                )}
              </div>

              <div className="text-input-section">
                <h3>⌨️ Text Input</h3>
                <div className="text-controls">
                  <textarea
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    placeholder="Enter your observations, measurements, or data..."
                    rows={3}
                    disabled={!currentExperiment}
                  />
                  <button 
                    onClick={processTextInput}
                    disabled={!textInput.trim() || !currentExperiment}
                  >
                    📤 Process Text
                  </button>
                </div>
              </div>

              <div className="examples-section">
                <h3>💡 Example Inputs</h3>
                <div className="examples">
                  <div className="example">"The mass of the gold compound is 0.1598 grams"</div>
                  <div className="example">"Volume of dichloromethane is 25.5 mL"</div>
                  <div className="example">"I observe black particles forming at the bottom"</div>
                  <div className="example">"Temperature is 28.5 degrees Celsius"</div>
                  <div className="example">"Pressure reading shows 102.3 kPa"</div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default LabDashboard;