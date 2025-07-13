import React, { useState, useEffect } from 'react';
import './App.css';
import ExperimentDashboard from './components/ExperimentDashboard';
import VoiceDataEntry from './components/VoiceDataEntry';
import SafetyMonitor from './components/SafetyMonitor';
import AgentStatus from './components/AgentStatus';
import DataVisualization from './components/DataVisualization';

interface Experiment {
  experiment_id: string;
  step_num: number;
  status: string;
  mass_gold: number;
  mass_toab: number;
  mass_sulfur: number;
  mass_nabh4: number;
  mass_final: number;
  volume_toluene: number;
  volume_nanopure_rt: number;
  volume_nanopure_cold: number;
  safety_status: string;
  observations: string;
  created_at: string;
  updated_at: string;
}

interface SafetyAlert {
  experiment_id: string;
  parameter: string;
  value: number;
  threshold: number;
  severity: string;
  timestamp: string;
}

interface Agent {
  agent_id: string;
  agent_type: string;
  status: string;
  current_task?: string;
  last_updated: string;
}

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [currentExperiment, setCurrentExperiment] = useState<Experiment | null>(null);
  const [safetyAlerts, setSafetyAlerts] = useState<SafetyAlert[]>([]);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [activeTab, setActiveTab] = useState<string>('dashboard');

  // Fetch experiments
  const fetchExperiments = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/experiments`);
      const data = await response.json();
      setExperiments(data);
      
      if (data.length > 0 && !currentExperiment) {
        setCurrentExperiment(data[0]);
      }
    } catch (error) {
      console.error('Failed to fetch experiments:', error);
    }
  };

  // Fetch safety alerts
  const fetchSafetyAlerts = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/safety/alerts`);
      const data = await response.json();
      setSafetyAlerts(data);
    } catch (error) {
      console.error('Failed to fetch safety alerts:', error);
    }
  };

  // Fetch agent statuses
  const fetchAgents = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/agents`);
      const data = await response.json();
      setAgents(data);
    } catch (error) {
      console.error('Failed to fetch agents:', error);
    }
  };

  // Create new experiment
  const createExperiment = async () => {
    const experimentId = `exp_${Date.now()}`;
    try {
      const response = await fetch(`${API_BASE_URL}/experiments?experiment_id=${experimentId}`, {
        method: 'POST',
      });
      
      if (response.ok) {
        const newExperiment = await response.json();
        setExperiments([...experiments, newExperiment]);
        setCurrentExperiment(newExperiment);
      }
    } catch (error) {
      console.error('Failed to create experiment:', error);
    }
  };

  // Update experiment data
  const updateExperimentData = async () => {
    if (currentExperiment) {
      try {
        const response = await fetch(`${API_BASE_URL}/experiments/${currentExperiment.experiment_id}`);
        const updatedExperiment = await response.json();
        setCurrentExperiment(updatedExperiment);
        
        // Update in experiments list
        setExperiments(experiments.map(exp => 
          exp.experiment_id === updatedExperiment.experiment_id ? updatedExperiment : exp
        ));
      } catch (error) {
        console.error('Failed to update experiment data:', error);
      }
    }
  };

  // Polling for real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      fetchExperiments();
      fetchSafetyAlerts();
      fetchAgents();
      updateExperimentData();
    }, 5000); // Poll every 5 seconds

    return () => clearInterval(interval);
  }, [currentExperiment]);

  // Initial data fetch
  useEffect(() => {
    fetchExperiments();
    fetchSafetyAlerts();
    fetchAgents();
  }, []);

  const renderTabContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return (
          <ExperimentDashboard 
            experiment={currentExperiment}
            experiments={experiments}
            onExperimentSelect={setCurrentExperiment}
            onCreateExperiment={createExperiment}
          />
        );
      case 'voice':
        return (
          <VoiceDataEntry 
            experiment={currentExperiment}
            onDataRecorded={updateExperimentData}
          />
        );
      case 'safety':
        return (
          <SafetyMonitor 
            alerts={safetyAlerts}
            experiment={currentExperiment}
          />
        );
      case 'agents':
        return (
          <AgentStatus 
            agents={agents}
          />
        );
      case 'visualization':
        return (
          <DataVisualization 
            experiment={currentExperiment}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🧬 WeaveHacks Lab Automation Platform</h1>
        <div className="experiment-info">
          {currentExperiment && (
            <span>
              Current: {currentExperiment.experiment_id} | 
              Step: {currentExperiment.step_num} | 
              Status: <span className={`status-${currentExperiment.status.replace('_', '-')}`}>
                {currentExperiment.status.replace('_', ' ').toUpperCase()}
              </span>
            </span>
          )}
        </div>
      </header>

      <nav className="tab-navigation">
        <button 
          className={activeTab === 'dashboard' ? 'active' : ''}
          onClick={() => setActiveTab('dashboard')}
        >
          📊 Dashboard
        </button>
        <button 
          className={activeTab === 'voice' ? 'active' : ''}
          onClick={() => setActiveTab('voice')}
        >
          🎤 Voice Entry
        </button>
        <button 
          className={activeTab === 'safety' ? 'active' : ''}
          onClick={() => setActiveTab('safety')}
        >
          🛡️ Safety Monitor
        </button>
        <button 
          className={activeTab === 'agents' ? 'active' : ''}
          onClick={() => setActiveTab('agents')}
        >
          🤖 Agents
        </button>
        <button 
          className={activeTab === 'visualization' ? 'active' : ''}
          onClick={() => setActiveTab('visualization')}
        >
          📈 Data Viz
        </button>
      </nav>

      <main className="app-content">
        {renderTabContent()}
      </main>

      <footer className="app-footer">
        <div className="system-status">
          <span className={`status-indicator ${safetyAlerts.some(a => a.severity === 'critical') ? 'critical' : 'safe'}`}>
            {safetyAlerts.some(a => a.severity === 'critical') ? '🚨 SAFETY ALERT' : '✅ System Safe'}
          </span>
          <span className="agent-count">
            {agents.filter(a => a.status === 'active').length}/{agents.length} Agents Active
          </span>
          <span className="last-update">
            Last Update: {new Date().toLocaleTimeString()}
          </span>
        </div>
      </footer>
    </div>
  );
}

export default App;