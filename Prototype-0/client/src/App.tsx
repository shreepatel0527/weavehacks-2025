import React, { useState } from 'react';
import './App.css';
import ChatInterface from './components/ChatInterface';
import DataIngestion from './components/DataIngestion';
import VisualizationViewer from './components/VisualizationViewer';

function App() {
  const [activeTab, setActiveTab] = useState<'chat' | 'data' | 'viz'>('chat');
  const [lastVizId, setLastVizId] = useState<string | null>(null);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Claude-Flow Web App</h1>
        <nav>
          <button 
            className={activeTab === 'chat' ? 'active' : ''}
            onClick={() => setActiveTab('chat')}
          >
            Chat
          </button>
          <button 
            className={activeTab === 'data' ? 'active' : ''}
            onClick={() => setActiveTab('data')}
          >
            Data Ingestion
          </button>
          <button 
            className={activeTab === 'viz' ? 'active' : ''}
            onClick={() => setActiveTab('viz')}
          >
            Visualizations
          </button>
        </nav>
      </header>
      
      <main className="App-main">
        {activeTab === 'chat' && <ChatInterface onVisualizationGenerated={setLastVizId} />}
        {activeTab === 'data' && <DataIngestion />}
        {activeTab === 'viz' && <VisualizationViewer highlightVizId={lastVizId} />}
      </main>
    </div>
  );
}

export default App;
