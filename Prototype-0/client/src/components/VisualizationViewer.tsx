import React, { useState, useEffect } from 'react';

interface Visualization {
  vizId: string;
  filename: string;
}

interface VisualizationViewerProps {
  highlightVizId?: string | null;
}

const VisualizationViewer: React.FC<VisualizationViewerProps> = ({ highlightVizId }) => {
  const [visualizations, setVisualizations] = useState<Visualization[]>([]);
  const [selectedViz, setSelectedViz] = useState<string | null>(null);

  useEffect(() => {
    fetchVisualizations();
  }, []);

  const fetchVisualizations = async () => {
    try {
      const response = await fetch('http://localhost:3001/api/visualizations');
      const data = await response.json();
      if (data.success) {
        setVisualizations(data.visualizations);
      }
    } catch (error) {
      console.error('Failed to fetch visualizations:', error);
    }
  };

  const getVizUrl = (vizId: string) => {
    return `http://localhost:3001/api/visualizations/${vizId}`;
  };

  return (
    <div className="visualization-container">
      <h2>Visualizations</h2>
      
      {selectedViz && (
        <div style={{ marginBottom: '20px' }}>
          <img 
            src={getVizUrl(selectedViz)} 
            alt="Selected visualization"
            style={{ maxWidth: '100%', maxHeight: '600px', margin: '0 auto', display: 'block' }}
          />
          <button onClick={() => setSelectedViz(null)} style={{ marginTop: '10px' }}>
            Close
          </button>
        </div>
      )}

      <div className="viz-grid">
        {visualizations.length === 0 ? (
          <p>No visualizations generated yet</p>
        ) : (
          visualizations.map(viz => (
            <div 
              key={viz.vizId} 
              className={`viz-item ${highlightVizId === viz.vizId ? 'highlighted' : ''}`}
              onClick={() => setSelectedViz(viz.vizId)}
            >
              <img src={getVizUrl(viz.vizId)} alt={viz.filename} />
              <div className="viz-info">
                <p>{viz.filename}</p>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default VisualizationViewer;