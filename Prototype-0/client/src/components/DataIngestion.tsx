import React, { useState, useEffect } from 'react';

interface DataFile {
  fileId: string;
  filename: string;
}

const DataIngestion: React.FC = () => {
  const [files, setFiles] = useState<DataFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);
  const [jsonInput, setJsonInput] = useState('');

  useEffect(() => {
    fetchFiles();
  }, []);

  const fetchFiles = async () => {
    try {
      const response = await fetch('http://localhost:3001/api/ingest');
      const data = await response.json();
      if (data.success) {
        setFiles(data.files);
      }
    } catch (error) {
      console.error('Failed to fetch files:', error);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setMessage(null);

    const formData = new FormData();
    formData.append('data', file);

    try {
      const response = await fetch('http://localhost:3001/api/ingest', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (data.success) {
        setMessage({ type: 'success', text: `File uploaded successfully! ID: ${data.fileId}` });
        fetchFiles();
      } else {
        throw new Error(data.error || 'Upload failed');
      }
    } catch (error) {
      setMessage({ 
        type: 'error', 
        text: error instanceof Error ? error.message : 'Upload failed' 
      });
    } finally {
      setUploading(false);
    }
  };

  const handleJsonSubmit = async () => {
    if (!jsonInput.trim()) return;

    setUploading(true);
    setMessage(null);

    try {
      const response = await fetch('http://localhost:3001/api/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: JSON.parse(jsonInput) })
      });

      const data = await response.json();

      if (data.success) {
        setMessage({ type: 'success', text: `Data ingested successfully! ID: ${data.fileId}` });
        setJsonInput('');
        fetchFiles();
      } else {
        throw new Error(data.error || 'Ingestion failed');
      }
    } catch (error) {
      setMessage({ 
        type: 'error', 
        text: error instanceof Error ? error.message : 'Invalid JSON or ingestion failed' 
      });
    } finally {
      setUploading(false);
    }
  };

  const generateVisualization = async (fileId: string) => {
    try {
      const response = await fetch('http://localhost:3001/api/visualizations/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataFileId: fileId })
      });

      const data = await response.json();

      if (data.success) {
        setMessage({ 
          type: 'success', 
          text: `Visualization generated! ID: ${data.vizId}` 
        });
      } else {
        throw new Error(data.error || 'Visualization failed');
      }
    } catch (error) {
      setMessage({ 
        type: 'error', 
        text: error instanceof Error ? error.message : 'Visualization failed' 
      });
    }
  };

  return (
    <div className="data-ingestion-container">
      <h2>Data Ingestion</h2>
      
      <div className="upload-area">
        <h3>Upload JSON File</h3>
        <input
          type="file"
          accept=".json"
          onChange={handleFileUpload}
          disabled={uploading}
        />
        <p>or</p>
        <h3>Paste JSON Data</h3>
        <textarea
          value={jsonInput}
          onChange={(e) => setJsonInput(e.target.value)}
          placeholder='{"key": "value", "data": [1, 2, 3]}'
          rows={5}
          style={{ width: '100%', marginBottom: '10px' }}
        />
        <button 
          onClick={handleJsonSubmit} 
          disabled={uploading || !jsonInput.trim()}
        >
          Submit JSON
        </button>
      </div>

      {message && (
        <div className={`${message.type}-message`}>
          {message.text}
        </div>
      )}

      <div className="file-list">
        <h3>Uploaded Files</h3>
        {files.length === 0 ? (
          <p>No files uploaded yet</p>
        ) : (
          files.map(file => (
            <div key={file.fileId} className="file-item">
              <span>{file.filename}</span>
              <button onClick={() => generateVisualization(file.fileId)}>
                Generate Visualization
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default DataIngestion;