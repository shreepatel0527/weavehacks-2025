# Claude-Flow Web App Documentation

## Overview

This web application provides a simplified Poe.com-like interface that uses claude-flow as its underlying AI system. It's designed for demo purposes with no authentication required.

## Features

1. **Chat Interface**: Interactive Q&A with claude-flow
2. **Data Ingestion**: Upload JSON files for analysis
3. **Code Execution**: Run Python code for data processing
4. **Visualization**: Generate charts and graphs from data
5. **File Storage**: Local scratch space for data and results

## Architecture

### Backend (Node.js/Express)
- `/api/chat` - Chat interface with claude-flow
- `/api/ingest` - JSON data upload and storage
- `/api/visualizations` - Generate and retrieve visualizations

### Frontend (React/TypeScript)
- Chat UI component
- Data ingestion interface
- Visualization viewer

### Claude-Flow Integration
The app simulates claude-flow responses for demo purposes. In production, it would spawn actual claude-flow processes.

## Getting Started

### Quick Start
```bash
./start.sh
```

### Manual Setup
1. Install dependencies:
   ```bash
   npm install
   cd client && npm install
   ```

2. Start development server:
   ```bash
   npm run dev
   ```

### Production Build
```bash
./build.sh
npm start
```

## API Usage

### Chat API
```bash
POST /api/chat
{
  "message": "Hello, claude-flow!",
  "context": {}
}
```

### Data Ingestion
```bash
POST /api/ingest
{
  "data": { "key": "value", "values": [1, 2, 3] }
}
```

### Generate Visualization
```bash
POST /api/visualizations/generate
{
  "dataFileId": "uuid-here",
  "visualizationType": "bar"
}
```

## Testing

Run tests:
```bash
npm test
```

## Directory Structure
- `/scratch` - Temporary files and code execution
- `/uploads` - Ingested JSON data files
- `/visualizations` - Generated visualization images

## Development Notes

- The app runs on port 3001 by default
- Frontend development server runs on port 3000
- All data is stored locally (no database required)
- Python 3 and matplotlib required for visualizations