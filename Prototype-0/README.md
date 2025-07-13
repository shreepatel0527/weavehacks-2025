# Claude-Flow Web App

A simplified web application inspired by Poe.com that uses claude-flow as its AI backend.

## Features

- Chat interface for Q&A interactions
- JSON API for data ingestion
- Python code execution for data processing
- Visualization rendering capabilities
- Single-user demo mode (no authentication)

## Architecture

- **Backend**: Node.js/Express API server
- **Frontend**: React-based chat UI
- **AI Engine**: claude-flow integration
- **Data Storage**: Local file system
- **Code Execution**: Python subprocess

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

3. Open http://localhost:3000 in your browser

## API Endpoints

- `POST /api/chat` - Send messages to claude-flow
- `POST /api/ingest` - Upload JSON data
- `GET /api/visualizations/:id` - Retrieve generated visualizations