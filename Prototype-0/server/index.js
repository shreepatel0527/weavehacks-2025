const express = require('express');
const cors = require('cors');
const path = require('path');
const http = require('http');
require('dotenv').config();

const chatRoutes = require('./routes/chat');
const ingestRoutes = require('./routes/ingest');
const visualizationRoutes = require('./routes/visualization');
const { router: safetyRoutes, safetyService } = require('./routes/safety');
const { router: dataCollectionRoutes, dataAgent } = require('./routes/data_collection');
const { router: sensorDataRoutes, sensorDataService } = require('./routes/sensor_data');
const WebSocketHandler = require('./websocket');

const app = express();
const PORT = process.env.PORT || 3001;
const server = http.createServer(app);

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, '../client/build')));

app.use('/api/chat', chatRoutes);
app.use('/api/ingest', ingestRoutes);
app.use('/api/visualizations', visualizationRoutes);
app.use('/api/safety', safetyRoutes);
app.use('/api/data-collection', dataCollectionRoutes);
app.use('/api/sensor-data', sensorDataRoutes);

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../client/build', 'index.html'));
});

// Initialize WebSocket handler
const wsHandler = new WebSocketHandler(server);

// Setup WebSocket integration for safety monitoring
if (safetyRoutes.setupWebSocket) {
  safetyRoutes.setupWebSocket(wsHandler);
}

server.listen(PORT, () => {
  console.log(`🚀 WeaveHacks 2025 Lab Automation Server running on port ${PORT}`);
  console.log(`🔌 WebSocket available at ws://localhost:${PORT}/ws`);
  console.log(`🧬 Safety monitoring available at /api/safety`);
  console.log(`🎙️  Data collection available at /api/data-collection`);
  console.log(`⚙️  Claude-Flow mode: ${process.env.CLAUDE_FLOW_MODE || 'simulation'}`);
  console.log(`📊 Dashboard available at http://localhost:${PORT}`);
});