# Claude-Flow Web App Setup Guide

## Prerequisites

1. **macOS** (tested on your current system)
2. **Node.js 16+** and npm
3. **Python 3.8+** with pip
4. **Claude-Flow CLI** installed and configured with your credentials
5. **Homebrew** (for Python package management)

## Quick Start

```bash
# 1. Clone and enter the directory
cd /Users/User/Documents/Source/Hackathon

# 2. Run the setup script
./start.sh
```

The app will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:3001
- WebSocket: ws://localhost:3001/ws

## Manual Setup

### 1. Install Python Dependencies

```bash
# Option A: Use our setup script
./setup-python.sh

# Option B: Manual installation
pip3 install matplotlib pandas numpy scipy seaborn plotly
```

### 2. Configure Claude Access

The app automatically uses your existing Claude-Flow installation and credentials. No API key configuration needed!

To verify Claude-Flow is working:
```bash
claude-flow --version
```

### 3. Install Node Dependencies

```bash
# Backend
npm install

# Frontend
cd client && npm install && cd ..
```

### 4. Start the Application

```bash
# Development mode (with hot reload)
npm run dev

# Production mode
./build.sh
npm start
```

## Features

### 1. Chat Interface
- Real-time streaming responses via WebSocket
- Code syntax highlighting
- Inline visualization display

### 2. Data Ingestion
- Upload JSON files through the UI
- Paste JSON data directly
- Files stored in `/uploads` directory

### 3. Code Execution
- Python code execution with matplotlib support
- Automatic visualization generation
- Results displayed in chat

### 4. Voice Interface (Coming Soon)
- See VOICE_ARCHITECTURE.md for planned features

## Usage Examples

### Analyzing Data
1. Upload a JSON file via the "Data Ingestion" tab
2. In chat, ask: "Analyze the sales_data.json file"
3. Claude will process and provide insights

### Creating Visualizations
1. Upload data first
2. Ask: "Create a bar chart of monthly revenue from sales_data.json"
3. The visualization will appear inline

### Running Custom Code
1. Ask: "Write Python code to calculate correlation matrix"
2. Claude will write and execute the code
3. Results and visualizations appear in chat

## Troubleshooting

### WebSocket Connection Issues
- Check if port 3001 is available
- Ensure no firewall blocking WebSocket connections
- The app falls back to HTTP if WebSocket fails

### Claude-Flow Not Working
- Verify `claude-flow` command works in terminal
- Check `CLAUDE_FLOW_MODE=native` in .env file
- Ensure your Claude credentials are configured

### Python Execution Errors
- Install required packages: `pip3 install -r requirements.txt`
- Check Python 3 is available as `python3`
- Verify matplotlib backend with: `python3 -c "import matplotlib; print(matplotlib.get_backend())"`

## Port Configuration

Default ports:
- Frontend dev server: 3000
- Backend API: 3001

To change ports, edit:
- Backend: `.env` file (PORT=3001)
- Frontend: `package.json` (PORT=3000)

## Security Notes

- This is a demo app with no authentication
- All data is stored locally
- Claude credentials are accessed from your system
- Not intended for production use

## Development

### Adding New Features
1. Backend routes in `/server/routes/`
2. Frontend components in `/client/src/components/`
3. Claude-Flow integration in `/server/claude-flow/`

### Running Tests
```bash
npm test
```

### Code Style
```bash
npm run lint
```

## Known Limitations

1. Single user only (no multi-tenancy)
2. Local file storage (no database)
3. Basic error handling
4. Limited to JSON data format
5. Visualizations are static PNGs

## Next Steps

1. Implement voice interface (see VOICE_ARCHITECTURE.md)
2. Add support for more data formats (CSV, Excel)
3. Enhance visualization options
4. Add data persistence
5. Implement user sessions