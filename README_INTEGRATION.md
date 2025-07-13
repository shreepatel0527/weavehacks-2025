# WeaveHacks 2025 - Integrated Lab Automation Platform

## 🎯 Overview

This integration connects the **Prototype-1 Streamlit frontend** with the **weavehacks_flow-1 backend agents** to create a full-stack lab automation platform for nanoparticle synthesis.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI API   │    │  Agent System   │
│   (Frontend)    │◄──►│   (Backend)     │◄──►│  (Processing)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
   • Voice Input         • REST Endpoints       • Data Collection
   • Data Dashboard      • Experiment Mgmt      • Lab Control  
   • Safety Monitor      • Chemistry Calcs      • Safety Monitor
   • AI Assistant        • Real-time APIs       • Weave Tracking
```

## 🚀 Quick Start

### 1. Start the Integrated Platform

```bash
# Option 1: Use the startup script
./start_integrated_platform.sh

# Option 2: Manual startup
# Terminal 1 - Backend API
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2 - Frontend UI  
streamlit run integrated_app.py --server.port 8501
```

### 2. Access the Applications

- **Frontend UI**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 🧪 Core Features

### **Experiment Management**
- Create and track nanoparticle synthesis experiments
- Real-time progress monitoring (Step X/12)
- Data persistence via REST API

### **Voice Data Entry**
- Speech-to-text using Whisper
- Natural language data recording
- Example: *"Gold mass is 0.1576 grams"*

### **Chemistry Calculations**
- Automated stoichiometry calculations
- Sulfur compound amount (3 eq. relative to gold)
- NaBH₄ amount (10 eq. relative to gold)  
- Percent yield calculations

### **Safety Monitoring**
- Real-time parameter alerts
- Temperature and pressure thresholds
- Automated safety notifications

### **AI Lab Assistant**
- Claude and Gemini integration
- Context-aware experiment guidance
- Protocol assistance and troubleshooting

## 🔗 Component Integration

### **Frontend (integrated_app.py)**
- Streamlit-based UI with 4 main tabs
- Voice recording and transcription
- Real-time data visualization
- API client for backend communication

### **Backend (backend/main.py)**
- FastAPI REST API with 15+ endpoints
- Experiment state management
- Chemistry calculation engine
- Safety alert system

### **Integration Bridge (integration_bridge.py)**
- Handles import path resolution
- Connects Prototype-1 and weavehacks_flow-1
- Provides fallback implementations

### **Agent System**
- **DataCollectionAgent**: Voice-to-data recording
- **LabControlAgent**: Instrument automation
- **SafetyMonitoringAgent**: Real-time monitoring

## 📊 API Endpoints

### Experiments
- `POST /experiments` - Create experiment
- `GET /experiments/{id}` - Get experiment details
- `PUT /experiments/{id}/status` - Update status

### Data Collection  
- `POST /data` - Record measurement data
- `GET /experiments/{id}/data` - Get experiment data

### Safety
- `POST /safety/alert` - Create safety alert
- `GET /safety/alerts` - Get all alerts

### Calculations
- `POST /calculations/sulfur-amount` - Calculate sulfur needed
- `POST /calculations/nabh4-amount` - Calculate NaBH₄ needed
- `POST /calculations/percent-yield` - Calculate yield

## 🧪 Testing

### Run Integration Tests
```bash
# Start backend first
cd backend && uvicorn main:app --reload &

# Run test suite
python test_integration.py
```

### Test Coverage
- ✅ Backend API health checks
- ✅ Experiment CRUD operations
- ✅ Data recording and retrieval
- ✅ Chemistry calculations
- ✅ Safety alert system
- ✅ Agent functionality
- ✅ Performance benchmarks

## 🛠️ Development

### Project Structure
```
weavehacks-2025/
├── integrated_app.py           # Main Streamlit application
├── integration_bridge.py       # Import path resolver
├── backend/                    # FastAPI backend
│   ├── main.py                # API server
│   └── requirements.txt       # Backend dependencies
├── Prototype-1/               # Original Streamlit frontend
├── weavehacks_flow-1/         # Agent system
└── test_integration.py        # Test suite
```

### Key Technologies
- **Frontend**: Streamlit, Plotly, Speech Recognition
- **Backend**: FastAPI, Pydantic, Uvicorn
- **Agents**: CrewAI, Weave, W&B
- **AI**: Claude CLI, Google Gemini API
- **Audio**: Whisper, SoundDevice

## 🔧 Configuration

### Environment Variables
```bash
# Optional - for enhanced features
export WANDB_API_KEY="your_wandb_key"
export GOOGLE_API_KEY="your_gemini_key"
```

### Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

## 📈 Monitoring

### Weave Integration
- All operations tracked with `@weave.op()` decorators
- Experiment workflows logged
- Agent interactions captured
- Performance metrics collected

### Real-time Monitoring
- Live experiment progress tracking
- Safety parameter monitoring
- Agent status dashboard
- API health checks

## 🎯 Use Cases

### 1. **Voice-Controlled Data Entry**
Scientist wearing gloves can record measurements hands-free:
```
"TOAB mass is 0.25 grams"
→ System records: TOAB = 0.25g in current experiment
```

### 2. **Automated Calculations** 
System calculates required amounts based on stoichiometry:
```
Gold: 0.1576g recorded
→ Calculates: Sulfur needed = 0.1675g (3 equivalents)
```

### 3. **Safety Monitoring**
Real-time alerts for dangerous conditions:
```
Temperature: 85°C (threshold: 80°C)
→ Generates warning alert
→ Notifies scientist
```

### 4. **AI Guidance**
Context-aware lab assistance:
```
User: "What's my next step?"
AI: "Based on your current experiment (Step 5/12), you should now calculate and weigh the PhCH₂CH₂SH. You need 0.1675g based on your gold mass."
```

## 🚨 Safety Features

- Real-time parameter monitoring
- Automated threshold checking  
- Emergency shutdown capabilities
- Alert escalation system
- Safety protocol guidance

## 📱 User Interface

### Dashboard Tab
- Experiment selection and creation
- Progress tracking with visual indicators
- Live data display with metrics
- Chemistry calculation buttons

### Voice Entry Tab  
- Audio recording interface
- Real-time transcription
- Manual data entry fallback
- Automatic data parsing

### Safety Monitor Tab
- Parameter threshold visualization
- Alert history and status
- Emergency controls
- Safety protocol reminders

### AI Assistant Tab
- Contextual chat interface
- Model selection (Claude/Gemini)
- Quick action buttons
- Experiment-aware responses

This integrated platform provides a comprehensive solution for lab automation, combining the intuitive Streamlit interface with powerful backend agents and real-time monitoring capabilities.