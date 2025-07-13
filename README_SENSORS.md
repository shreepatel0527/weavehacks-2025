# 🌡️ Real-Time Sensor Monitoring System

## 🎯 Overview

The WeaveHacks Lab Automation Platform now includes **real-time temperature and pressure monitoring** with simulated sensor data, safety alerts, and live visualization.

## 🔥 New Features Added

### **🔬 Sensor Simulation Backend**
- `backend/sensor_simulator.py` - Realistic sensor data generation
- Multiple sensor types: Temperature, Pressure, Gas levels
- Experiment-specific behavior profiles
- Safety threshold monitoring
- Background data generation (5-second intervals)

### **📊 Live Frontend Dashboard**
- Real-time temperature and pressure charts
- Safety threshold visualization (warning/critical lines)
- Multi-sensor display with location tracking
- Auto-refresh capability
- Color-coded safety alerts

### **🛡️ Safety Monitoring**
- Automatic threshold checking
- Warning (90°C temp, 115 kPa pressure)
- Critical (100°C temp, 125 kPa pressure)
- Real-time alert generation
- Visual safety indicators

## 🚀 Quick Demo

### 1. Start the Platform
```bash
# Terminal 1 - Backend with sensors
cd backend
uvicorn main:app --reload

# Terminal 2 - Frontend
streamlit run integrated_app.py
```

### 2. Test Sensor Simulation
```bash
# Run automated demo
python demo_sensor_data.py
```

### 3. Use the Web Interface
1. Open http://localhost:8501
2. Go to **Dashboard** tab → Create experiment
3. Go to **Safety Monitor** tab → Start sensor monitoring
4. Watch live temperature/pressure data!

## 📊 Sensor Data Features

### **Temperature Monitoring**
- **TEMP_001**: Reaction vessel (primary sensor)
- **TEMP_002**: Fume hood (ambient monitoring)
- Range: 20-150°C depending on experiment
- Realistic heating/cooling cycles

### **Pressure Monitoring**
- **PRESS_001**: Reaction vessel pressure
- **PRESS_002**: Nitrogen line pressure
- Range: 98-125 kPa
- Pressure follows temperature changes

### **Gas Level Monitoring**
- **GAS_001**: Fume hood gas detector
- Range: 150-5000+ ppm
- Gradual increase during experiments

## 🧪 Experiment Profiles

### **Nanoparticle Synthesis** (Default)
- Temperature: 20-85°C
- Pressure: 100-105 kPa
- Duration: 4 hours
- Gradual heating, stable reaction, cooling

### **High Temperature Reaction**
- Temperature: 25-150°C
- Pressure: 98-110 kPa
- Duration: 6 hours
- Extended high-temperature phase

### **Pressure Reaction**
- Temperature: 40-80°C
- Pressure: 100-120 kPa
- Duration: 8 hours
- Elevated pressure throughout

## 🔌 API Endpoints

### Sensor Control
```bash
# Start monitoring
POST /sensors/start-experiment?experiment_type=nanoparticle_synthesis&experiment_id=exp_001

# Get live status
GET /sensors/status

# Get recent readings
GET /sensors/readings?count=50

# Stop monitoring
POST /sensors/stop

# Get experiment types
GET /sensors/experiment-types
```

### Example Response
```json
{
  "experiment": {
    "active": true,
    "experiment_type": "Au Nanoparticle Synthesis",
    "elapsed_hours": 0.25,
    "progress_percent": 6.25
  },
  "recent_readings": [
    {
      "sensor_id": "TEMP_001",
      "sensor_type": "temperature",
      "value": 45.2,
      "units": "°C",
      "timestamp": "2025-01-13T15:30:45",
      "location": "reaction_vessel"
    }
  ],
  "safety_alerts": []
}
```

## 🎨 Frontend Interface

### **Safety Monitor Tab**
1. **Experiment Controls**
   - Select experiment type dropdown
   - Start/Stop monitoring buttons
   - Progress tracking

2. **Live Charts**
   - Temperature line charts with thresholds
   - Pressure monitoring with safety zones
   - Multi-sensor visualization

3. **Current Metrics**
   - Real-time sensor values
   - Color-coded safety status
   - Trend indicators

4. **Safety Alerts**
   - Warning (🟡) and Critical (🔴) alerts
   - Automatic threshold detection
   - Alert history display

### **Data Table**
- Expandable recent readings table
- Timestamp, sensor ID, values, location
- Sortable by time (newest first)

## ⚡ Real-Time Features

### **Auto-Refresh**
- Optional 10-second auto-refresh
- Live data updates without page reload
- Maintains chart history

### **Safety Monitoring**
- Continuous threshold checking
- Immediate visual alerts
- Color-coded metrics (green/yellow/red)

### **Data Persistence**
- Background sensor data collection
- Queue-based data storage
- Historical data retention

## 🧪 Realistic Sensor Behavior

### **Temperature Simulation**
```python
# Experiment phases
Initial heating (0-30 min): 20°C → 85°C
Stable reaction (30-120 min): 85°C ± 2°C
Cooling phase (120-180 min): 85°C → 35°C
Final stable (180+ min): 35°C ± 1°C
```

### **Pressure Simulation**
```python
# Pressure follows temperature with lag
Gradual increase (0-45 min): 100 → 105 kPa
High pressure (45-150 min): 105 ± 0.5 kPa
Pressure release (150+ min): 105 → 102 kPa
```

### **Noise and Variation**
- Realistic sensor noise (±1-2°C, ±0.5 kPa)
- Location-specific offsets
- Time-based variations

## 🚨 Safety Features

### **Threshold System**
```python
Temperature:
  Warning: 90°C
  Critical: 100°C

Pressure:
  Warning: 115 kPa  
  Critical: 125 kPa

Gas Levels:
  Warning: 1000 ppm
  Critical: 5000 ppm
```

### **Alert Actions**
- Visual warnings in UI
- API alert endpoints
- Safety status tracking
- Automatic experiment flagging

## 🔧 Configuration

### **Sensor Settings**
```python
# Add new sensors in sensor_simulator.py
sensors = {
    "TEMP_003": {"type": SensorType.TEMPERATURE, "location": "cooling_bath"},
    "PRESS_003": {"type": SensorType.PRESSURE, "location": "vacuum_line"},
}
```

### **Safety Thresholds**
```python
# Modify thresholds in sensor_simulator.py
safety_thresholds = {
    SensorType.TEMPERATURE: {"warning": 85, "critical": 95},
    SensorType.PRESSURE: {"warning": 110, "critical": 120},
}
```

## 📈 Data Visualization

### **Plotly Charts**
- Interactive time-series plots
- Multiple sensor traces
- Safety threshold lines
- Hover information

### **Metrics Display**
- Streamlit metric components
- Delta indicators
- Color-coded status

### **Layout**
- Responsive design
- Multi-column layouts
- Expandable sections

## 🎯 Use Cases

### **1. Real-Time Monitoring**
```
Scientist starts nanoparticle synthesis
→ Selects "nanoparticle_synthesis" experiment
→ Clicks "Start Sensor Monitoring"
→ Watches live temperature/pressure charts
→ Gets safety alerts if thresholds exceeded
```

### **2. Safety Compliance**
```
Temperature reaches 92°C (above 90°C warning)
→ System generates warning alert
→ Yellow indicator appears
→ Scientist adjusts heating parameters
```

### **3. Data Analysis**
```
Experiment completes after 4 hours
→ Reviews temperature/pressure trends
→ Analyzes heating/cooling cycles
→ Validates experiment parameters
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI       │    │  Sensor         │
│   Frontend      │◄──►│   Backend       │◄──►│  Simulator      │
│                 │    │                 │    │                 │
│ • Live Charts   │    │ • REST APIs     │    │ • Data Gen      │
│ • Safety UI     │    │ • Experiment    │    │ • Threading     │
│ • Controls      │    │ • Monitoring    │    │ • Profiles      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Future Enhancements

### **Planned Features**
- WebSocket real-time streaming
- Historical data export
- Custom alert rules
- Multi-experiment monitoring
- Mobile-responsive design

### **Advanced Monitoring**
- pH sensors
- Flow rate monitoring
- Vibration detection
- Camera integration

This sensor monitoring system provides a comprehensive foundation for lab automation with real-time safety monitoring and data visualization!