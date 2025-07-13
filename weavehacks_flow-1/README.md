# WeaveHacks Lab Automation Platform

A comprehensive AI-powered laboratory automation platform for nanoparticle synthesis experiments, featuring voice recognition, video monitoring, safety systems, and real-time data collection.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the backend
cd backend && python -m uvicorn main:app --reload

# Run the Streamlit interface
streamlit run integrated_app.py
```

## 📁 Project Structure

```
weavehacks_flow-1/
├── src/weavehacks_flow/          # Core application code
│   ├── agents/                   # AI agents for lab automation
│   ├── utils/                    # Utility functions and APIs
│   ├── config/                   # Configuration settings
│   └── integrations/             # External API integrations
├── backend/                      # FastAPI backend server
├── frontend/components/          # Streamlit UI components
├── tests_new/                    # All test files
├── examples/                     # Demo scripts and utilities
├── integrated_app.py             # Main Streamlit application
├── integration_bridge.py         # Component integration layer
└── requirements.txt              # Python dependencies
```

## 🧪 Core Features

### Laboratory Automation Agents

**Data Collection Agent**
- Voice-powered data entry using OpenAI Whisper
- Automatic compound recognition and value extraction
- CSV export and state management
- Real-time experiment tracking

**Safety Monitoring Agent**
- Real-time parameter monitoring (temperature, pressure, pH)
- Automatic alert generation for threshold violations
- Emergency halt procedures
- Configurable safety thresholds per experiment type

**Video Monitoring Agent**
- OpenCV-based video capture and analysis
- Motion detection and color change monitoring
- Liquid level tracking
- Automated screenshot capture for documentation

**Lab Control Agent**
- Equipment control and automation
- Instrument status monitoring
- Protocol step management
- Automated workflow execution

### AI Assistant Integration

**Claude Interface**
- Intelligent experiment guidance
- Real-time troubleshooting assistance
- Protocol optimization suggestions
- Scientific literature integration

**Voice Recognition**
- OpenAI Whisper integration for speech-to-text
- Natural language processing for measurement extraction
- Multi-language support
- Noise filtering and audio preprocessing

### Web Interface

**Streamlit Dashboard**
- Real-time experiment monitoring
- Interactive data visualization
- Voice-enabled data entry
- Protocol step tracking
- Safety alert display

**Backend API**
- RESTful FastAPI server
- Real-time data persistence
- Chemistry calculation endpoints
- Experiment state management

## 🔧 Installation & Setup

### System Requirements

- Python 3.9+
- Streamlit 1.37.0+ (for audio input support)
- OpenCV (for video monitoring)
- OpenAI Whisper (for voice recognition)
- FastAPI + Uvicorn (for backend)

### Detailed Installation

1. **Clone and navigate to project:**
   ```bash
   cd weavehacks_flow-1
   ```

2. **Install core dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install optional components:**
   ```bash
   # For voice recognition
   pip install openai-whisper
   
   # For advanced audio processing
   pip install sounddevice soundfile
   
   # For W&B logging (optional)
   pip install wandb weave
   ```

4. **Setup environment variables:**
   ```bash
   # Optional: Set API keys for AI services
   export ANTHROPIC_API_KEY="your_claude_api_key"
   export GOOGLE_API_KEY="your_gemini_api_key"
   export WEAVE_API_KEY="your_wandb_api_key"
   ```

### Hardware Setup

**Camera Access (macOS):**
```bash
# Reset camera permissions if needed
tccutil reset Camera
```

**Microphone Setup:**
- Ensure microphone permissions are granted
- Use HTTPS for web-based microphone access
- Test with `examples/diagnose_audio.py`

## 🎯 Usage Guide

### Starting the System

1. **Start Backend Server:**
   ```bash
   cd backend
   python -m uvicorn main:app --reload --port 8000
   ```

2. **Launch Web Interface:**
   ```bash
   streamlit run integrated_app.py
   ```

3. **Access Application:**
   - Open browser to `http://localhost:8501`
   - Backend API docs at `http://localhost:8000/docs`

### Basic Workflow

1. **Create Experiment**
   - Go to Dashboard tab
   - Enter experiment ID (e.g., "exp_001")
   - Click "Create Experiment"

2. **Record Data**
   - Use Voice Entry tab for speech input
   - Say: "Gold mass is 0.1576 grams"
   - Or use manual data entry form

3. **Monitor Safety**
   - Safety Monitor tab shows real-time alerts
   - Configure thresholds per experiment type
   - Emergency halt available

4. **Track Progress**
   - Protocol Steps tab shows current step
   - Data Panel displays all recorded measurements
   - AI Assistant provides guidance

### Voice Commands

The system recognizes natural speech patterns:

```
"Gold mass is 0.1576 grams"
"TOAB mass is 0.25 g"
"Toluene volume is 10 milliliters"
"Final nanoparticle mass is 0.08 grams"
```

**Supported Compounds:**
- Gold (HAuCl₄·3H₂O)
- TOAB
- Sulfur (PhCH₂CH₂SH)
- NaBH₄
- Toluene
- Nanopure water
- Au₂₅ nanoparticles

**Supported Units:**
- Mass: grams, g
- Volume: milliliters, ml, mL

## 🧪 Chemistry Calculations

### Automated Calculations

The platform includes built-in chemistry calculations:

- **Sulfur Amount**: Calculate PhCH₂CH₂SH needed based on gold mass
- **NaBH₄ Amount**: Calculate reducing agent quantity
- **Percent Yield**: Compare actual vs theoretical yield
- **TOAB Ratio**: Phase transfer catalyst calculations

### Custom Calculations

Add new calculations in `src/weavehacks_flow/utils/chemistry_calculations.py`:

```python
@weave_op()
@safe_execute
def custom_calculation(input_mass: float) -> Dict[str, float]:
    """Your custom chemistry calculation"""
    result = input_mass * some_factor
    return {"output": result, "units": "g"}
```

## 🔍 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests_new/ -v

# Run specific test categories
python -m pytest tests_new/test_chemistry_calculations.py -v
python -m pytest tests_new/test_voice_recognition_agent.py -v

# Run with coverage
python -m pytest tests_new/ --cov=src/weavehacks_flow --cov-report=html
```

### Test Categories

- **Chemistry Calculations**: Validate all chemistry functions
- **Voice Recognition**: Test speech processing and transcription
- **Video Monitoring**: Camera and image analysis tests
- **API Integration**: Backend endpoint validation
- **Agent Functionality**: Individual agent behavior tests

### Manual Testing

Use example scripts for manual validation:

```bash
# Test voice recognition
python examples/diagnose_audio.py

# Test system dependencies
python examples/check_dependencies.py

# Test Streamlit compatibility
python examples/check_streamlit_version.py
```

## 🔧 Configuration

### Experiment Types

Configure in `src/weavehacks_flow/config/settings.py`:

```python
EXPERIMENT_TYPES = {
    "gold_nanoparticle_rt": {
        "name": "Gold Nanoparticle Synthesis (Room Temp)",
        "temp_range": (20.0, 25.0),
        "pressure_range": (100.0, 102.0),
        "ph_range": (6.0, 8.0)
    }
}
```

### Safety Thresholds

Customize safety monitoring parameters:

```python
SAFETY_THRESHOLDS = {
    "temperature": {"min": 15.0, "max": 30.0, "units": "°C"},
    "pressure": {"min": 99.0, "max": 103.0, "units": "kPa"},
    "ph": {"min": 5.0, "max": 9.0, "units": "pH"}
}
```

### Chemistry Constants

Modify molecular weights and calculations:

```python
# Molecular weights (g/mol)
MW_HAuCl4_3H2O = 393.83
MW_Au = 196.97
MW_TOAB = 546.78
MW_NaBH4 = 37.83
```

## 🐛 Troubleshooting

### Common Issues

**Streamlit Audio Input Error:**
```
AttributeError: module 'streamlit' has no attribute 'audio_input'
```
**Solution:** Upgrade Streamlit:
```bash
pip install --upgrade streamlit>=1.37.0
```

**Backend Connection Failed:**
```
Connection Error: HTTPConnectionPool
```
**Solution:** Start the backend server:
```bash
cd backend && python -m uvicorn main:app --reload
```

**Voice Recognition Not Working:**
- Check microphone permissions
- Install Whisper: `pip install openai-whisper`
- Use file upload as fallback

**Camera Access Denied (macOS):**
```bash
tccutil reset Camera
# Then restart terminal and grant permissions
```

### Debug Information

The Voice Entry tab includes a debug panel showing:
- Current experiment ID
- Backend connectivity status
- Platform availability
- Audio input capability
- Streamlit version

### API Troubleshooting

**API Error 404:**
- Experiment doesn't exist
- Create experiment first in Dashboard

**API Error 500:**
- Backend server error
- Check backend logs
- Validate data format

**API Error 400:**
- Invalid request data
- Check experiment ID format
- Validate measurement values

## 📊 Data Management

### Experiment Data Structure

```json
{
  "experiment_id": "exp_001",
  "status": "in_progress",
  "step_num": 3,
  "mass_gold": 0.1576,
  "mass_toab": 0.25,
  "volume_toluene": 10.0,
  "safety_status": "safe",
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-01T00:30:00Z"
}
```

### Data Export

Export experiment data:
- CSV format for spreadsheet analysis
- JSON format for programmatic access
- Real-time API access via REST endpoints

### Backup and Recovery

- Automatic state persistence in backend
- Manual export via Data Panel
- API-based data retrieval

## 🤖 AI Integration

### Claude Assistant

Configure Anthropic API access:
```bash
export ANTHROPIC_API_KEY="your_api_key"
```

Provides:
- Intelligent experiment guidance
- Real-time troubleshooting
- Protocol optimization
- Literature recommendations

### Gemini Integration

Configure Google AI access:
```bash
export GOOGLE_API_KEY="your_api_key"
```

Features:
- Alternative AI model option
- Multi-modal understanding
- Advanced reasoning capabilities

### W&B Weave Logging

Optional experiment tracking:
```bash
export WEAVE_API_KEY="your_wandb_api_key"
```

Enables:
- Experiment versioning
- Performance monitoring
- Collaborative analysis
- Model tracking

## 🛡️ Safety & Security

### Safety Monitoring

- **Real-time parameter tracking**
- **Automated alert generation**
- **Emergency halt procedures**
- **Configurable thresholds**
- **Alert escalation protocols**

### Security Considerations

- **API key management**: Store in environment variables
- **Input validation**: All user inputs are sanitized
- **Error handling**: Graceful failure modes
- **Access control**: Local deployment recommended
- **Data privacy**: No automatic cloud upload

### Best Practices

1. **Always validate measurements** before proceeding
2. **Monitor safety alerts** continuously during experiments
3. **Keep backup recordings** of voice entries
4. **Regular system health checks** using diagnostic tools
5. **Update dependencies** regularly for security patches

## 📈 Performance Optimization

### System Performance

- **Memory usage**: ~500MB for full system
- **CPU usage**: <10% during normal operation
- **Storage**: ~1GB for models and data
- **Network**: Minimal (local API calls only)

### Optimization Tips

1. **Preload Whisper model** for faster voice processing
2. **Use smaller model sizes** for resource-constrained systems
3. **Enable caching** for repeated calculations
4. **Optimize video frame rates** based on needs
5. **Regular cleanup** of temporary files

## 🔄 Development

### Adding New Features

1. **Create agent**: Extend base agent classes
2. **Add API endpoints**: Update backend/main.py
3. **Update UI**: Add Streamlit components
4. **Write tests**: Add comprehensive test coverage
5. **Update documentation**: Modify this README

### Code Standards

- **Type hints**: Use throughout codebase
- **Error handling**: Comprehensive exception management
- **Logging**: Structured logging with appropriate levels
- **Testing**: >80% code coverage target
- **Documentation**: Docstrings for all public functions

### Contributing

1. Fork the repository
2. Create feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Update documentation
6. Submit pull request

## 📋 TODO & Roadmap

### Current TODO Items

- [ ] **Enhanced Error Recovery**: Automatic retry mechanisms for failed operations
- [ ] **Advanced Analytics**: Statistical analysis of experimental data
- [ ] **Protocol Templates**: Pre-defined experiment workflows
- [ ] **Mobile Interface**: Responsive design for tablet/phone access
- [ ] **Batch Processing**: Support for multiple parallel experiments
- [ ] **Advanced Visualizations**: 3D plotting and interactive charts
- [ ] **Integration Testing**: End-to-end workflow validation
- [ ] **Performance Benchmarks**: Systematic performance testing
- [ ] **User Authentication**: Multi-user support with role management
- [ ] **Cloud Deployment**: Docker containerization and cloud setup guides

### Future Enhancements

- [ ] **Machine Learning**: Predictive models for experiment outcomes
- [ ] **IoT Integration**: Direct sensor integration via protocols
- [ ] **Advanced AI**: Custom model training for lab-specific tasks
- [ ] **Workflow Automation**: Fully automated experimental pipelines
- [ ] **Collaborative Features**: Multi-user real-time collaboration
- [ ] **Advanced Safety**: Predictive safety monitoring with ML
- [ ] **Custom Hardware**: Integration with lab-specific equipment
- [ ] **Regulatory Compliance**: GLP/GMP compliance features

### Recent Updates

- ✅ **Enhanced Voice Processing**: Comprehensive error handling and validation
- ✅ **API Robustness**: Detailed error messages and recovery suggestions
- ✅ **Streamlit Compatibility**: Full support for audio input across versions
- ✅ **Code Restructuring**: Organized project structure and documentation
- ✅ **ArXiv Integration**: Enhanced scientific paper search functionality
- ✅ **Test Suite**: Comprehensive testing framework with 70+ tests
- ✅ **Documentation**: Complete user guide and API documentation

## 🆘 Support

### Getting Help

1. **Check this README** for common solutions
2. **Review debug information** in the Voice Entry tab
3. **Check backend logs** for detailed error messages
4. **Run diagnostic scripts** in the examples/ directory
5. **Consult API documentation** at `/docs` endpoint

### Reporting Issues

When reporting bugs, include:
- Python version and OS
- Streamlit version
- Complete error messages
- Steps to reproduce
- System configuration

### Community

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and community support
- **Wiki**: Additional documentation and tutorials

---

*WeaveHacks Lab Automation Platform - Advancing Scientific Research Through AI*