# Manual Testing Guide for WeaveHacks Lab Assistant

## 🎯 Overview
This guide provides step-by-step instructions for manually testing the video monitoring and voice recognition features of the WeaveHacks Lab Assistant.

## 🚀 Quick Start

### Prerequisites
1. Python 3.9+ installed
2. Microphone connected and permissions granted
3. Camera/webcam available (for video features)
4. Dependencies installed: `pip install -r requirements.txt`

### Running the Application
```bash
# Navigate to project directory
cd /Users/User/Documents/Source/Weavehacks-2025-Base

# Run the Streamlit UI
python3 run_lab_assistant.py ui

# Alternative: Run unified lab assistant directly
python3 unified_lab_assistant.py
```

## 🎤 Voice Recognition Testing

### 1. Basic Voice Input Test
1. **Launch the application**
   ```bash
   python3 unified_lab_assistant.py
   ```

2. **Navigate to AI Assistant tab**
   - Look for the "🎙️ AI Assistant" tab
   - Select your preferred AI model (Claude or Gemini)

3. **Test microphone access**
   - Click the microphone icon or audio input option
   - Speak clearly: "Testing voice recognition"
   - Verify the text appears in the chat

### 2. Voice Commands for Data Collection
Test these voice commands:
- "The mass is 2.5 grams"
- "Temperature reading 23.5 celsius"
- "Adding 10 milliliters of solution"
- "pH level is 7.2"
- "Observing yellow precipitate formation"

**Expected behavior**: The system should extract numerical values and units correctly.

### 3. Audio Diagnostics
If voice recognition fails:
```bash
# Run audio diagnostics
cd weavehacks_flow-1
python3 diagnose_audio.py
```

This will show:
- Available audio devices
- Current default device
- Permission status
- Troubleshooting recommendations

### 4. Testing Different Audio Devices
1. **List available devices**: Run diagnostics to see device list
2. **Switch devices**: In the app, look for audio device selection
3. **Test each device**: Try recording with different microphones

### 5. Continuous Voice Monitoring
1. Start an experiment in the app
2. Enable "Continuous Voice Mode" if available
3. Speak measurements as you work:
   - "Starting titration"
   - "Added 5 ml of reagent"
   - "Color changed to pink"
   - "End point reached at 23.5 ml"

## 📹 Video Monitoring Testing

### 1. Camera Setup Test
1. **Check camera availability**
   ```python
   # Test script to verify camera
   import cv2
   cap = cv2.VideoCapture(0)
   if cap.isOpened():
       print("Camera available!")
       cap.release()
   else:
       print("Camera not found")
   ```

2. **In the application**
   - Look for video monitoring features
   - Check if camera preview is visible

### 2. Motion Detection Test
1. **Setup**: Position camera to view experiment area
2. **Start monitoring**: Enable video monitoring in the app
3. **Test motion detection**:
   - Move objects in camera view
   - Check if motion events are logged
   - Verify motion regions are highlighted

### 3. Color Change Detection
1. **Prepare colored solutions** (if available)
2. **Set reference color**: Point camera at initial solution
3. **Add reagent** to change color
4. **Verify**: System detects and logs color change

### 4. Overnight Monitoring Simulation
1. **Start experiment monitoring**
   ```python
   # In the app, look for "Start Overnight Monitoring"
   # Set experiment ID: "test_overnight_001"
   ```

2. **Simulate events**:
   - Move objects periodically
   - Change lighting conditions
   - Introduce color changes

3. **Check results**:
   - View event log
   - Check recorded video (if available)
   - Review motion/change statistics

### 5. Recording Test
1. **Start recording**: Look for record button in video panel
2. **Perform actions** for 30 seconds
3. **Stop recording**
4. **Verify**: Check recordings folder for output file

## 🧪 Integration Testing

### 1. Voice + Data Collection
1. Navigate to "📊 Data Collection" tab
2. Use voice to input measurements:
   - "Gold chloride mass is 0.1576 grams"
   - "TOAB mass is 0.2534 grams"
3. Verify values appear in correct fields

### 2. Safety Monitoring Integration
1. Go to "🚨 Safety" tab
2. Test voice alerts:
   - "Temperature is too high"
   - "Pressure exceeding limits"
3. Verify safety warnings appear

### 3. Full Experiment Workflow
1. **Start new experiment**
2. **Follow protocol steps** using voice:
   - Record measurements vocally
   - Note observations
   - Navigate steps hands-free
3. **Monitor safety parameters**
4. **Export data** at completion

## 🔧 Troubleshooting

### Voice Recognition Issues

| Problem | Solution |
|---------|----------|
| "No microphone found" | Grant permissions in system settings |
| "Audio device error -1" | Run `python3 diagnose_audio.py` |
| Poor transcription | Speak clearly, reduce background noise |
| No voice option visible | Check if whisper model is installed |

### Video Monitoring Issues

| Problem | Solution |
|---------|----------|
| "Camera not found" | Check camera connection and permissions |
| Black video preview | Try different camera index (0, 1, 2...) |
| Motion not detected | Adjust sensitivity in settings |
| High CPU usage | Lower FPS or resolution |

### Quick Fixes

1. **Restart audio services**:
   - macOS: `sudo killall coreaudiod`
   - Windows: Restart Windows Audio service
   - Linux: `pulseaudio --kill && pulseaudio --start`

2. **Camera permissions**:
   - macOS: System Preferences > Security & Privacy > Camera
   - Windows: Settings > Privacy > Camera
   - Linux: Check user groups (`groups $USER`)

## 📊 Performance Testing

### Voice Recognition Benchmarks
- **Expected latency**: < 2 seconds for 5-second audio
- **Accuracy target**: > 90% for clear speech
- **Memory usage**: < 200MB additional

### Video Monitoring Benchmarks
- **Frame rate**: 30 FPS for 640x480
- **Motion detection**: < 100ms per frame
- **CPU usage**: < 30% single core

## 🎯 Test Scenarios

### Scenario 1: Wet Lab Synthesis
1. Start app and select "Nanoparticle Synthesis" protocol
2. Use voice to record each measurement
3. Monitor temperature via video feed
4. Note color changes vocally
5. Export complete dataset

### Scenario 2: Overnight Reaction
1. Set up camera facing reaction vessel
2. Start overnight monitoring
3. Leave running for 2+ hours
4. Review motion/change events
5. Check for anomalies in log

### Scenario 3: Multi-Modal Input
1. Enable both voice and video
2. Speak while performing actions
3. Verify synchronized logging
4. Test event correlation

## 📝 Reporting Issues

When reporting issues, include:
1. **System info**: OS, Python version
2. **Error messages**: Full traceback
3. **Device info**: Microphone/camera models
4. **Steps to reproduce**
5. **Log files** from `logs/` directory

## ✅ Testing Checklist

### Initial Setup
- [ ] Python environment configured
- [ ] Dependencies installed
- [ ] Microphone permissions granted
- [ ] Camera permissions granted
- [ ] Application launches successfully

### Voice Recognition
- [ ] Basic voice input works
- [ ] Numerical extraction accurate
- [ ] Multiple phrasings understood
- [ ] Error messages helpful
- [ ] Fallback to text input works

### Video Monitoring
- [ ] Camera preview visible
- [ ] Motion detection triggers
- [ ] Color changes detected
- [ ] Recording saves files
- [ ] Events logged correctly

### Integration
- [ ] Voice + data collection works
- [ ] Safety alerts function
- [ ] Multi-agent coordination
- [ ] Data export successful
- [ ] Performance acceptable

## 🎉 Success Criteria

The system is working correctly when:
1. ✅ Voice commands are recognized with >90% accuracy
2. ✅ Video monitoring captures all significant events
3. ✅ Data is accurately recorded and stored
4. ✅ Safety monitoring provides timely alerts
5. ✅ System remains stable during extended use

---

**Need Help?** 
- Check `AUDIO_TROUBLESHOOTING.md` for audio issues
- Review `INTEGRATION_TEST_STRATEGY.md` for technical details
- Run automated tests: `python3 -m pytest tests/ -v`