# Audio Troubleshooting Guide

## "Error querying device -1" Fix

This error occurs when the voice recognition system cannot access your microphone. Here's how to fix it:

### Quick Diagnosis

Run this command to get a comprehensive audio diagnosis:

```bash
python diagnose_audio.py
```

This will show you:
- Available audio devices
- Default device settings
- System-specific troubleshooting steps
- Suggested fixes

### Common Solutions

#### 1. **Grant Microphone Permissions**

**macOS:**
- Go to System Preferences > Security & Privacy > Privacy > Microphone
- Enable access for Terminal or your Python application

**Windows:**
- Go to Settings > Privacy > Microphone
- Enable "Allow apps to access your microphone"

**Linux:**
- Add your user to the audio group: `sudo usermod -a -G audio $USER`
- Restart your session

#### 2. **Check Hardware**
- Ensure microphone is connected and enabled
- Test microphone in other applications
- Try different USB ports (for USB microphones)

#### 3. **Restart Audio Services**

**macOS:**
```bash
sudo killall coreaudiod
```

**Windows:**
- Restart Windows Audio service in Services.msc

**Linux:**
```bash
pulseaudio --kill
pulseaudio --start
```

#### 4. **Install Audio Dependencies**

The system should auto-install, but if needed:

**macOS:**
```bash
brew install portaudio
```

**Ubuntu/Debian:**
```bash
sudo apt-get install portaudio19-dev libasound2-dev
```

**Windows:**
- Audio dependencies are included with Python packages

### Enhanced Voice Recognition Features

The voice recognition agent now includes:

- **Auto-detection** of audio issues
- **Graceful fallback** to manual input when voice fails
- **Device selection** options
- **Comprehensive diagnostics**
- **Helpful error messages**

### Manual Device Selection

If auto-detection fails, you can manually specify a device:

```python
from weavehacks_flow.agents.voice_recognition_agent import SpeechRecognizerAgent

# Create agent with specific device
agent = SpeechRecognizerAgent(device_id=1)

# Or set device after creation
agent.set_audio_device(device_id=1)
```

### Debugging Commands

```python
# Get device information
agent.get_device_info()

# Test microphone access
agent.test_microphone()

# Run full diagnostics
agent.run_audio_diagnostics()
```

### Integration with Data Collection

The data collection agent now automatically:
- Detects audio issues
- Provides helpful error messages  
- Falls back to manual input
- Guides users to run diagnostics

```
⚠️  Voice recognition not available. Audio system not properly initialized.
🔧 To troubleshoot, run: python diagnose_audio.py
📝 Falling back to manual input...
```

### Still Having Issues?

1. **Run full diagnosis**: `python diagnose_audio.py`
2. **Check the output** for specific recommendations
3. **Try different audio devices** if multiple are available
4. **Restart your system** if audio services are corrupted
5. **Check system audio settings** to ensure microphone is set as default

### Technical Details

The voice recognition system uses:
- **OpenAI Whisper** for speech recognition
- **SoundDevice** for audio capture
- **PortAudio** as the underlying audio library

Common issues stem from PortAudio not being able to access the default audio input device (device -1), which indicates system-level audio configuration problems.