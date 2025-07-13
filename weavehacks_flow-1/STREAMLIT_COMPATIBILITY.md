# Streamlit Compatibility Guide

## Audio Input Compatibility

### Issue: `AttributeError: module 'streamlit' has no attribute 'audio_input'`

The `st.audio_input` widget was introduced in Streamlit 1.37.0 (August 2024). If you're using an older version, you'll get this error.

### Solutions

#### 1. **Upgrade Streamlit** (Recommended)
```bash
# Check your current version
python3 -c "import streamlit; print(streamlit.__version__)"

# Upgrade to latest version
pip3 install --user --upgrade streamlit>=1.37.0
```

#### 2. **Use the Built-in Fallback**
The integrated_app.py now includes automatic fallback:
- Detects if `audio_input` is available
- Falls back to file upload for older versions
- Shows clear instructions for both options

#### 3. **Check Your Version**
Run the version check script:
```bash
python3 check_streamlit_version.py
```

This will show:
- Your current Streamlit version
- Available features
- Recommendations for upgrades

## Feature Availability by Version

| Feature | Required Version | Fallback Available |
|---------|-----------------|-------------------|
| `st.audio_input` | 1.37.0+ | ✅ File upload |
| `st.chat_input` | 1.24.0+ | ✅ Text input |
| `st.chat_message` | 1.24.0+ | ✅ Container |
| `st.tabs` | 1.10.0+ | ✅ Radio buttons |
| `st.metric` | 0.81.0+ | ✅ Text display |
| `st.expander` | 0.68.0+ | ❌ None |
| `st.form` | 0.81.0+ | ❌ None |

## Voice Input Alternatives

### For Streamlit < 1.37.0:

1. **File Upload Method**:
   - Record audio on phone/computer
   - Save as WAV, MP3, OGG, or M4A
   - Upload via file uploader
   
2. **External Recording**:
   - Use phone voice recorder
   - Use computer's built-in recorder
   - Upload the file

3. **Browser Recording** (Advanced):
   - Use custom JavaScript component
   - Requires HTTPS for microphone access
   - See `voice_input_widget.py` for example

## Manual Data Entry Fallback

If voice input isn't working, the app provides manual entry:

1. Select compound from dropdown
2. Enter numerical value
3. Select units (g or mL)
4. Click "Record Data"

## Testing Audio Features

```python
# Test if audio_input is available
import streamlit as st

if hasattr(st, 'audio_input'):
    print("✅ audio_input is available")
else:
    print("❌ audio_input not available")
    print(f"   Current version: {st.__version__}")
    print("   Required version: 1.37.0+")
```

## Common Issues and Fixes

### 1. **Old Streamlit Version**
```bash
# Fix: Upgrade
pip3 install --user --upgrade streamlit
```

### 2. **Whisper Not Installed**
```bash
# Fix: Install whisper (optional for voice)
pip3 install --user openai-whisper
```

### 3. **No Microphone Access**
- Browser may block microphone
- Use HTTPS (not HTTP)
- Or use file upload method

### 4. **Audio Format Issues**
- Supported: WAV, MP3, OGG, M4A, WEBM
- Best: WAV or MP3
- Keep recordings under 30 seconds

## Minimal Requirements

The app will work with:
- Streamlit >= 1.32.0 (basic features)
- Python >= 3.9
- No audio dependencies (manual entry only)

## Full Feature Requirements

For all features including voice:
- Streamlit >= 1.37.0
- openai-whisper (for transcription)
- sounddevice (for audio recording)
- Python >= 3.9

## Checking Compatibility

Run this to check all dependencies:
```bash
# Check everything
python3 check_streamlit_version.py

# Quick version check
python3 -c "import streamlit as st; print(f'Streamlit {st.__version__}')"
```