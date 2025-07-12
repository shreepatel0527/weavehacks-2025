# AI Chat Interface

A minimal Streamlit web app that interfaces with Claude CLI and Google Gemini 2.5 Pro, with speech recognition support.

## Prerequisites

- Claude CLI installed and accessible via `claude -p` command
- Google Gemini API key stored at `$HOME/Documents/Ephemeral/gapi`
- Python 3.8 or higher
- Microphone access for speech input (optional)

## Installation

```bash
pip install -r requirements.txt
```

For speech recognition support, you'll also need ffmpeg:
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg`
- Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Running the App

### Unified version with both audio input methods:
```bash
streamlit run app_unified.py
```

### Legacy versions:
- `app.py` - System microphone version only
- `app_with_browser_audio.py` - Browser audio version only

The app will open in your default web browser at http://localhost:8501

## Features

- **Dual AI Support**: Choose between Claude CLI and Google Gemini 2.5 Pro
- **Dual Audio Input Methods**: 
  - System Microphone: Direct recording with configurable duration
  - Browser Recording: Uses Streamlit's native audio input widget
- Clean chat interface similar to popular AI chat platforms
- **Speech-to-text input using OpenAI Whisper** (high accuracy)
- Persistent chat history during the session with model attribution
- Timestamps for all messages
- **Verbose logging** to stderr for all AI queries and responses
- Real-time connection status for both AI services
- Clear chat history option
- Configurable Whisper model sizes for accuracy/speed tradeoff

## Speech Recognition

The app uses OpenAI's Whisper model for high-accuracy speech recognition:
- **Tiny**: Fastest, lowest accuracy
- **Base**: Good balance (default)
- **Small**: Better accuracy, slower
- **Medium**: Best accuracy for real-time use

## Architecture

- `app_unified.py` - Unified application with both audio input methods
- `app.py` - Legacy system microphone version
- `app_with_browser_audio.py` - Legacy browser audio version
- `claude_interface.py` - Claude CLI interface with verbose logging
- `gemini_interface.py` - Google Gemini API interface with secure key handling
- `speech_recognition_module.py` - Speech recognition with Whisper
- Modular design for easy extension

## Security Notes

- Google API key is read from RAM disk at `$HOME/Documents/Ephemeral/gapi`
- API key is never echoed or stored elsewhere
- All AI interactions are logged to stderr for transparency