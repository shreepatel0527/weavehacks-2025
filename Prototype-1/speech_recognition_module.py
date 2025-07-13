#!/usr/bin/env python3
"""
Speech Recognition Module
Handles speech-to-text conversion using Whisper.
"""

import os
import tempfile
import logging

try:
    import whisper
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechRecognizer:
    """Speech recognition using Whisper."""
    
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None
        self.available = SPEECH_AVAILABLE
        
        if self.available:
            try:
                self.model = whisper.load_model(model_size)
                logger.info(f"Loaded Whisper model: {model_size}")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                self.available = False
    
    def record_and_transcribe(self, duration: float = 5.0, sample_rate: int = 16000) -> tuple[bool, str]:
        """Record audio and transcribe to text."""
        if not self.available or not self.model:
            return False, "Speech recognition not available"
        
        try:
            logger.info(f"Recording for {duration} seconds...")
            
            # Record audio
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32
            )
            sd.wait()  # Wait for recording to complete
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_data, sample_rate)
                tmp_path = tmp_file.name
            
            try:
                # Transcribe
                result = self.model.transcribe(tmp_path, language="en", fp16=False)
                text = result["text"].strip()
                
                return bool(text), text if text else "No speech detected"
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
                
        except Exception as e:
            return False, f"Speech recognition error: {str(e)}"
    
    def test_audio_devices(self) -> str:
        """Test and list available audio devices."""
        if not SPEECH_AVAILABLE:
            return "Speech recognition dependencies not available"
        
        try:
            devices = sd.query_devices()
            return str(devices)
        except Exception as e:
            return f"Error querying audio devices: {e}"