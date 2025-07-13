import streamlit as st
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import os
import whisper
from typing import Optional, Tuple
import warnings
import logging
warnings.filterwarnings("ignore")
import weave
import wandb

class SpeechRecognizerAgent:
    """High-accuracy speech recognition using OpenAI Whisper."""
    
    def __init__(self, model_size: str = "base", sample_rate: int = 16000):
        """
        Initialize the speech recognizer with Whisper model.
        
        Args:
            model_size: Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large')
                       Larger models are more accurate but slower
            sample_rate: Audio sample rate (default 16kHz for Whisper)
        """
        self.model_size = model_size
        self.model = None
        self.sample_rate = sample_rate
        self.recording = False
        self.logger = logging.getLogger(__name__)
    @weave.op()  
    def load_model(self):
        """Load the Whisper model."""
        if self.model is None:
            try:
                with st.spinner(f"Loading Whisper {self.model_size} model..."):
                    self.model = whisper.load_model(self.model_size)
                self.logger.info(f"Whisper {self.model_size} model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load Whisper model: {e}")
                raise RuntimeError(f"Could not load Whisper model: {e}")
    
    @weave.op()
    def record_audio(self, duration: float = 5.0) -> np.ndarray:
        """
        Record audio from the microphone.
        
        Args:
            duration: Maximum recording duration in seconds
            
        Returns:
            Audio data as numpy array
        """
        try:
            st.info("🎤 Recording... Speak now!")
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()  # Wait until recording is finished
            self.logger.info(f"Audio recorded successfully: {duration}s")
            return audio_data.flatten()
        except sd.PortAudioError as e:
            self.logger.error(f"Audio device error: {e}")
            raise RuntimeError(f"Microphone access failed: {e}")
        except Exception as e:
            self.logger.error(f"Recording failed: {e}")
            raise RuntimeError(f"Audio recording failed: {e}")
    
    @weave.op() 
    def transcribe_audio(self, audio_data: np.ndarray) -> Tuple[bool, str]:
        """
        Transcribe audio data to text using Whisper.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Tuple of (success: bool, transcribed_text: str)
        """
        tmp_path = None
        try:
            # Validate audio data
            if audio_data is None or len(audio_data) == 0:
                return False, "No audio data provided"
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_data, self.sample_rate)
                tmp_path = tmp_file.name
            
            # Load model if not already loaded
            self.load_model()
            
            # Transcribe
            with st.spinner("Transcribing..."):
                result = self.model.transcribe(
                    tmp_path,
                    language="en",  # Can be made configurable
                    fp16=False
                )
            
            transcribed_text = result["text"].strip()
            self.logger.info(f"Transcription successful: '{transcribed_text[:50]}...'")
            
            if transcribed_text:
                return True, transcribed_text
            else:
                return False, "No speech detected"
                
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return False, f"Transcription error: {str(e)}"
        finally:
            # Always clean up temporary file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to cleanup temp file {tmp_path}: {cleanup_error}")
    
    @weave.op()
    def record_and_transcribe(self, duration: float = 5.0) -> Tuple[bool, str]:
        """
        Record audio and transcribe it in one operation.
        
        Args:
            duration: Maximum recording duration in seconds
            
        Returns:
            Tuple of (success: bool, transcribed_text: str)
        """
        try:
            audio_data = self.record_audio(duration)
            return self.transcribe_audio(audio_data)
        except Exception as e:
            return False, f"Recording error: {str(e)}"

    def get_device_info(self) -> dict:
        """Get audio device information for debugging."""
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            return {
                "devices": devices,
                "default_input": default_input,
                "sample_rate": self.sample_rate
            }
        except Exception as e:
            self.logger.error(f"Failed to get device info: {e}")
            return {"error": str(e)}
    
    def test_microphone(self) -> bool:
        """Test if microphone is accessible."""
        try:
            # Record 0.1 seconds of audio to test
            test_duration = 0.1
            sd.rec(int(test_duration * self.sample_rate), 
                  samplerate=self.sample_rate, channels=1)
            sd.wait()
            return True
        except Exception as e:
            self.logger.error(f"Microphone test failed: {e}")
            return False