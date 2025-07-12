import streamlit as st
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import os
import whisper
from typing import Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


class SpeechRecognizer:
    """High-accuracy speech recognition using OpenAI Whisper."""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize the speech recognizer with Whisper model.
        
        Args:
            model_size: Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large')
                       Larger models are more accurate but slower
        """
        self.model_size = model_size
        self.model = None
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.recording = False
        self.audio_data = []
        
    def load_model(self):
        """Load the Whisper model."""
        if self.model is None:
            with st.spinner(f"Loading Whisper {self.model_size} model..."):
                self.model = whisper.load_model(self.model_size)
    
    def record_audio(self, duration: float = 5.0) -> np.ndarray:
        """
        Record audio from the microphone.
        
        Args:
            duration: Maximum recording duration in seconds
            
        Returns:
            Audio data as numpy array
        """
        st.info("🎤 Recording... Speak now!")
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        return audio_data.flatten()
    
    def transcribe_audio(self, audio_data: np.ndarray) -> Tuple[bool, str]:
        """
        Transcribe audio data to text using Whisper.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Tuple of (success: bool, transcribed_text: str)
        """
        try:
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
            
            # Clean up
            os.unlink(tmp_path)
            
            transcribed_text = result["text"].strip()
            
            if transcribed_text:
                return True, transcribed_text
            else:
                return False, "No speech detected"
                
        except Exception as e:
            return False, f"Transcription error: {str(e)}"
    
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


class StreamlitAudioRecorder:
    """Streamlit-specific audio recording interface using audio_recorder component."""
    
    @staticmethod
    def get_audio_recorder_component():
        """Get the audio recorder component for Streamlit."""
        try:
            from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration
            import av
            
            class AudioProcessor(AudioProcessorBase):
                def __init__(self):
                    self.audio_frames = []
                
                def recv(self, frame):
                    self.audio_frames.append(frame)
                    return frame
            
            return AudioProcessor
            
        except ImportError:
            return None
    
    @staticmethod
    def create_simple_recorder():
        """Create a simple audio recorder interface."""
        import streamlit as st
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("🎤 Record", use_container_width=True, type="primary"):
                return True
        
        with col2:
            st.empty()  # Placeholder for recording status
            
        return False