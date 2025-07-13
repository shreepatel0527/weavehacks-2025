import streamlit as st
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import os
import whisper
from typing import Optional, Tuple, Dict, Any
import warnings
import logging
warnings.filterwarnings("ignore")
try:
    import weave
except ImportError:
    weave = None

try:
    import wandb
except ImportError:
    wandb = None
try:
    from ..utils.audio_diagnostics import AudioDiagnostics
except ImportError:
    # Fallback for when utils module is not available
    AudioDiagnostics = None

class SpeechRecognizerAgent:
    """High-accuracy speech recognition using OpenAI Whisper."""
    
    def __init__(self, model_size: str = "base", sample_rate: int = 16000, device_id: Optional[int] = None):
        """
        Initialize the speech recognizer with Whisper model.
        
        Args:
            model_size: Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large')
                       Larger models are more accurate but slower
            sample_rate: Audio sample rate (default 16kHz for Whisper)
            device_id: Specific audio input device ID to use (None for default)
        """
        self.model_size = model_size
        self.model = None
        self.sample_rate = sample_rate
        self.device_id = device_id
        self.recording = False
        self.audio_initialized = False
        self.logger = logging.getLogger(__name__)
        
        # Initialize audio diagnostics
        self._initialize_audio_system()
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
        if not self.audio_initialized:
            raise RuntimeError("Audio system not properly initialized. Run audio diagnostics for troubleshooting.")
        
        try:
            st.info("🎤 Recording... Speak now!")
            
            # Use specific device if set, otherwise default
            device = self.device_id if self.device_id is not None else None
            
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                device=device
            )
            sd.wait()  # Wait until recording is finished
            self.logger.info(f"Audio recorded successfully: {duration}s on device {device}")
            return audio_data.flatten()
            
        except sd.PortAudioError as e:
            self.logger.error(f"Audio device error: {e}")
            error_msg = self._get_helpful_audio_error_message(e)
            raise RuntimeError(error_msg)
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

    def _initialize_audio_system(self):
        """Initialize and validate audio system."""
        try:
            # Basic device query test
            devices = sd.query_devices()
            default_input = sd.default.device[0] if sd.default.device else None
            
            if default_input is None or default_input == -1:
                self.logger.warning("No default input device found")
                # Try to find and suggest an input device
                if AudioDiagnostics:
                    suggested_device = AudioDiagnostics.suggest_device_selection(devices)
                    if suggested_device is not None:
                        self.device_id = suggested_device
                        self.logger.info(f"Using suggested device: {suggested_device}")
            
            # Test basic functionality
            test_success = self._test_audio_access()
            self.audio_initialized = test_success
            
            if not test_success:
                self.logger.warning("Audio system test failed")
                
        except Exception as e:
            self.logger.error(f"Audio system initialization failed: {e}")
            self.audio_initialized = False
    
    def _test_audio_access(self) -> bool:
        """Test basic audio access without recording."""
        try:
            # Quick device access test
            device = self.device_id if self.device_id is not None else None
            
            # Very short test recording
            test_duration = 0.1
            audio_data = sd.rec(
                int(test_duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                device=device
            )
            sd.wait()
            return True
        except Exception as e:
            self.logger.error(f"Audio access test failed: {e}")
            return False
    
    def _get_helpful_audio_error_message(self, error: Exception) -> str:
        """Generate helpful error message based on the specific audio error."""
        error_str = str(error).lower()
        
        if "device -1" in error_str or "invalid device" in error_str:
            return (
                "❌ Audio Device Error: No valid microphone found.\n"
                "🔧 Quick fixes to try:\n"
                "1. Check if microphone is connected and enabled\n"
                "2. Grant microphone permissions to this application\n"
                "3. Restart audio services on your system\n"
                "4. Try running: python -c 'from weavehacks_flow.utils.audio_diagnostics import AudioDiagnostics; print(AudioDiagnostics.run_full_diagnosis())'\n"
                f"Original error: {error}"
            )
        elif "permission" in error_str or "access" in error_str:
            return (
                "❌ Permission Error: Cannot access microphone.\n"
                "🔧 Grant microphone permissions in system settings.\n"
                f"Original error: {error}"
            )
        else:
            return f"❌ Audio Error: {error}\n🔧 Run audio diagnostics for troubleshooting help."
    
    def get_device_info(self) -> dict:
        """Get comprehensive audio device information for debugging."""
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0] if sd.default.device else None
            
            info = {
                "devices": devices,
                "default_input": default_input,
                "current_device": self.device_id,
                "sample_rate": self.sample_rate,
                "audio_initialized": self.audio_initialized
            }
            
            # Add suggested device if available
            if AudioDiagnostics and devices:
                suggested = AudioDiagnostics.suggest_device_selection(devices)
                info["suggested_device"] = suggested
            
            return info
        except Exception as e:
            self.logger.error(f"Failed to get device info: {e}")
            return {"error": str(e)}
    
    def test_microphone(self) -> bool:
        """Test if microphone is accessible."""
        return self._test_audio_access()
    
    def run_audio_diagnostics(self) -> str:
        """Run comprehensive audio diagnostics and return report."""
        if AudioDiagnostics:
            return AudioDiagnostics.run_full_diagnosis()
        else:
            return "Audio diagnostics not available. Please check audio_diagnostics module."
    
    def set_audio_device(self, device_id: Optional[int] = None) -> bool:
        """Set specific audio device and test it."""
        self.device_id = device_id
        success = self._test_audio_access()
        self.audio_initialized = success
        
        if success:
            self.logger.info(f"Successfully set audio device to: {device_id}")
        else:
            self.logger.warning(f"Failed to set audio device to: {device_id}")
        
        return success