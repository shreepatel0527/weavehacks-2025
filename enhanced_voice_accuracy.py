"""
Enhanced voice processing with improved accuracy in noisy lab environments
"""
import numpy as np
import sounddevice as sd
import webrtcvad
import wave
import io
from collections import deque
import threading
import queue
import time
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Callable
import speech_recognition as sr
import noisereduce as nr
from scipy import signal
from scipy.io import wavfile
import audioop
import logging
import weave

class NoiseProfile:
    """Adaptive noise profile for lab environment"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.noise_samples = deque(maxlen=100)
        self.background_noise = None
        self.noise_threshold = None
        self.update_interval = 5.0  # seconds
        self.last_update = time.time()
        
    def add_sample(self, audio_data: np.ndarray):
        """Add noise sample for profile building"""
        self.noise_samples.append(audio_data)
        
        # Update profile periodically
        if time.time() - self.last_update > self.update_interval:
            self.update_profile()
    
    def update_profile(self):
        """Update noise profile from collected samples"""
        if len(self.noise_samples) < 10:
            return
        
        # Concatenate samples
        all_samples = np.concatenate(list(self.noise_samples))
        
        # Calculate noise statistics
        self.background_noise = np.mean(np.abs(all_samples))
        self.noise_threshold = np.percentile(np.abs(all_samples), 95)
        
        self.last_update = time.time()
    
    def is_calibrated(self) -> bool:
        """Check if noise profile is calibrated"""
        return self.background_noise is not None

class AudioPreprocessor:
    """Advanced audio preprocessing for lab environments"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.noise_profile = NoiseProfile(sample_rate)
        
        # Filters
        self.highpass_cutoff = 100  # Hz
        self.lowpass_cutoff = 7000  # Hz
        
        # Create filters
        self._create_filters()
    
    def _create_filters(self):
        """Create audio filters"""
        nyquist = self.sample_rate / 2
        
        # Highpass filter (remove low frequency noise)
        self.highpass_sos = signal.butter(
            4, self.highpass_cutoff / nyquist, 
            'highpass', output='sos'
        )
        
        # Lowpass filter (remove high frequency noise)
        self.lowpass_sos = signal.butter(
            4, self.lowpass_cutoff / nyquist,
            'lowpass', output='sos'
        )
        
        # Notch filter for 60Hz hum
        self.notch_b, self.notch_a = signal.iirnotch(
            60.0, 30.0, self.sample_rate
        )
    
    def preprocess(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply preprocessing to audio"""
        # Convert to float for processing
        audio_float = audio_data.astype(np.float32)
        
        # Apply filters
        filtered = signal.sosfilt(self.highpass_sos, audio_float)
        filtered = signal.sosfilt(self.lowpass_sos, filtered)
        filtered = signal.filtfilt(self.notch_b, self.notch_a, filtered)
        
        # Noise reduction if profile available
        if self.noise_profile.is_calibrated():
            filtered = self._reduce_noise(filtered)
        
        # Normalize
        max_val = np.max(np.abs(filtered))
        if max_val > 0:
            filtered = filtered / max_val * 0.8
        
        # Convert back to int16
        return (filtered * 32767).astype(np.int16)
    
    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise reduction"""
        # Use spectral subtraction
        return nr.reduce_noise(
            y=audio,
            sr=self.sample_rate,
            stationary=True,
            prop_decrease=0.8
        )

class EnhancedVAD:
    """Enhanced Voice Activity Detection for noisy environments"""
    
    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16000):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration = 30  # ms
        
        # Energy-based detection parameters
        self.energy_threshold = None
        self.energy_history = deque(maxlen=50)
        self.zero_crossing_threshold = 0.1
        
        # State tracking
        self.speech_frames = deque(maxlen=10)
        self.min_speech_frames = 3
        
    def is_speech(self, frame: bytes, use_energy: bool = True) -> bool:
        """Detect if frame contains speech"""
        # WebRTC VAD
        vad_result = self.vad.is_speech(frame, self.sample_rate)
        
        if not use_energy:
            return vad_result
        
        # Energy-based detection
        audio_data = np.frombuffer(frame, dtype=np.int16)
        energy = np.sqrt(np.mean(audio_data ** 2))
        
        # Update energy threshold
        self.energy_history.append(energy)
        if self.energy_threshold is None and len(self.energy_history) > 20:
            self.energy_threshold = np.mean(self.energy_history) * 1.5
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
        zcr = zero_crossings / len(audio_data)
        
        # Combined detection
        energy_speech = (
            self.energy_threshold is not None and 
            energy > self.energy_threshold and
            zcr < self.zero_crossing_threshold
        )
        
        # Update speech frame buffer
        self.speech_frames.append(vad_result or energy_speech)
        
        # Require multiple consecutive frames
        speech_count = sum(self.speech_frames)
        return speech_count >= self.min_speech_frames

class MultiEngineRecognizer:
    """Multi-engine speech recognition for improved accuracy"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engines = ['google', 'sphinx']  # Can add more
        
        # Confidence thresholds
        self.min_confidence = 0.6
        self.agreement_threshold = 0.8
        
        # Domain-specific vocabulary
        self.lab_vocabulary = [
            "temperature", "pressure", "nitrogen", "oxygen",
            "start", "stop", "pause", "resume", "experiment",
            "increase", "decrease", "set", "measure", "calibrate",
            "safety", "emergency", "alert", "warning",
            "stirring", "heating", "cooling", "reaction"
        ]
        
        # Initialize offline engine
        self._init_offline_engine()
    
    def _init_offline_engine(self):
        """Initialize offline speech recognition"""
        try:
            # Adjust for ambient noise
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
        except:
            pass
    
    def recognize(self, audio_data: sr.AudioData) -> Dict[str, any]:
        """Recognize speech using multiple engines"""
        results = {}
        
        # Try each engine
        for engine in self.engines:
            try:
                if engine == 'google':
                    text = self.recognizer.recognize_google(
                        audio_data,
                        show_all=True
                    )
                    results['google'] = self._parse_google_result(text)
                    
                elif engine == 'sphinx':
                    text = self.recognizer.recognize_sphinx(
                        audio_data,
                        keyword_entries=self._get_keyword_entries()
                    )
                    results['sphinx'] = {'transcript': text, 'confidence': 0.7}
                    
            except Exception as e:
                results[engine] = {'error': str(e)}
        
        # Combine results
        return self._combine_results(results)
    
    def _parse_google_result(self, result: any) -> Dict:
        """Parse Google recognition result"""
        if isinstance(result, dict) and 'alternative' in result:
            alternatives = result['alternative']
            if alternatives:
                best = alternatives[0]
                return {
                    'transcript': best.get('transcript', ''),
                    'confidence': best.get('confidence', 0.5)
                }
        elif isinstance(result, str):
            return {'transcript': result, 'confidence': 0.8}
        
        return {'transcript': '', 'confidence': 0.0}
    
    def _get_keyword_entries(self) -> List[Tuple[str, float]]:
        """Get keyword entries for Sphinx"""
        return [(word, 1.0) for word in self.lab_vocabulary]
    
    def _combine_results(self, results: Dict) -> Dict:
        """Combine results from multiple engines"""
        valid_results = []
        
        for engine, result in results.items():
            if 'error' not in result and result.get('transcript'):
                valid_results.append(result)
        
        if not valid_results:
            return {'transcript': '', 'confidence': 0.0, 'error': 'No recognition'}
        
        # If only one result, return it
        if len(valid_results) == 1:
            return valid_results[0]
        
        # Find consensus
        transcripts = [r['transcript'].lower() for r in valid_results]
        
        # Simple agreement check
        if len(set(transcripts)) == 1:
            # All agree
            return {
                'transcript': valid_results[0]['transcript'],
                'confidence': min(0.95, max(r['confidence'] for r in valid_results))
            }
        
        # Return highest confidence
        best = max(valid_results, key=lambda r: r.get('confidence', 0))
        return best

class EnhancedVoiceProcessor:
    """Enhanced voice processor with improved accuracy"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 channels: int = 1,
                 use_noise_reduction: bool = True):
        
        self.sample_rate = sample_rate
        self.channels = channels
        self.use_noise_reduction = use_noise_reduction
        
        # Components
        self.preprocessor = AudioPreprocessor(sample_rate)
        self.vad = EnhancedVAD(aggressiveness=2, sample_rate=sample_rate)
        self.recognizer = MultiEngineRecognizer()
        
        # Buffers
        self.audio_buffer = deque(maxlen=int(sample_rate * 10))  # 10 seconds
        self.speech_buffer = []
        self.silence_frames = 0
        self.max_silence_frames = 15  # ~450ms at 30ms frames
        
        # State
        self.is_recording = False
        self.is_speaking = False
        self.calibrating = True
        self.calibration_frames = 0
        self.calibration_needed = 50  # frames
        
        # Callbacks
        self.speech_callbacks = []
        
        # Logging
        self.logger = logging.getLogger('enhanced_voice')
        
        # Initialize W&B
        weave.init('enhanced-voice-processor')
    
    def start(self):
        """Start voice processing"""
        self.is_recording = True
        
        # Start audio stream
        self.stream = sd.InputStream(
            callback=self._audio_callback,
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=int(self.sample_rate * 0.03),  # 30ms blocks
            dtype='int16'
        )
        self.stream.start()
        
        self.logger.info("Voice processing started")
    
    def stop(self):
        """Stop voice processing"""
        self.is_recording = False
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        self.logger.info("Voice processing stopped")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Process audio stream"""
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        
        # Convert to mono if needed
        audio_data = indata[:, 0] if self.channels > 1 else indata.flatten()
        
        # Add to buffer
        self.audio_buffer.extend(audio_data)
        
        # Calibration phase
        if self.calibrating:
            self._calibrate_noise(audio_data)
            return
        
        # Preprocess
        if self.use_noise_reduction:
            audio_data = self.preprocessor.preprocess(audio_data)
        
        # Convert to bytes for VAD
        audio_bytes = audio_data.tobytes()
        
        # Voice activity detection
        is_speech = self.vad.is_speech(audio_bytes)
        
        if is_speech:
            self._handle_speech(audio_data)
        else:
            self._handle_silence()
    
    def _calibrate_noise(self, audio_data: np.ndarray):
        """Calibrate noise profile"""
        self.preprocessor.noise_profile.add_sample(audio_data)
        self.calibration_frames += 1
        
        if self.calibration_frames >= self.calibration_needed:
            self.calibrating = False
            self.preprocessor.noise_profile.update_profile()
            self.logger.info("Noise calibration complete")
    
    def _handle_speech(self, audio_data: np.ndarray):
        """Handle speech frames"""
        if not self.is_speaking:
            self.is_speaking = True
            self.speech_buffer = []
            self.logger.debug("Speech started")
        
        self.speech_buffer.extend(audio_data)
        self.silence_frames = 0
    
    def _handle_silence(self):
        """Handle silence frames"""
        if self.is_speaking:
            self.silence_frames += 1
            
            if self.silence_frames >= self.max_silence_frames:
                # End of speech
                self.is_speaking = False
                self._process_speech()
                self.speech_buffer = []
                self.logger.debug("Speech ended")
    
    def _process_speech(self):
        """Process completed speech"""
        if len(self.speech_buffer) < self.sample_rate * 0.3:  # Min 300ms
            return
        
        # Convert to AudioData
        audio_data = np.array(self.speech_buffer, dtype=np.int16)
        
        # Create WAV in memory
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, self.sample_rate, audio_data)
        wav_buffer.seek(0)
        
        # Convert to AudioData for recognition
        with sr.AudioFile(wav_buffer) as source:
            audio = self.recognizer.recognizer.record(source)
        
        # Recognize
        result = self.recognizer.recognize(audio)
        
        # Log result
        weave.log({
            'voice_recognition': {
                'transcript': result.get('transcript', ''),
                'confidence': result.get('confidence', 0),
                'duration': len(self.speech_buffer) / self.sample_rate
            }
        })
        
        # Call callbacks
        if result.get('transcript'):
            for callback in self.speech_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    self.logger.error(f"Callback error: {e}")
    
    def register_speech_callback(self, callback: Callable):
        """Register callback for recognized speech"""
        self.speech_callbacks.append(callback)
    
    def adjust_for_ambient_noise(self, duration: float = 1.0):
        """Adjust for ambient noise"""
        self.calibrating = True
        self.calibration_frames = 0
        self.calibration_needed = int(duration * self.sample_rate / 480)  # 30ms frames
        
        self.logger.info("Adjusting for ambient noise...")

# Command processor with fuzzy matching
class LabCommandProcessor:
    """Process voice commands with fuzzy matching"""
    
    def __init__(self):
        self.commands = {
            'start_experiment': ['start experiment', 'begin experiment', 'start'],
            'stop_experiment': ['stop experiment', 'end experiment', 'stop'],
            'emergency_stop': ['emergency stop', 'emergency', 'abort'],
            'increase_temperature': ['increase temperature', 'heat up', 'warmer'],
            'decrease_temperature': ['decrease temperature', 'cool down', 'cooler'],
            'status_report': ['status report', 'current status', 'what is the status'],
            'read_temperature': ['what is the temperature', 'temperature reading', 'current temperature'],
            'read_pressure': ['what is the pressure', 'pressure reading', 'current pressure']
        }
        
        # Fuzzy matching threshold
        self.similarity_threshold = 0.7
    
    def process(self, transcript: str) -> Dict[str, any]:
        """Process transcript to command"""
        transcript_lower = transcript.lower().strip()
        
        # Direct matching
        for command, phrases in self.commands.items():
            for phrase in phrases:
                if phrase in transcript_lower:
                    return {
                        'command': command,
                        'confidence': 0.9,
                        'parameters': self._extract_parameters(transcript_lower, command)
                    }
        
        # Fuzzy matching
        best_match = self._fuzzy_match(transcript_lower)
        if best_match['similarity'] >= self.similarity_threshold:
            return {
                'command': best_match['command'],
                'confidence': best_match['similarity'],
                'parameters': self._extract_parameters(transcript_lower, best_match['command'])
            }
        
        return {'command': None, 'confidence': 0.0}
    
    def _fuzzy_match(self, transcript: str) -> Dict:
        """Fuzzy match against known commands"""
        from difflib import SequenceMatcher
        
        best_match = {'command': None, 'similarity': 0.0}
        
        for command, phrases in self.commands.items():
            for phrase in phrases:
                similarity = SequenceMatcher(None, transcript, phrase).ratio()
                if similarity > best_match['similarity']:
                    best_match = {'command': command, 'similarity': similarity}
        
        return best_match
    
    def _extract_parameters(self, transcript: str, command: str) -> Dict:
        """Extract parameters from transcript"""
        parameters = {}
        
        # Extract numbers
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', transcript)
        if numbers:
            parameters['value'] = float(numbers[0])
        
        # Extract units
        if 'degrees' in transcript or 'celsius' in transcript:
            parameters['unit'] = 'celsius'
        elif 'fahrenheit' in transcript:
            parameters['unit'] = 'fahrenheit'
        
        return parameters

# Example usage
def demo_enhanced_voice():
    """Demonstrate enhanced voice processing"""
    processor = EnhancedVoiceProcessor(use_noise_reduction=True)
    command_processor = LabCommandProcessor()
    
    def handle_speech(result: Dict):
        """Handle recognized speech"""
        transcript = result.get('transcript', '')
        confidence = result.get('confidence', 0)
        
        print(f"Recognized: '{transcript}' (confidence: {confidence:.2f})")
        
        # Process command
        command = command_processor.process(transcript)
        if command['command']:
            print(f"Command: {command['command']} (confidence: {command['confidence']:.2f})")
            if command.get('parameters'):
                print(f"Parameters: {command['parameters']}")
    
    processor.register_speech_callback(handle_speech)
    
    try:
        print("Starting enhanced voice processor...")
        print("Calibrating for ambient noise...")
        
        processor.start()
        processor.adjust_for_ambient_noise(duration=2.0)
        
        print("\nListening... (speak clearly)")
        print("Try commands like:")
        print("  - 'Start experiment'")
        print("  - 'What is the temperature?'")
        print("  - 'Emergency stop'")
        
        # Run for demo
        time.sleep(30)
        
    finally:
        processor.stop()
        print("Voice processing stopped")

if __name__ == "__main__":
    demo_enhanced_voice()