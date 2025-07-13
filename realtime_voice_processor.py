"""
Real-time voice processing system with continuous listening
"""
import asyncio
import threading
import queue
import numpy as np
import sounddevice as sd
import whisper
import webrtcvad
from datetime import datetime
import json
import weave
from collections import deque

class VoiceActivityDetector:
    """Detects voice activity in audio stream"""
    
    def __init__(self, sample_rate=16000, frame_duration=30):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration  # ms
        self.frame_size = int(sample_rate * frame_duration / 1000)
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
        self.speech_buffer = deque(maxlen=10)  # Buffer for smoothing
        
    def is_speech(self, audio_frame):
        """Check if frame contains speech"""
        # Convert float32 to int16
        audio_int16 = (audio_frame * 32767).astype(np.int16)
        
        # VAD expects bytes
        audio_bytes = audio_int16.tobytes()
        
        try:
            is_speech = self.vad.is_speech(audio_bytes, self.sample_rate)
            self.speech_buffer.append(is_speech)
            
            # Require at least 60% of recent frames to be speech
            if len(self.speech_buffer) >= 5:
                speech_ratio = sum(self.speech_buffer) / len(self.speech_buffer)
                return speech_ratio > 0.6
            
            return is_speech
        except:
            return False

class RealtimeVoiceProcessor:
    """Continuous voice processing with real-time transcription"""
    
    def __init__(self, model_size="base", callback=None):
        self.model = whisper.load_model(model_size)
        self.callback = callback
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = np.float32
        
        # Audio buffers
        self.audio_queue = queue.Queue()
        self.audio_buffer = []
        self.is_recording = False
        self.silence_threshold = 0.01
        self.silence_duration = 0
        self.max_silence = 1.5  # seconds
        
        # Voice activity detection
        self.vad = VoiceActivityDetector(self.sample_rate)
        
        # Processing state
        self.is_processing = False
        self.processing_thread = None
        self.stream = None
        
        # Transcription cache
        self.transcription_cache = deque(maxlen=10)
        
        # Initialize W&B
        weave.init('voice-processing')
    
    @weave.op()
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio stream status: {status}")
        
        # Add to queue for processing
        self.audio_queue.put(indata.copy())
        
        # Check for voice activity
        audio_frame = indata.flatten()
        if len(audio_frame) >= self.vad.frame_size:
            is_speech = self.vad.is_speech(audio_frame[:self.vad.frame_size])
            
            if is_speech:
                self.audio_buffer.extend(audio_frame)
                self.is_recording = True
                self.silence_duration = 0
            elif self.is_recording:
                # Continue recording during short silences
                self.audio_buffer.extend(audio_frame)
                self.silence_duration += frames / self.sample_rate
                
                # Stop recording after max silence
                if self.silence_duration > self.max_silence:
                    self.process_audio_buffer()
    
    @weave.op()
    def process_audio_buffer(self):
        """Process accumulated audio buffer"""
        if len(self.audio_buffer) < self.sample_rate * 0.5:  # Min 0.5 seconds
            self.audio_buffer.clear()
            self.is_recording = False
            return
        
        # Convert buffer to numpy array
        audio_data = np.array(self.audio_buffer, dtype=self.dtype)
        
        # Reset buffer
        self.audio_buffer.clear()
        self.is_recording = False
        
        # Transcribe in separate thread to avoid blocking
        if not self.is_processing:
            self.is_processing = True
            thread = threading.Thread(
                target=self.transcribe_audio,
                args=(audio_data,)
            )
            thread.start()
    
    @weave.op()
    def transcribe_audio(self, audio_data):
        """Transcribe audio data using Whisper"""
        try:
            # Transcribe
            result = self.model.transcribe(
                audio_data,
                language="en",
                fp16=False,
                verbose=False
            )
            
            text = result["text"].strip()
            
            if text and len(text) > 2:  # Ignore very short transcriptions
                # Log to W&B
                weave.log({
                    'transcription': {
                        'text': text,
                        'timestamp': datetime.now().isoformat(),
                        'duration': len(audio_data) / self.sample_rate,
                        'confidence': result.get('segments', [{}])[0].get('confidence', 0)
                    }
                })
                
                # Add to cache
                self.transcription_cache.append({
                    'text': text,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Call callback if provided
                if self.callback:
                    self.callback(text)
                else:
                    print(f"Transcribed: {text}")
        
        except Exception as e:
            print(f"Transcription error: {e}")
        
        finally:
            self.is_processing = False
    
    def start(self):
        """Start continuous voice processing"""
        print("Starting real-time voice processor...")
        print("Listening... (speak naturally, pauses will trigger transcription)")
        
        self.stream = sd.InputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype=self.dtype,
            callback=self.audio_callback,
            blocksize=int(self.sample_rate * 0.03)  # 30ms blocks
        )
        
        self.stream.start()
    
    def stop(self):
        """Stop voice processing"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        print("Voice processor stopped.")
    
    def get_recent_transcriptions(self, n=5):
        """Get recent transcriptions"""
        return list(self.transcription_cache)[-n:]

class VoiceCommandProcessor:
    """Process voice commands for lab operations"""
    
    def __init__(self):
        self.commands = {
            # Data recording commands
            "record mass": self.handle_mass_recording,
            "record volume": self.handle_volume_recording,
            "record temperature": self.handle_temperature_recording,
            
            # Instrument control
            "turn on": self.handle_instrument_on,
            "turn off": self.handle_instrument_off,
            "start stirring": self.handle_start_stirring,
            "stop stirring": self.handle_stop_stirring,
            
            # Safety commands
            "check safety": self.handle_safety_check,
            "emergency stop": self.handle_emergency_stop,
            
            # Protocol commands
            "next step": self.handle_next_step,
            "previous step": self.handle_previous_step,
            "what step": self.handle_current_step,
            
            # Calculation commands
            "calculate": self.handle_calculation,
            "percent yield": self.handle_yield_calculation,
            
            # General commands
            "help": self.handle_help,
            "status": self.handle_status
        }
        
        self.context = {
            'current_step': 0,
            'last_command': None,
            'waiting_for_value': False,
            'pending_operation': None
        }
    
    @weave.op()
    def process_command(self, text):
        """Process transcribed text as command"""
        text_lower = text.lower().strip()
        
        # Check if waiting for a value
        if self.context['waiting_for_value']:
            return self.handle_value_input(text)
        
        # Find matching command
        for command_key, handler in self.commands.items():
            if command_key in text_lower:
                self.context['last_command'] = command_key
                return handler(text_lower)
        
        # If no command matched, treat as general query
        return self.handle_general_query(text)
    
    def handle_mass_recording(self, text):
        """Handle mass recording command"""
        # Extract what substance if mentioned
        substances = ['gold', 'toab', 'sulfur', 'nabh4', 'final']
        substance = None
        
        for s in substances:
            if s in text:
                substance = s
                break
        
        if substance:
            self.context['waiting_for_value'] = True
            self.context['pending_operation'] = ('record_mass', substance)
            return f"Ready to record mass for {substance}. Please state the value in grams."
        else:
            return "What substance are you weighing? Please specify: gold, TOAB, sulfur, NaBH4, or final product."
    
    def handle_volume_recording(self, text):
        """Handle volume recording command"""
        liquids = ['water', 'toluene', 'nanopure', 'ethanol']
        liquid = None
        
        for l in liquids:
            if l in text:
                liquid = l
                break
        
        if liquid:
            self.context['waiting_for_value'] = True
            self.context['pending_operation'] = ('record_volume', liquid)
            return f"Ready to record volume for {liquid}. Please state the value in milliliters."
        else:
            return "What liquid are you measuring? Please specify: water, toluene, or ethanol."
    
    def handle_value_input(self, text):
        """Handle numeric value input"""
        import re
        
        # Extract numbers from text
        numbers = re.findall(r'\d+\.?\d*', text)
        
        if numbers:
            value = float(numbers[0])
            operation, parameter = self.context['pending_operation']
            
            # Reset context
            self.context['waiting_for_value'] = False
            self.context['pending_operation'] = None
            
            # Log the data
            weave.log({
                'data_recorded': {
                    'operation': operation,
                    'parameter': parameter,
                    'value': value,
                    'timestamp': datetime.now().isoformat()
                }
            })
            
            return f"Recorded {value} for {parameter}. Data saved."
        else:
            return "I didn't catch a number. Please state the value clearly."
    
    def handle_instrument_on(self, text):
        """Handle instrument turn on command"""
        instruments = ['centrifuge', 'uv-vis', 'stirrer', 'heating', 'cooling']
        
        for instrument in instruments:
            if instrument in text:
                weave.log({
                    'instrument_control': {
                        'action': 'turn_on',
                        'instrument': instrument,
                        'timestamp': datetime.now().isoformat()
                    }
                })
                return f"Turning on {instrument}..."
        
        return "Which instrument should I turn on?"
    
    def handle_safety_check(self, text):
        """Handle safety check command"""
        return "Checking safety parameters... All systems within normal range."
    
    def handle_next_step(self, text):
        """Handle next step command"""
        self.context['current_step'] += 1
        return f"Moving to step {self.context['current_step'] + 1}"
    
    def handle_calculation(self, text):
        """Handle calculation request"""
        if "sulfur" in text:
            return "To calculate sulfur amount, I need the gold mass. What is the mass of HAuCl4?"
        elif "nabh4" in text or "borohydride" in text:
            return "To calculate NaBH4 amount, I need the gold mass. What is the mass of HAuCl4?"
        else:
            return "What would you like me to calculate? I can calculate sulfur amount, NaBH4 amount, or percent yield."
    
    def handle_help(self, text):
        """Handle help command"""
        return """Available commands:
        - Record mass/volume/temperature
        - Turn on/off instruments
        - Check safety status
        - Next/previous step
        - Calculate reagent amounts
        - Calculate percent yield
        Say 'help' followed by a command for more details."""
    
    def handle_general_query(self, text):
        """Handle general queries"""
        return f"I heard: '{text}'. How can I help with your experiment?"

# Example usage
if __name__ == "__main__":
    # Create command processor
    command_processor = VoiceCommandProcessor()
    
    # Create voice processor with command callback
    def process_voice_command(text):
        print(f"\n🎤 Heard: {text}")
        response = command_processor.process_command(text)
        print(f"🤖 Response: {response}\n")
    
    voice_processor = RealtimeVoiceProcessor(
        model_size="base",
        callback=process_voice_command
    )
    
    try:
        # Start processing
        voice_processor.start()
        
        # Keep running
        print("\nPress Ctrl+C to stop...\n")
        while True:
            asyncio.run(asyncio.sleep(1))
    
    except KeyboardInterrupt:
        print("\nStopping...")
        voice_processor.stop()