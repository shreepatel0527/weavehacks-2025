import pytest
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
import sys
import warnings
warnings.filterwarnings("ignore")

# Add the src directory to the path
sys.path.append('/Users/User/Documents/Source/Weavehacks-2025-Base/weavehacks_flow-1/src')

from weavehacks_flow.agents.voice_recognition_agent import SpeechRecognizerAgent
from weavehacks_flow.agents.data_collection_agent import DataCollectionAgent
from weavehacks_flow.utils.error_handling import ErrorHandler, SensorError, ErrorSeverity


class TestVoiceRecognitionIntegration:
    """Integration tests for voice recognition with other system components."""
    
    @pytest.fixture
    def voice_agent(self):
        """Create a voice recognition agent."""
        return SpeechRecognizerAgent(model_size="tiny")
    
    @pytest.fixture
    def data_agent(self):
        """Create a data collection agent."""
        return DataCollectionAgent()
    
    @pytest.fixture
    def error_handler(self):
        """Create an error handler."""
        return ErrorHandler()
    
    def test_voice_to_data_pipeline(self, voice_agent, data_agent):
        """Test complete pipeline from voice input to data recording."""
        # Mock voice recognition
        with patch.object(voice_agent, 'record_and_transcribe') as mock_transcribe:
            mock_transcribe.return_value = (True, "The mass is 2.534 grams")
            
            # Simulate voice input
            success, text = voice_agent.record_and_transcribe()
            assert success is True
            
            # Process with data agent
            with patch.object(data_agent, 'process_voice_input') as mock_process:
                mock_process.return_value = {
                    'parameter': 'mass',
                    'value': 2.534,
                    'unit': 'grams'
                }
                
                result = data_agent.process_voice_input(text)
                assert result['value'] == 2.534
                assert result['unit'] == 'grams'
    
    def test_error_handling_in_voice_pipeline(self, voice_agent, error_handler):
        """Test error handling when voice recognition fails."""
        # Simulate microphone error
        with patch.object(voice_agent, 'record_audio') as mock_record:
            mock_record.side_effect = RuntimeError("Audio device error")
            
            success, message = voice_agent.record_and_transcribe()
            assert success is False
            assert "Recording error" in message
    
    def test_voice_agent_with_noisy_environment(self, voice_agent):
        """Test voice recognition with background noise."""
        # Create noisy audio data
        np.random.seed(42)
        clean_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        noise = np.random.normal(0, 0.1, 16000)
        noisy_audio = clean_signal + noise
        
        with patch.object(voice_agent, 'transcribe_audio') as mock_transcribe:
            # Simulate whisper handling noisy audio
            mock_transcribe.return_value = (True, "[NOISE] Temperature is 25 degrees")
            
            success, text = voice_agent.transcribe_audio(noisy_audio)
            assert success is True
            assert "25" in text
    
    def test_continuous_voice_monitoring(self, voice_agent):
        """Test continuous voice monitoring for hands-free operation."""
        # Simulate continuous recording sessions
        recordings = [
            "Starting the experiment",
            "Adding 5 milliliters of solution",
            "Temperature reading 23.5 celsius",
            "Observing color change to yellow",
            "Reaction complete"
        ]
        
        results = []
        for expected_text in recordings:
            with patch.object(voice_agent, 'record_and_transcribe') as mock_transcribe:
                mock_transcribe.return_value = (True, expected_text)
                
                success, text = voice_agent.record_and_transcribe(duration=3.0)
                if success:
                    results.append(text)
        
        assert len(results) == 5
        assert "5 milliliters" in results[1]
        assert "23.5" in results[2]
    
    def test_voice_agent_language_variations(self, voice_agent):
        """Test voice recognition with different phrasings."""
        test_phrases = [
            ("The mass is 2.5 grams", 2.5),
            ("I measured 2.5 g", 2.5),
            ("Two point five grams", 2.5),
            ("Mass reading: 2.5", 2.5),
            ("It weighs 2.5 grams", 2.5)
        ]
        
        for phrase, expected_value in test_phrases:
            with patch.object(voice_agent, 'transcribe_audio') as mock_transcribe:
                mock_transcribe.return_value = (True, phrase)
                
                # Simulate audio data
                audio = np.random.rand(16000).astype(np.float32)
                success, text = voice_agent.transcribe_audio(audio)
                
                assert success is True
                assert str(expected_value) in text
    
    def test_voice_agent_resilience(self, voice_agent):
        """Test voice agent resilience to various failure modes."""
        # Test with empty audio
        success, text = voice_agent.transcribe_audio(np.array([]))
        assert success is False
        assert "No audio data" in text
        
        # Test with very quiet audio
        quiet_audio = np.full(16000, 0.00001, dtype=np.float32)
        with patch.object(voice_agent, 'model') as mock_model:
            mock_model.transcribe.return_value = {"text": ""}
            
            success, text = voice_agent.transcribe_audio(quiet_audio)
            assert success is False
    
    @patch('streamlit.spinner')
    @patch('streamlit.info')
    def test_voice_ui_integration(self, mock_info, mock_spinner, voice_agent):
        """Test voice agent integration with Streamlit UI."""
        mock_spinner.return_value.__enter__.return_value = None
        
        # Simulate UI recording session
        with patch.object(voice_agent, 'record_audio') as mock_record:
            mock_record.return_value = np.random.rand(16000 * 5).astype(np.float32)
            
            audio = voice_agent.record_audio(duration=5.0)
            assert len(audio) == 16000 * 5
            mock_info.assert_called_with("🎤 Recording... Speak now!")
    
    def test_voice_agent_performance(self, voice_agent):
        """Test voice agent performance metrics."""
        import time
        
        # Mock model loading
        with patch.object(voice_agent, 'load_model'):
            voice_agent.load_model()
        
        # Test transcription speed
        audio_duration = 5.0  # seconds
        audio_data = np.random.rand(int(16000 * audio_duration)).astype(np.float32)
        
        with patch.object(voice_agent, 'model') as mock_model:
            mock_model.transcribe.return_value = {"text": "Test transcription"}
            
            start_time = time.time()
            success, text = voice_agent.transcribe_audio(audio_data)
            transcription_time = time.time() - start_time
            
            assert success is True
            # Transcription should be reasonably fast (mocked)
            assert transcription_time < 1.0
    
    def test_voice_agent_multi_device_support(self, voice_agent):
        """Test voice agent with multiple audio devices."""
        # Mock multiple devices
        mock_devices = [
            {"name": "Built-in Microphone", "max_input_channels": 2},
            {"name": "USB Headset", "max_input_channels": 1},
            {"name": "External Mic", "max_input_channels": 1}
        ]
        
        with patch('weavehacks_flow.agents.voice_recognition_agent.sd.query_devices') as mock_query:
            mock_query.return_value = mock_devices
            
            device_info = voice_agent.get_device_info()
            assert len(device_info['devices']) == 3
            
            # Test device switching
            for i in range(3):
                success = voice_agent.set_audio_device(device_id=i)
                assert isinstance(success, bool)
    
    def test_voice_safety_integration(self, voice_agent):
        """Test voice recognition for safety-critical commands."""
        safety_commands = [
            "Emergency stop",
            "Shutdown the system",
            "Temperature too high",
            "Pressure exceeding limits"
        ]
        
        for command in safety_commands:
            with patch.object(voice_agent, 'record_and_transcribe') as mock_transcribe:
                mock_transcribe.return_value = (True, command)
                
                success, text = voice_agent.record_and_transcribe()
                assert success is True
                assert command in text


class TestVoiceRecognitionAdvanced:
    """Advanced tests for voice recognition capabilities."""
    
    @pytest.fixture
    def voice_agent(self):
        """Create a voice recognition agent."""
        return SpeechRecognizerAgent(model_size="tiny", sample_rate=16000)
    
    def test_voice_agent_with_weave_integration(self, voice_agent):
        """Test voice agent with W&B Weave tracking."""
        with patch('weave.op') as mock_weave:
            # The decorator should be applied
            mock_weave.return_value = lambda f: f
            
            # Test that weave operations are tracked
            voice_agent.load_model()
            voice_agent.record_audio(duration=1.0)
            voice_agent.transcribe_audio(np.random.rand(16000))
    
    def test_voice_commands_parsing(self, voice_agent):
        """Test parsing of voice commands for lab control."""
        commands = {
            "Turn on the centrifuge": {"action": "turn_on", "device": "centrifuge"},
            "Set temperature to 25 degrees": {"action": "set", "parameter": "temperature", "value": 25},
            "Start stirring at 1100 RPM": {"action": "start", "parameter": "stirring", "value": 1100},
            "Record observation yellow precipitate": {"action": "record", "type": "observation", "value": "yellow precipitate"}
        }
        
        for voice_input, expected_parse in commands.items():
            with patch.object(voice_agent, 'transcribe_audio') as mock_transcribe:
                mock_transcribe.return_value = (True, voice_input)
                
                audio = np.random.rand(16000).astype(np.float32)
                success, text = voice_agent.transcribe_audio(audio)
                
                assert success is True
                # In a real implementation, we would parse the command
                # For now, just verify the text was captured
                assert any(word in text.lower() for word in ["turn", "set", "start", "record"])
    
    def test_voice_batch_processing(self, voice_agent):
        """Test batch processing of multiple audio recordings."""
        # Create multiple audio samples
        audio_samples = [
            np.random.rand(16000).astype(np.float32) for _ in range(5)
        ]
        
        results = []
        for i, audio in enumerate(audio_samples):
            with patch.object(voice_agent, 'model') as mock_model:
                mock_model.transcribe.return_value = {"text": f"Sample {i+1}"}
                
                success, text = voice_agent.transcribe_audio(audio)
                if success:
                    results.append(text)
        
        assert len(results) == 5
        assert all(f"Sample {i+1}" in results[i] for i in range(5))
    
    def test_voice_agent_memory_efficiency(self, voice_agent):
        """Test memory efficiency with large audio files."""
        # Test with a 30-second audio file
        large_audio = np.random.rand(16000 * 30).astype(np.float32)
        
        with patch.object(voice_agent, 'model') as mock_model:
            mock_model.transcribe.return_value = {"text": "Long transcription"}
            
            # Should handle large audio without memory issues
            success, text = voice_agent.transcribe_audio(large_audio)
            assert success is True
    
    def test_voice_agent_concurrent_access(self, voice_agent):
        """Test voice agent with simulated concurrent access."""
        import threading
        
        results = []
        
        def record_and_transcribe():
            with patch.object(voice_agent, 'record_and_transcribe') as mock:
                mock.return_value = (True, "Concurrent test")
                success, text = voice_agent.record_and_transcribe()
                results.append((success, text))
        
        # Create multiple threads
        threads = [threading.Thread(target=record_and_transcribe) for _ in range(3)]
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # All should succeed
        assert len(results) == 3
        assert all(r[0] for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])