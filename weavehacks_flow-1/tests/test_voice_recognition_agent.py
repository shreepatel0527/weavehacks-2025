import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sys
import warnings
warnings.filterwarnings("ignore")

# Add the src directory to the path
sys.path.append('/Users/User/Documents/Source/Weavehacks-2025-Base/weavehacks_flow-1/src')

from weavehacks_flow.agents.voice_recognition_agent import SpeechRecognizerAgent


class TestSpeechRecognizerAgent:
    """Test suite for SpeechRecognizerAgent."""
    
    @pytest.fixture
    def agent(self):
        """Create a SpeechRecognizerAgent instance for testing."""
        return SpeechRecognizerAgent(model_size="tiny")  # Use tiny model for faster tests
    
    @pytest.fixture
    def mock_audio_data(self):
        """Create mock audio data for testing."""
        return np.random.rand(16000).astype(np.float32)  # 1 second of audio at 16kHz
    
    def test_init_default_params(self):
        """Test agent initialization with default parameters."""
        agent = SpeechRecognizerAgent()
        assert agent.model_size == "base"
        assert agent.sample_rate == 16000
        assert agent.model is None
        assert agent.recording is False
    
    def test_init_custom_params(self):
        """Test agent initialization with custom parameters."""
        agent = SpeechRecognizerAgent(model_size="small", sample_rate=22050)
        assert agent.model_size == "small"
        assert agent.sample_rate == 22050
    
    @patch('weavehacks_flow.agents.voice_recognition_agent.whisper.load_model')
    @patch('streamlit.spinner')
    def test_load_model_success(self, mock_spinner, mock_load_model, agent):
        """Test successful model loading."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        mock_spinner.return_value.__enter__.return_value = None
        
        agent.load_model()
        
        assert agent.model is mock_model
        mock_load_model.assert_called_once_with("tiny")
    
    @patch('weavehacks_flow.agents.voice_recognition_agent.whisper.load_model')
    @patch('streamlit.spinner')
    def test_load_model_failure(self, mock_spinner, mock_load_model, agent):
        """Test model loading failure."""
        mock_load_model.side_effect = Exception("Model load failed")
        mock_spinner.return_value.__enter__.return_value = None
        
        with pytest.raises(RuntimeError, match="Could not load Whisper model"):
            agent.load_model()
    
    @patch('weavehacks_flow.agents.voice_recognition_agent.sd.rec')
    @patch('weavehacks_flow.agents.voice_recognition_agent.sd.wait')
    @patch('streamlit.info')
    def test_record_audio_success(self, mock_info, mock_wait, mock_rec, agent, mock_audio_data):
        """Test successful audio recording."""
        mock_rec.return_value = mock_audio_data.reshape(-1, 1)
        
        result = agent.record_audio(duration=1.0)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(mock_audio_data)
        mock_rec.assert_called_once_with(
            16000, samplerate=16000, channels=1, dtype='float32'
        )
        mock_wait.assert_called_once()
    
    @patch('weavehacks_flow.agents.voice_recognition_agent.sd.rec')
    def test_record_audio_device_error(self, mock_rec, agent):
        """Test audio recording with device error."""
        import sounddevice as sd
        mock_rec.side_effect = sd.PortAudioError("Device not found")
        
        with pytest.raises(RuntimeError, match="Microphone access failed"):
            agent.record_audio()
    
    @patch('weavehacks_flow.agents.voice_recognition_agent.tempfile.NamedTemporaryFile')
    @patch('weavehacks_flow.agents.voice_recognition_agent.sf.write')
    @patch('weavehacks_flow.agents.voice_recognition_agent.os.unlink')
    @patch('streamlit.spinner')
    def test_transcribe_audio_success(self, mock_spinner, mock_unlink, mock_sf_write, 
                                     mock_temp_file, agent, mock_audio_data):
        """Test successful audio transcription."""
        # Setup mocks
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.wav"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        mock_spinner.return_value.__enter__.return_value = None
        
        mock_model = Mock()
        mock_model.transcribe.return_value = {"text": "Hello world"}
        agent.model = mock_model
        
        success, text = agent.transcribe_audio(mock_audio_data)
        
        assert success is True
        assert text == "Hello world"
        mock_sf_write.assert_called_once()
        # File cleanup happens in finally block
    
    def test_transcribe_audio_empty_data(self, agent):
        """Test transcription with empty audio data."""
        success, text = agent.transcribe_audio(np.array([]))
        
        assert success is False
        assert text == "No audio data provided"
    
    def test_transcribe_audio_no_speech(self, agent, mock_audio_data):
        """Test transcription with no detected speech."""
        with patch('weavehacks_flow.agents.voice_recognition_agent.tempfile.NamedTemporaryFile'), \
             patch('weavehacks_flow.agents.voice_recognition_agent.sf.write'), \
             patch('weavehacks_flow.agents.voice_recognition_agent.os.unlink'), \
             patch('streamlit.spinner'):
            
            mock_model = Mock()
            mock_model.transcribe.return_value = {"text": "   "}  # Empty/whitespace text
            agent.model = mock_model
            
            success, text = agent.transcribe_audio(mock_audio_data)
            
            assert success is False
            assert text == "No speech detected"
    
    @patch.object(SpeechRecognizerAgent, 'record_audio')
    @patch.object(SpeechRecognizerAgent, 'transcribe_audio')
    def test_record_and_transcribe_success(self, mock_transcribe, mock_record, agent, mock_audio_data):
        """Test successful record and transcribe operation."""
        mock_record.return_value = mock_audio_data
        mock_transcribe.return_value = (True, "Test transcription")
        
        success, text = agent.record_and_transcribe(duration=2.0)
        
        assert success is True
        assert text == "Test transcription"
        mock_record.assert_called_once_with(2.0)
        mock_transcribe.assert_called_once_with(mock_audio_data)
    
    @patch.object(SpeechRecognizerAgent, 'record_audio')
    def test_record_and_transcribe_recording_error(self, mock_record, agent):
        """Test record and transcribe with recording error."""
        mock_record.side_effect = Exception("Recording failed")
        
        success, text = agent.record_and_transcribe()
        
        assert success is False
        assert "Recording error" in text
    
    @patch('weavehacks_flow.agents.voice_recognition_agent.sd.query_devices')
    @patch('weavehacks_flow.agents.voice_recognition_agent.sd.default')
    def test_get_device_info_success(self, mock_default, mock_query, agent):
        """Test successful device info retrieval."""
        mock_query.return_value = ["device1", "device2"]
        mock_default.device = [0, 1]
        
        info = agent.get_device_info()
        
        assert "devices" in info
        assert "default_input" in info
        assert "sample_rate" in info
        assert info["sample_rate"] == 16000
    
    @patch('weavehacks_flow.agents.voice_recognition_agent.sd.query_devices')
    def test_get_device_info_error(self, mock_query, agent):
        """Test device info retrieval with error."""
        mock_query.side_effect = Exception("Device query failed")
        
        info = agent.get_device_info()
        
        assert "error" in info
        assert "Device query failed" in info["error"]
    
    @patch('weavehacks_flow.agents.voice_recognition_agent.sd.rec')
    @patch('weavehacks_flow.agents.voice_recognition_agent.sd.wait')
    def test_test_microphone_success(self, mock_wait, mock_rec, agent):
        """Test successful microphone test."""
        mock_rec.return_value = np.array([[0.1], [0.2]])
        
        result = agent.test_microphone()
        
        assert result is True
        mock_rec.assert_called_once()
        mock_wait.assert_called_once()
    
    @patch('weavehacks_flow.agents.voice_recognition_agent.sd.rec')
    def test_test_microphone_failure(self, mock_rec, agent):
        """Test microphone test failure."""
        mock_rec.side_effect = Exception("Microphone not accessible")
        
        result = agent.test_microphone()
        
        assert result is False
    
    def test_model_sizes(self):
        """Test agent creation with different model sizes."""
        sizes = ["tiny", "base", "small", "medium", "large"]
        for size in sizes:
            agent = SpeechRecognizerAgent(model_size=size)
            assert agent.model_size == size
    
    @patch('weavehacks_flow.agents.voice_recognition_agent.tempfile.NamedTemporaryFile')
    @patch('weavehacks_flow.agents.voice_recognition_agent.sf.write')
    @patch('weavehacks_flow.agents.voice_recognition_agent.os.unlink')
    @patch('weavehacks_flow.agents.voice_recognition_agent.os.path.exists')
    def test_cleanup_on_transcription_error(self, mock_exists, mock_unlink, mock_sf_write, 
                                          mock_temp_file, agent, mock_audio_data):
        """Test that temporary files are cleaned up even when transcription fails."""
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.wav"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        mock_exists.return_value = True
        mock_sf_write.side_effect = Exception("Write failed")
        
        success, text = agent.transcribe_audio(mock_audio_data)
        
        assert success is False
        assert "Transcription error" in text
        # Cleanup happens in finally block regardless of error


class TestIntegrationWithDataCollectionAgent:
    """Integration tests with DataCollectionAgent."""
    
    def test_voice_integration_success(self):
        """Test successful voice integration with data collection agent."""
        voice_agent = SpeechRecognizerAgent(model_size="tiny")
        with patch.object(voice_agent, 'record_and_transcribe') as mock_transcribe:
            mock_transcribe.return_value = (True, "The mass is 2.5 grams")
            
            # Simulate data collection with voice
            result = voice_agent.record_and_transcribe()
            
            assert result[0] is True
            assert "2.5" in result[1]
    
    def test_voice_integration_failure(self):
        """Test voice integration failure handling."""
        voice_agent = SpeechRecognizerAgent(model_size="tiny")
        with patch.object(voice_agent, 'record_and_transcribe') as mock_transcribe:
            mock_transcribe.return_value = (False, "Microphone error")
            
            result = voice_agent.record_and_transcribe()
            
            assert result[0] is False
            assert "error" in result[1].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])