#!/usr/bin/env python3
"""
Pytest configuration and fixtures for the lab automation platform tests
"""

import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data"""
    temp_dir = tempfile.mkdtemp(prefix="weavehacks_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_streamlit_session():
    """Mock Streamlit session state for testing"""
    with patch('streamlit.session_state') as mock_session:
        mock_session.current_experiment_id = "test_experiment"
        mock_session.current_experiment = {
            "experiment_id": "test_experiment",
            "status": "in_progress",
            "mass_gold": 0.1576,
            "mass_toab": 0.25
        }
        
        # Mock platform
        mock_platform = Mock()
        mock_platform.record_data_via_api.return_value = (True, {"status": "success"})
        mock_platform.get_experiment.return_value = (True, {"experiment_id": "test_experiment"})
        mock_session.platform = mock_platform
        
        yield mock_session


@pytest.fixture
def mock_whisper_model():
    """Mock Whisper model for voice recognition tests"""
    mock_model = Mock()
    mock_model.transcribe.return_value = {
        "text": "Gold mass is 0.1576 grams"
    }
    return mock_model


@pytest.fixture
def sample_experiment_data():
    """Sample experiment data for testing"""
    return {
        "experiment_id": "test_001",
        "status": "in_progress",
        "step_num": 3,
        "mass_gold": 0.1576,
        "mass_toab": 0.25,
        "mass_sulfur": 0.1659,
        "mass_nabh4": 0.1514,
        "mass_final": 0.08,
        "volume_toluene": 10.0,
        "volume_nanopure_rt": 5.0,
        "volume_nanopure_cold": 2.0,
        "safety_status": "safe",
        "observations": "Test experiment proceeding normally"
    }


@pytest.fixture
def sample_voice_transcripts():
    """Sample voice transcripts for testing"""
    return [
        "Gold mass is 0.1576 grams",
        "TOAB mass is 0.25 g",
        "Toluene volume is 10 milliliters",
        "The final nanoparticle mass is 0.08 grams",
        "Sulfur amount is 0.2 grams",
        "NaBH4 mass is 0.1514 g"
    ]


@pytest.fixture
def chemistry_test_cases():
    """Test cases for chemistry calculations"""
    return [
        {
            "gold_mass": 0.1576,
            "expected_sulfur": 0.1659,
            "expected_nabh4": 0.1514,
            "actual_yield": 0.08,
            "expected_yield_percent": 44.4
        },
        {
            "gold_mass": 0.3152,  # Double the amount
            "expected_sulfur": 0.3318,
            "expected_nabh4": 0.3028,
            "actual_yield": 0.16,
            "expected_yield_percent": 44.4
        }
    ]


def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may require network/services)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take several seconds)"
    )
    config.addinivalue_line(
        "markers", "requires_backend: marks tests that require the backend server"
    )
    config.addinivalue_line(
        "markers", "requires_audio: marks tests that require audio hardware"
    )
    config.addinivalue_line(
        "markers", "requires_video: marks tests that require video hardware"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names and content"""
    for item in items:
        # Mark integration tests
        if "integration" in item.name.lower() or "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark tests that require backend
        if "api" in item.name.lower() or "backend" in item.name.lower():
            item.add_marker(pytest.mark.requires_backend)
        
        # Mark tests that require audio
        if "voice" in item.name.lower() or "audio" in item.name.lower() or "whisper" in item.name.lower():
            item.add_marker(pytest.mark.requires_audio)
        
        # Mark tests that require video
        if "video" in item.name.lower() or "camera" in item.name.lower() or "opencv" in item.name.lower():
            item.add_marker(pytest.mark.requires_video)


# Utility functions for tests
def create_mock_audio_file(duration=1.0, sample_rate=16000):
    """Create a mock audio file for testing"""
    import numpy as np
    import io
    import wave
    
    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Create WAV file in memory
    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    audio_buffer.seek(0)
    return audio_buffer


def create_mock_image(width=640, height=480):
    """Create a mock image for testing"""
    import numpy as np
    
    # Create random RGB image
    image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return image


@pytest.fixture
def mock_audio_file():
    """Mock audio file for testing"""
    return create_mock_audio_file()


@pytest.fixture
def mock_image():
    """Mock image for testing"""
    return create_mock_image()


# Skip decorators for common conditions
skip_if_no_backend = pytest.mark.skipif(
    "not config.getoption('--run-backend-tests')",
    reason="Backend tests disabled (use --run-backend-tests to enable)"
)

skip_if_no_audio = pytest.mark.skipif(
    "not config.getoption('--run-audio-tests')",
    reason="Audio tests disabled (use --run-audio-tests to enable)"
)

skip_if_no_video = pytest.mark.skipif(
    "not config.getoption('--run-video-tests')",
    reason="Video tests disabled (use --run-video-tests to enable)"
)


def pytest_addoption(parser):
    """Add command line options for pytest"""
    parser.addoption(
        "--run-backend-tests",
        action="store_true",
        default=False,
        help="Run tests that require backend server"
    )
    parser.addoption(
        "--run-audio-tests",
        action="store_true",
        default=False,
        help="Run tests that require audio hardware"
    )
    parser.addoption(
        "--run-video-tests",
        action="store_true",
        default=False,
        help="Run tests that require video hardware"
    )
    parser.addoption(
        "--run-integration-tests",
        action="store_true",
        default=False,
        help="Run integration tests that may require network access"
    )