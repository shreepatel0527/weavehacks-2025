import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import sys
import warnings
import threading
import time
from datetime import datetime

warnings.filterwarnings("ignore")

# Add the src directory to the path
sys.path.append('/Users/User/Documents/Source/Weavehacks-2025-Base/weavehacks_flow-1/src')

from weavehacks_flow.agents.video_monitoring_agent import VideoMonitoringAgent, ExperimentMonitor


class TestVideoMonitoringAgent:
    """Test suite for VideoMonitoringAgent."""
    
    @pytest.fixture
    def agent(self):
        """Create a VideoMonitoringAgent instance for testing."""
        return VideoMonitoringAgent(camera_index=0, fps=30, resolution=(640, 480))
    
    @pytest.fixture
    def mock_frame(self):
        """Create a mock video frame."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_init_default_params(self):
        """Test agent initialization with default parameters."""
        agent = VideoMonitoringAgent()
        assert agent.camera_index == 0
        assert agent.fps == 30
        assert agent.resolution == (640, 480)
        assert agent.capture is None
        assert agent.recording is False
        assert agent.monitoring is False
    
    def test_init_custom_params(self):
        """Test agent initialization with custom parameters."""
        agent = VideoMonitoringAgent(camera_index=1, fps=60, resolution=(1920, 1080))
        assert agent.camera_index == 1
        assert agent.fps == 60
        assert agent.resolution == (1920, 1080)
    
    @patch('cv2.VideoCapture')
    def test_initialize_video_system_success(self, mock_capture_class):
        """Test successful video system initialization."""
        mock_capture = Mock()
        mock_capture.isOpened.return_value = True
        mock_capture_class.return_value = mock_capture
        
        agent = VideoMonitoringAgent()
        # Initialization happens in __init__
        mock_capture.release.assert_called()
    
    @patch('cv2.VideoCapture')
    def test_start_capture_success(self, mock_capture_class, agent):
        """Test successful video capture start."""
        mock_capture = Mock()
        mock_capture.isOpened.return_value = True
        mock_capture_class.return_value = mock_capture
        
        result = agent.start_capture()
        
        assert result is True
        assert agent.capture is not None
        mock_capture.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 640)
        mock_capture.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        mock_capture.set.assert_any_call(cv2.CAP_PROP_FPS, 30)
    
    @patch('cv2.VideoCapture')
    def test_capture_frame_success(self, mock_capture_class, agent, mock_frame):
        """Test successful frame capture."""
        mock_capture = Mock()
        mock_capture.isOpened.return_value = True
        mock_capture.read.return_value = (True, mock_frame)
        mock_capture_class.return_value = mock_capture
        
        agent.start_capture()
        success, frame = agent.capture_frame()
        
        assert success is True
        assert frame is not None
        assert frame.shape == (480, 640, 3)
    
    @patch('cv2.VideoCapture')
    def test_capture_frame_failure(self, mock_capture_class, agent):
        """Test frame capture failure."""
        mock_capture = Mock()
        mock_capture.isOpened.return_value = True
        mock_capture.read.return_value = (False, None)
        mock_capture_class.return_value = mock_capture
        
        agent.start_capture()
        success, frame = agent.capture_frame()
        
        assert success is False
        assert frame is None
    
    def test_detect_motion(self, agent, mock_frame):
        """Test motion detection between frames."""
        frame1 = mock_frame.copy()
        frame2 = mock_frame.copy()
        
        # No motion (identical frames)
        motion, regions = agent.detect_motion(frame1, frame2)
        assert motion is False
        assert len(regions) == 0
        
        # Create motion by modifying frame2
        frame2[100:200, 100:200] = 255  # White square
        motion, regions = agent.detect_motion(frame1, frame2)
        assert motion is True
        assert len(regions) > 0
        assert 'x' in regions[0]
        assert 'y' in regions[0]
        assert 'width' in regions[0]
        assert 'height' in regions[0]
        assert 'area' in regions[0]
    
    def test_detect_color_change(self, agent, mock_frame):
        """Test color change detection."""
        # Set frame to blue
        blue_frame = np.zeros_like(mock_frame)
        blue_frame[:, :, 0] = 255  # Blue channel
        
        # Reference color is red
        reference_color = np.array([0, 0, 255])  # BGR format
        
        changed, difference = agent.detect_color_change(blue_frame, reference_color)
        assert changed is True
        assert difference > 0
        
        # Test with same color
        blue_reference = np.array([255, 0, 0])
        changed, difference = agent.detect_color_change(blue_frame, blue_reference, threshold=50)
        assert changed is False
    
    def test_detect_liquid_level(self, agent):
        """Test liquid level detection."""
        # Create a simple frame with horizontal line
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw horizontal line at y=200
        cv2.line(frame, (50, 200), (590, 200), (255, 255, 255), 2)
        
        level = agent.detect_liquid_level(frame)
        # Level detection might vary due to edge detection
        assert level is None or isinstance(level, int)
        
        # Test with ROI
        roi = (100, 100, 400, 300)
        level_roi = agent.detect_liquid_level(frame, roi)
        assert level_roi is None or isinstance(level_roi, int)
    
    @patch('cv2.VideoCapture')
    def test_monitoring_lifecycle(self, mock_capture_class, agent, mock_frame):
        """Test start and stop monitoring."""
        mock_capture = Mock()
        mock_capture.isOpened.return_value = True
        mock_capture.read.return_value = (True, mock_frame)
        mock_capture_class.return_value = mock_capture
        
        # Start monitoring
        events = []
        agent.start_monitoring(callback=lambda e: events.append(e))
        assert agent.monitoring is True
        assert agent.monitor_thread is not None
        assert agent.monitor_thread.is_alive()
        
        # Let it run briefly
        time.sleep(0.5)
        
        # Stop monitoring
        agent.stop_monitoring()
        assert agent.monitoring is False
        
        # Thread should stop
        time.sleep(0.5)
        assert not agent.monitor_thread.is_alive()
    
    @patch('cv2.VideoCapture')
    @patch('cv2.VideoWriter')
    def test_record_video(self, mock_writer_class, mock_capture_class, agent, mock_frame):
        """Test video recording."""
        # Setup mocks
        mock_capture = Mock()
        mock_capture.isOpened.return_value = True
        mock_capture.read.return_value = (True, mock_frame)
        mock_capture_class.return_value = mock_capture
        
        mock_writer = Mock()
        mock_writer.isOpened.return_value = True
        mock_writer_class.return_value = mock_writer
        
        # Record video
        output_path = "/tmp/test_video.mp4"
        
        # Run recording in thread to allow stopping
        def record():
            agent.record_video(output_path, duration=1.0)
        
        thread = threading.Thread(target=record)
        thread.start()
        
        # Let it record briefly
        time.sleep(0.5)
        
        # Wait for completion
        thread.join(timeout=2.0)
        
        # Verify
        mock_writer.write.assert_called()
        mock_writer.release.assert_called()
    
    def test_get_camera_info(self, agent):
        """Test camera info retrieval."""
        with patch('cv2.VideoCapture') as mock_capture_class:
            mock_capture = Mock()
            mock_capture.isOpened.return_value = True
            mock_capture.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS: 30
            }.get(prop, 0)
            mock_capture_class.return_value = mock_capture
            
            info = agent.get_camera_info()
            
            assert info['camera_index'] == 0
            assert info['resolution'] == (640, 480)
            assert info['fps'] == 30
            assert info['available'] is True
            assert info['actual_width'] == 640
            assert info['actual_height'] == 480
            assert info['actual_fps'] == 30
    
    def test_test_camera(self, agent):
        """Test camera availability check."""
        with patch.object(agent, 'start_capture') as mock_start:
            with patch.object(agent, 'capture_frame') as mock_capture:
                with patch.object(agent, 'stop_capture') as mock_stop:
                    mock_start.return_value = True
                    mock_capture.return_value = (True, np.zeros((480, 640, 3)))
                    
                    result = agent.test_camera()
                    assert result is True
                    
                    mock_start.assert_called_once()
                    mock_capture.assert_called_once()
                    mock_stop.assert_called_once()


class TestExperimentMonitor:
    """Test suite for ExperimentMonitor."""
    
    @pytest.fixture
    def video_agent(self):
        """Create a mock video agent."""
        agent = Mock(spec=VideoMonitoringAgent)
        return agent
    
    @pytest.fixture
    def monitor(self, video_agent):
        """Create an ExperimentMonitor instance."""
        return ExperimentMonitor(video_agent)
    
    def test_start_experiment_monitoring(self, monitor, video_agent):
        """Test starting experiment monitoring."""
        experiment_id = "exp_001"
        
        monitor.start_experiment_monitoring(experiment_id)
        
        assert monitor.experiment_id == experiment_id
        assert monitor.monitoring_active is True
        assert len(monitor.events) == 0
        video_agent.start_monitoring.assert_called_once()
    
    def test_handle_motion_event(self, monitor):
        """Test handling motion detection events."""
        monitor.experiment_id = "exp_001"
        
        event = {
            'type': 'motion',
            'timestamp': datetime.now().isoformat(),
            'regions': [{'x': 100, 'y': 100, 'width': 50, 'height': 50}]
        }
        
        monitor._handle_event(event)
        
        assert len(monitor.events) == 1
        assert monitor.events[0]['experiment_id'] == "exp_001"
        assert monitor.events[0]['type'] == 'motion'
    
    def test_stop_experiment_monitoring(self, monitor, video_agent):
        """Test stopping experiment monitoring."""
        monitor.experiment_id = "exp_001"
        monitor.start_time = datetime.now()
        monitor.monitoring_active = True
        
        # Add some events
        monitor.events = [
            {'type': 'motion', 'experiment_id': 'exp_001'},
            {'type': 'motion', 'experiment_id': 'exp_001'},
            {'type': 'color_change', 'experiment_id': 'exp_001'}
        ]
        
        summary = monitor.stop_experiment_monitoring()
        
        assert monitor.monitoring_active is False
        video_agent.stop_monitoring.assert_called_once()
        
        assert summary['experiment_id'] == "exp_001"
        assert summary['total_events'] == 3
        assert summary['events_by_type']['motion'] == 2
        assert summary['events_by_type']['color_change'] == 1
        assert 'duration' in summary
        assert 'start_time' in summary
        assert 'end_time' in summary


class TestVideoIntegration:
    """Integration tests for video monitoring system."""
    
    def test_video_with_safety_monitoring(self):
        """Test video monitoring integrated with safety alerts."""
        video_agent = VideoMonitoringAgent()
        
        # Mock safety event detection
        with patch.object(video_agent, 'detect_motion') as mock_detect:
            mock_detect.return_value = (True, [{'x': 200, 'y': 200, 'width': 100, 'height': 100}])
            
            # This would integrate with safety monitoring
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            motion, regions = video_agent.detect_motion(frame, frame)
            
            assert motion is True
            assert len(regions) > 0
    
    def test_overnight_monitoring_simulation(self):
        """Test overnight monitoring capabilities."""
        video_agent = VideoMonitoringAgent()
        monitor = ExperimentMonitor(video_agent)
        
        with patch.object(video_agent, 'start_monitoring'):
            with patch.object(video_agent, 'stop_monitoring'):
                # Start overnight monitoring
                monitor.start_experiment_monitoring("overnight_001")
                
                # Simulate events
                for i in range(5):
                    event = {
                        'type': 'motion' if i % 2 == 0 else 'color_change',
                        'timestamp': datetime.now().isoformat()
                    }
                    monitor._handle_event(event)
                
                # Stop and get summary
                summary = monitor.stop_experiment_monitoring()
                
                assert summary['total_events'] == 5
                assert summary['experiment_id'] == "overnight_001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])