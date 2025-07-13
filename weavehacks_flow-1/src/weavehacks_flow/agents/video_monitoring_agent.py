"""
Video Monitoring Agent for Lab Experiments
Handles video capture, analysis, and monitoring for overnight experiments
"""
import cv2
import numpy as np
import streamlit as st
from typing import Optional, Tuple, Dict, Any, List
import warnings
import logging
import weave
import wandb
import tempfile
import os
from datetime import datetime
import threading
import queue
import time

warnings.filterwarnings("ignore")


class VideoMonitoringAgent:
    """Agent for monitoring experiments via video feed."""
    
    def __init__(self, camera_index: int = 0, fps: int = 30, resolution: Tuple[int, int] = (640, 480)):
        """
        Initialize the video monitoring agent.
        
        Args:
            camera_index: Index of the camera device (0 for default)
            fps: Frames per second for capture
            resolution: Video resolution (width, height)
        """
        self.camera_index = camera_index
        self.fps = fps
        self.resolution = resolution
        self.capture = None
        self.recording = False
        self.monitoring = False
        self.logger = logging.getLogger(__name__)
        
        # Thread management
        self.monitor_thread = None
        self.frame_queue = queue.Queue(maxsize=100)
        
        # Detection parameters
        self.motion_threshold = 25
        self.min_contour_area = 500
        
        # Initialize video system
        self._initialize_video_system()
    
    @weave.op()
    def _initialize_video_system(self):
        """Initialize and validate video system."""
        try:
            # Test camera access
            test_cap = cv2.VideoCapture(self.camera_index)
            if test_cap.isOpened():
                test_cap.release()
                self.logger.info(f"Camera {self.camera_index} initialized successfully")
                return True
            else:
                self.logger.warning(f"Camera {self.camera_index} not available")
                return False
        except Exception as e:
            self.logger.error(f"Video system initialization failed: {e}")
            return False
    
    @weave.op()
    def start_capture(self) -> bool:
        """Start video capture."""
        try:
            if self.capture is None or not self.capture.isOpened():
                self.capture = cv2.VideoCapture(self.camera_index)
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.capture.set(cv2.CAP_PROP_FPS, self.fps)
            
            if self.capture.isOpened():
                self.logger.info("Video capture started")
                return True
            else:
                self.logger.error("Failed to open video capture")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting capture: {e}")
            return False
    
    @weave.op()
    def stop_capture(self):
        """Stop video capture."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            self.logger.info("Video capture stopped")
    
    @weave.op()
    def capture_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture a single frame."""
        if self.capture is None or not self.capture.isOpened():
            return False, None
        
        try:
            ret, frame = self.capture.read()
            if ret:
                return True, frame
            else:
                self.logger.warning("Failed to capture frame")
                return False, None
        except Exception as e:
            self.logger.error(f"Error capturing frame: {e}")
            return False, None
    
    @weave.op()
    def detect_motion(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Detect motion between two frames.
        
        Returns:
            Tuple of (motion_detected, list of motion regions)
        """
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Calculate difference
            diff = cv2.absdiff(gray1, gray2)
            
            # Threshold
            _, thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_contour_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    motion_regions.append({
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h),
                        'area': float(area)
                    })
            
            return len(motion_regions) > 0, motion_regions
            
        except Exception as e:
            self.logger.error(f"Motion detection error: {e}")
            return False, []
    
    @weave.op()
    def detect_color_change(self, frame: np.ndarray, reference_color: np.ndarray, 
                           threshold: float = 30.0) -> Tuple[bool, float]:
        """
        Detect color changes in the frame.
        
        Args:
            frame: Current frame
            reference_color: Reference color in BGR format
            threshold: Color difference threshold
            
        Returns:
            Tuple of (color_changed, average_difference)
        """
        try:
            # Calculate mean color of the frame
            mean_color = cv2.mean(frame)[:3]
            
            # Calculate color difference
            color_diff = np.sqrt(sum((mean_color[i] - reference_color[i])**2 for i in range(3)))
            
            return color_diff > threshold, float(color_diff)
            
        except Exception as e:
            self.logger.error(f"Color detection error: {e}")
            return False, 0.0
    
    @weave.op()
    def detect_liquid_level(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> Optional[int]:
        """
        Detect liquid level in a container.
        
        Args:
            frame: Current frame
            roi: Region of interest (x, y, width, height)
            
        Returns:
            Liquid level in pixels from top, or None if not detected
        """
        try:
            # Apply ROI if specified
            if roi:
                x, y, w, h = roi
                frame = frame[y:y+h, x:x+w]
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find horizontal lines (potential liquid surface)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
            
            if lines is not None:
                # Find the topmost horizontal line
                horizontal_lines = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(y2 - y1) < 5:  # Nearly horizontal
                        horizontal_lines.append(y1)
                
                if horizontal_lines:
                    return min(horizontal_lines)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Liquid level detection error: {e}")
            return None
    
    @weave.op()
    def start_monitoring(self, callback=None):
        """
        Start continuous monitoring in a separate thread.
        
        Args:
            callback: Function to call with monitoring events
        """
        if self.monitoring:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(callback,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("Started video monitoring")
    
    def _monitoring_loop(self, callback):
        """Main monitoring loop running in separate thread."""
        if not self.start_capture():
            self.logger.error("Failed to start capture for monitoring")
            return
        
        prev_frame = None
        
        while self.monitoring:
            success, frame = self.capture_frame()
            if not success:
                time.sleep(0.1)
                continue
            
            # Put frame in queue for UI display
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            
            # Perform analysis
            if prev_frame is not None:
                # Motion detection
                motion_detected, motion_regions = self.detect_motion(prev_frame, frame)
                
                if motion_detected and callback:
                    callback({
                        'type': 'motion',
                        'timestamp': datetime.now().isoformat(),
                        'regions': motion_regions
                    })
            
            prev_frame = frame
            time.sleep(1.0 / self.fps)
        
        self.stop_capture()
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("Stopped video monitoring")
    
    @weave.op()
    def record_video(self, output_path: str, duration: float = 60.0) -> bool:
        """
        Record video to file.
        
        Args:
            output_path: Path to save video file
            duration: Recording duration in seconds
            
        Returns:
            Success status
        """
        try:
            if not self.start_capture():
                return False
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, self.fps, self.resolution)
            
            if not writer.isOpened():
                self.logger.error("Failed to open video writer")
                return False
            
            start_time = time.time()
            self.recording = True
            
            while self.recording and (time.time() - start_time) < duration:
                success, frame = self.capture_frame()
                if success:
                    writer.write(frame)
                else:
                    self.logger.warning("Frame capture failed during recording")
            
            writer.release()
            self.stop_capture()
            self.recording = False
            
            self.logger.info(f"Video recorded to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Recording error: {e}")
            self.recording = False
            return False
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information."""
        info = {
            'camera_index': self.camera_index,
            'resolution': self.resolution,
            'fps': self.fps,
            'available': False
        }
        
        try:
            cap = cv2.VideoCapture(self.camera_index)
            if cap.isOpened():
                info['available'] = True
                info['actual_width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                info['actual_height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                info['actual_fps'] = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
        except Exception as e:
            info['error'] = str(e)
        
        return info
    
    def test_camera(self) -> bool:
        """Test if camera is accessible."""
        try:
            success = self.start_capture()
            if success:
                ret, _ = self.capture_frame()
                self.stop_capture()
                return ret
            return False
        except Exception:
            return False


class ExperimentMonitor:
    """High-level experiment monitoring using video agent."""
    
    def __init__(self, video_agent: VideoMonitoringAgent):
        self.video_agent = video_agent
        self.events = []
        self.monitoring_active = False
        
    def start_experiment_monitoring(self, experiment_id: str):
        """Start monitoring an experiment."""
        self.experiment_id = experiment_id
        self.start_time = datetime.now()
        self.events = []
        
        # Start video monitoring with callback
        self.video_agent.start_monitoring(callback=self._handle_event)
        self.monitoring_active = True
        
        logging.info(f"Started monitoring experiment {experiment_id}")
    
    def _handle_event(self, event: Dict[str, Any]):
        """Handle monitoring events."""
        event['experiment_id'] = self.experiment_id
        self.events.append(event)
        
        # Log significant events
        if event['type'] == 'motion':
            logging.info(f"Motion detected at {event['timestamp']}")
        elif event['type'] == 'color_change':
            logging.info(f"Color change detected: {event['difference']}")
    
    def stop_experiment_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return summary."""
        self.video_agent.stop_monitoring()
        self.monitoring_active = False
        
        duration = (datetime.now() - self.start_time).total_seconds()
        
        summary = {
            'experiment_id': self.experiment_id,
            'duration': duration,
            'total_events': len(self.events),
            'events_by_type': self._summarize_events(),
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat()
        }
        
        return summary
    
    def _summarize_events(self) -> Dict[str, int]:
        """Summarize events by type."""
        summary = {}
        for event in self.events:
            event_type = event['type']
            summary[event_type] = summary.get(event_type, 0) + 1
        return summary