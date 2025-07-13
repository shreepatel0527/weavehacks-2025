"""
Video Monitoring Agent for Lab Experiments
Integrates video monitoring capabilities with the existing lab assistant system
"""
import numpy as np
import threading
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import weave
import wandb
from collections import deque
import logging

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    import warnings
    warnings.warn("OpenCV (cv2) not available. Video monitoring features will be disabled.")

def safe_wandb_log(data: dict):
    """Safely log to wandb, handling cases where wandb is not initialized"""
    try:
        wandb.log(data)
    except wandb.errors.UsageError:
        try:
            wandb.init(project="lab-assistant-agents", mode="disabled")
            wandb.log(data)
        except Exception:
            pass
    except Exception:
        pass

class EventType(Enum):
    COLOR_CHANGE = "color_change"
    MOTION_DETECTED = "motion_detected"
    OBJECT_DETECTED = "object_detected"
    ANOMALY = "anomaly"
    EXPERIMENT_PHASE = "experiment_phase"
    SAFETY_VIOLATION = "safety_violation"
    LIQUID_LEVEL = "liquid_level"
    BUBBLE_FORMATION = "bubble_formation"
    CRYSTALLIZATION = "crystallization"

@dataclass
class VideoEvent:
    timestamp: datetime
    event_type: EventType
    description: str
    confidence: float
    frame_number: int
    region_of_interest: Optional[Tuple[int, int, int, int]] = None
    image_data: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)

class ColorChangeAnalyzer:
    """Analyze color changes in video frames for chemical reactions"""
    
    def __init__(self, sensitivity: float = 15.0):
        self.sensitivity = sensitivity
        self.color_history = deque(maxlen=30)
        self.reference_colors = {}
        self.cv2_available = CV2_AVAILABLE
        
    def analyze(self, frame: np.ndarray, roi: Optional[Tuple] = None) -> Dict[str, Any]:
        """Analyze color in frame or ROI"""
        if not self.cv2_available:
            return {
                'current_color': None,
                'change_detected': False,
                'change_magnitude': 0,
                'color_name': 'unknown'
            }
            
        # Extract ROI if specified, otherwise use center region
        if roi:
            x, y, w, h = roi
            analysis_region = frame[y:y+h, x:x+w]
        else:
            h, w = frame.shape[:2]
            analysis_region = frame[h//4:3*h//4, w//4:3*w//4]
        
        # Convert to HSV and LAB color spaces
        hsv = cv2.cvtColor(analysis_region, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(analysis_region, cv2.COLOR_BGR2LAB)
        
        # Calculate color statistics
        hsv_mean = cv2.mean(hsv)[:3]
        lab_mean = cv2.mean(lab)[:3]
        
        # Store in history
        color_data = {
            'hsv': hsv_mean,
            'lab': lab_mean,
            'timestamp': datetime.now()
        }
        self.color_history.append(color_data)
        
        # Detect color changes
        change_detected = False
        change_magnitude = 0
        
        if len(self.color_history) >= 2:
            prev_color = self.color_history[-2]
            
            # Calculate color difference in Lab space (more perceptually uniform)
            lab_diff = np.sqrt(sum((a - b) ** 2 for a, b in 
                                 zip(lab_mean, prev_color['lab'])))
            
            if lab_diff > self.sensitivity:
                change_detected = True
                change_magnitude = lab_diff
        
        return {
            'current_color': color_data,
            'change_detected': change_detected,
            'change_magnitude': change_magnitude,
            'color_name': self._get_color_name(hsv_mean)
        }
    
    def _get_color_name(self, hsv: Tuple) -> str:
        """Get approximate color name from HSV values"""
        h, s, v = hsv
        
        # Simple color classification based on hue
        if s < 20:  # Low saturation
            if v > 200:
                return "white"
            elif v < 50:
                return "black"
            else:
                return "gray"
        elif h < 10 or h > 170:
            return "red"
        elif h < 25:
            return "orange"
        elif h < 35:
            return "yellow"
        elif h < 85:
            return "green"
        elif h < 130:
            return "blue"
        else:
            return "purple"

class MotionDetector:
    """Detect motion in video frames for stirring and mixing detection"""
    
    def __init__(self):
        self.cv2_available = CV2_AVAILABLE
        if self.cv2_available:
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.min_area = 500
        
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect motion in frame"""
        if not self.cv2_available:
            return {
                'motion_detected': False,
                'motion_regions': [],
                'total_motion_area': 0,
                'motion_mask': None
            }
            
        # Background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Remove shadows
        fg_mask[fg_mask == 127] = 0
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Analyze contours
        motion_regions = []
        total_motion_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                motion_regions.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'center': (x + w // 2, y + h // 2)
                })
                total_motion_area += area
        
        return {
            'motion_detected': len(motion_regions) > 0,
            'motion_regions': motion_regions,
            'total_motion_area': total_motion_area,
            'motion_mask': fg_mask
        }

class VideoMonitoringAgent:
    """Video monitoring agent compatible with existing lab assistant architecture"""
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.is_monitoring = False
        self.is_recording = False
        self.cv2_available = CV2_AVAILABLE
        
        # Analysis components
        self.color_analyzer = ColorChangeAnalyzer()
        self.motion_detector = MotionDetector()
        
        # Video capture
        self.capture = None
        self.frame_queue = queue.Queue(maxsize=30)
        self.event_queue = queue.Queue(maxsize=100)
        
        # Recording
        self.video_writer = None
        self.recording_path = Path("recordings")
        self.recording_path.mkdir(exist_ok=True)
        
        # Threading
        self._stop_event = threading.Event()
        self._threads = []
        
        # Event handling
        self.event_callbacks = []
        self.event_history = deque(maxlen=1000)
        
        # Statistics
        self.frame_number = 0
        self.stats = {
            'frames_processed': 0,
            'events_detected': 0,
            'monitoring_duration': 0
        }
        
        # Logging
        self.logger = logging.getLogger('video_monitoring_agent')
        
        # Safety integration
        self.safety_violations = []
        
        # Initialize video system if available
        if self.cv2_available:
            self._initialize_video_system()
        else:
            self.logger.warning("OpenCV not available - video monitoring disabled")
    
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
    
    def test_camera(self) -> bool:
        """Test if camera is accessible."""
        if not self.cv2_available:
            return False
        try:
            cap = cv2.VideoCapture(self.camera_index)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                return ret
            return False
        except Exception:
            return False
        
    @weave.op()
    def start_monitoring(self, callback=None):
        """Start video monitoring"""
        if not self.cv2_available:
            return {"status": "error", "message": "OpenCV not available"}
            
        if self.is_monitoring:
            return {"status": "already_running"}
        
        try:
            # Open camera
            self.capture = cv2.VideoCapture(self.camera_index)
            if not self.capture.isOpened():
                # Try alternative camera indices
                for idx in [0, 1, 2]:
                    self.capture = cv2.VideoCapture(idx)
                    if self.capture.isOpened():
                        self.camera_index = idx
                        break
                else:
                    return {"status": "error", "message": "Cannot open any camera"}
            
            # Configure camera
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_monitoring = True
            self._stop_event.clear()
            
            # Register callback if provided
            if callback:
                self.register_callback(callback)
            
            # Start threads
            capture_thread = threading.Thread(
                target=self._capture_loop,
                name="VideoCaptureThread"
            )
            capture_thread.daemon = True
            capture_thread.start()
            self._threads.append(capture_thread)
            
            process_thread = threading.Thread(
                target=self._process_loop,
                name="VideoProcessThread"
            )
            process_thread.daemon = True
            process_thread.start()
            self._threads.append(process_thread)
            
            # Log to W&B
            safe_wandb_log({
                'video_monitoring': {
                    'action': 'start',
                    'camera_index': self.camera_index,
                    'timestamp': datetime.now().isoformat()
                }
            })
            
            self.logger.info("Video monitoring started")
            return {"status": "success", "message": "Video monitoring started"}
            
        except Exception as e:
            self.logger.error(f"Failed to start video monitoring: {e}")
            return {"status": "error", "message": str(e)}
    
    @weave.op()
    def stop_monitoring(self):
        """Stop video monitoring"""
        if not self.is_monitoring:
            return {"status": "not_running"}
        
        self.is_monitoring = False
        self._stop_event.set()
        
        # Wait for threads to finish
        for thread in self._threads:
            thread.join(timeout=2.0)
        
        # Release resources
        if self.capture:
            self.capture.release()
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
        
        # Log to W&B
        safe_wandb_log({
            'video_monitoring': {
                'action': 'stop',
                'frames_processed': self.stats['frames_processed'],
                'events_detected': self.stats['events_detected'],
                'timestamp': datetime.now().isoformat()
            }
        })
        
        self.logger.info("Video monitoring stopped")
        return {"status": "success", "message": "Video monitoring stopped"}
    
    def _capture_loop(self):
        """Capture frames from camera"""
        while not self._stop_event.is_set():
            if self.capture and self.capture.isOpened():
                ret, frame = self.capture.read()
                if ret:
                    try:
                        self.frame_queue.put_nowait((self.frame_number, frame))
                        self.frame_number += 1
                    except queue.Full:
                        pass  # Drop frame if queue is full
                else:
                    time.sleep(0.1)
            else:
                time.sleep(0.1)
    
    def _process_loop(self):
        """Process captured frames"""
        while not self._stop_event.is_set():
            try:
                frame_data = self.frame_queue.get(timeout=0.5)
                frame_num, frame = frame_data
                
                # Process every 3rd frame for performance
                if frame_num % 3 == 0:
                    events = self._process_frame(frame, frame_num)
                    
                    # Handle events
                    for event in events:
                        self._handle_event(event)
                
                # Record frame if recording
                if self.video_writer:
                    self.video_writer.write(frame)
                
                # Update statistics
                self.stats['frames_processed'] += 1
                
                # Log statistics periodically
                if frame_num % 100 == 0:
                    self._log_statistics()
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Frame processing error: {e}")
    
    @weave.op()
    def _process_frame(self, frame: np.ndarray, frame_num: int) -> List[VideoEvent]:
        """Process single frame for events"""
        events = []
        
        # Color analysis
        color_result = self.color_analyzer.analyze(frame)
        if color_result['change_detected']:
            event = VideoEvent(
                timestamp=datetime.now(),
                event_type=EventType.COLOR_CHANGE,
                description=f"Color change to {color_result['color_name']} (magnitude: {color_result['change_magnitude']:.1f})",
                confidence=min(color_result['change_magnitude'] / 50, 1.0),
                frame_number=frame_num,
                metadata=color_result
            )
            events.append(event)
        
        # Motion detection
        motion_result = self.motion_detector.detect(frame)
        if motion_result['motion_detected']:
            for region in motion_result['motion_regions']:
                event = VideoEvent(
                    timestamp=datetime.now(),
                    event_type=EventType.MOTION_DETECTED,
                    description=f"Motion detected (area: {region['area']})",
                    confidence=min(region['area'] / 5000, 1.0),
                    frame_number=frame_num,
                    region_of_interest=region['bbox'],
                    metadata=motion_result
                )
                events.append(event)
        
        return events
    
    def _handle_event(self, event: VideoEvent):
        """Handle detected event"""
        # Add to history
        self.event_history.append(event)
        
        # Add to queue
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            self.logger.warning("Event queue full")
        
        # Update statistics
        self.stats['events_detected'] += 1
        
        # Log to W&B
        safe_wandb_log({
            'video_event': {
                'type': event.event_type.value,
                'description': event.description,
                'confidence': event.confidence,
                'frame': event.frame_number,
                'timestamp': event.timestamp.isoformat()
            }
        })
        
        # Call callbacks
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Event callback error: {e}")
        
        # Check for safety violations
        self._check_safety_violations(event)
    
    def _check_safety_violations(self, event: VideoEvent):
        """Check if event indicates a safety violation"""
        # Example safety checks - customize based on experiment needs
        safety_keywords = ['explosion', 'fire', 'smoke', 'leak', 'spill']
        
        if any(keyword in event.description.lower() for keyword in safety_keywords):
            safety_event = VideoEvent(
                timestamp=datetime.now(),
                event_type=EventType.SAFETY_VIOLATION,
                description=f"Safety violation detected: {event.description}",
                confidence=event.confidence,
                frame_number=event.frame_number,
                metadata={'original_event': event.description}
            )
            self.safety_violations.append(safety_event)
            
            # Log safety violation
            safe_wandb_log({
                'safety_violation': {
                    'description': safety_event.description,
                    'timestamp': safety_event.timestamp.isoformat(),
                    'confidence': safety_event.confidence
                }
            })
    
    @weave.op()
    def start_recording(self, filename: Optional[str] = None):
        """Start video recording"""
        if not self.cv2_available:
            return {"status": "error", "message": "OpenCV not available"}
            
        if self.is_recording:
            return {"status": "already_recording"}
        
        try:
            if filename is None:
                filename = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            
            filepath = self.recording_path / filename
            
            # Get video properties
            if self.capture:
                fps = int(self.capture.get(cv2.CAP_PROP_FPS)) or 30
                width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
                height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            else:
                fps, width, height = 30, 1280, 720
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                str(filepath), fourcc, fps, (width, height)
            )
            
            self.is_recording = True
            
            # Log to W&B
            safe_wandb_log({
                'video_recording': {
                    'action': 'start',
                    'filename': filename,
                    'timestamp': datetime.now().isoformat()
                }
            })
            
            self.logger.info(f"Recording started: {filepath}")
            return {"status": "success", "message": f"Recording started: {filename}"}
            
        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}")
            return {"status": "error", "message": str(e)}
    
    @weave.op()
    def stop_recording(self):
        """Stop video recording"""
        if not self.is_recording:
            return {"status": "not_recording"}
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        self.is_recording = False
        
        # Log to W&B
        safe_wandb_log({
            'video_recording': {
                'action': 'stop',
                'timestamp': datetime.now().isoformat()
            }
        })
        
        self.logger.info("Recording stopped")
        return {"status": "success", "message": "Recording stopped"}
    
    def register_callback(self, callback: Callable[[VideoEvent], None]):
        """Register event callback"""
        self.event_callbacks.append(callback)
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame"""
        if self.capture and self.capture.isOpened():
            ret, frame = self.capture.read()
            return frame if ret else None
        return None
    
    def get_event_summary(self) -> Dict[str, Any]:
        """Get summary of detected events"""
        event_counts = {}
        for event in self.event_history:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return {
            'total_events': len(self.event_history),
            'event_counts': event_counts,
            'frames_processed': self.stats['frames_processed'],
            'safety_violations': len(self.safety_violations),
            'monitoring_active': self.is_monitoring,
            'recording_active': self.is_recording
        }
    
    @weave.op()
    def _log_statistics(self):
        """Log monitoring statistics"""
        stats = self.get_event_summary()
        
        safe_wandb_log({
            'video_stats': {
                'frames': self.stats['frames_processed'],
                'events': stats['total_events'],
                'queue_size': self.frame_queue.qsize(),
                'monitoring_active': self.is_monitoring,
                'recording_active': self.is_recording
            }
        })
    
    def get_latest_events(self, limit: int = 10) -> List[VideoEvent]:
        """Get latest video events"""
        return list(self.event_history)[-limit:]
    
    def has_safety_violations(self) -> bool:
        """Check if there are any safety violations"""
        return len(self.safety_violations) > 0
    
    def get_safety_violations(self) -> List[VideoEvent]:
        """Get all safety violations"""
        return self.safety_violations.copy()

# Integration helper function
def create_video_monitoring_agent_rebase_two(camera_index: int = 0) -> VideoMonitoringAgent:
    """Factory function to create video monitoring agent"""
    return VideoMonitoringAgent(camera_index=camera_index)
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
