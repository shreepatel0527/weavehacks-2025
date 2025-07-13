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
def create_video_monitoring_agent(camera_index: int = 0) -> VideoMonitoringAgent:
    """Factory function to create video monitoring agent"""
    return VideoMonitoringAgent(camera_index=camera_index)