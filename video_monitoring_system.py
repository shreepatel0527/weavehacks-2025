"""
Advanced video monitoring system for lab experiments
"""
import cv2
import numpy as np
import threading
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import base64
import io
from PIL import Image
import weave

class EventType(Enum):
    COLOR_CHANGE = "color_change"
    MOTION_DETECTED = "motion_detected"
    OBJECT_DETECTED = "object_detected"
    ANOMALY = "anomaly"
    EXPERIMENT_PHASE = "experiment_phase"
    SAFETY_VIOLATION = "safety_violation"

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

class VideoMonitoringSystem:
    """Advanced video monitoring with computer vision analysis"""
    
    def __init__(self, camera_index: int = 0, enable_ml: bool = True):
        self.camera_index = camera_index
        self.enable_ml = enable_ml
        self.is_monitoring = False
        
        # Video capture
        self.capture = None
        self.frame_rate = 30
        self.frame_buffer = queue.Queue(maxsize=100)
        self.event_queue = queue.Queue()
        
        # Processing threads
        self.capture_thread = None
        self.processing_thread = None
        
        # Analysis components
        self.color_analyzer = ColorChangeAnalyzer()
        self.motion_detector = MotionDetector()
        self.object_detector = ObjectDetector() if enable_ml else None
        self.anomaly_detector = AnomalyDetector()
        
        # Recording
        self.is_recording = False
        self.video_writer = None
        self.recording_path = Path("recordings")
        self.recording_path.mkdir(exist_ok=True)
        
        # Event callbacks
        self.event_callbacks = []
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'events_detected': 0,
            'recording_duration': 0
        }
        
        # Initialize W&B
        weave.init('video-monitoring')
    
    def start_monitoring(self):
        """Start video monitoring"""
        if self.is_monitoring:
            return
        
        # Open camera
        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")
        
        # Set camera properties
        self.capture.set(cv2.CAP_PROP_FPS, self.frame_rate)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.is_monitoring = True
        
        # Start threads
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("Video monitoring started")
    
    def stop_monitoring(self):
        """Stop video monitoring"""
        self.is_monitoring = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        
        if self.capture:
            self.capture.release()
        
        if self.is_recording:
            self.stop_recording()
        
        print("Video monitoring stopped")
    
    def _capture_frames(self):
        """Capture frames from camera"""
        frame_count = 0
        
        while self.is_monitoring:
            ret, frame = self.capture.read()
            if ret:
                # Add frame to buffer
                if not self.frame_buffer.full():
                    self.frame_buffer.put((frame_count, frame))
                    frame_count += 1
                
                # Record if enabled
                if self.is_recording and self.video_writer:
                    self.video_writer.write(frame)
            else:
                time.sleep(0.001)
    
    @weave.op()
    def _process_frames(self):
        """Process frames for analysis"""
        last_frame = None
        
        while self.is_monitoring:
            try:
                # Get frame from buffer
                frame_number, frame = self.frame_buffer.get(timeout=0.1)
                self.stats['frames_processed'] += 1
                
                # Analyze frame
                events = []
                
                # Color change detection
                if last_frame is not None:
                    color_events = self.color_analyzer.analyze(
                        last_frame, frame, frame_number
                    )
                    events.extend(color_events)
                
                # Motion detection
                motion_events = self.motion_detector.analyze(
                    frame, frame_number
                )
                events.extend(motion_events)
                
                # Object detection (if ML enabled)
                if self.object_detector:
                    object_events = self.object_detector.analyze(
                        frame, frame_number
                    )
                    events.extend(object_events)
                
                # Anomaly detection
                anomaly_events = self.anomaly_detector.analyze(
                    frame, frame_number, events
                )
                events.extend(anomaly_events)
                
                # Process events
                for event in events:
                    self._handle_event(event)
                
                # Update last frame
                last_frame = frame
                
                # Log periodically
                if frame_number % 100 == 0:
                    self._log_statistics()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Frame processing error: {e}")
    
    def _handle_event(self, event: VideoEvent):
        """Handle detected events"""
        # Add to queue
        self.event_queue.put(event)
        self.stats['events_detected'] += 1
        
        # Log to W&B
        weave.log({
            'video_event': {
                'type': event.event_type.value,
                'description': event.description,
                'confidence': event.confidence,
                'frame_number': event.frame_number,
                'timestamp': event.timestamp.isoformat()
            }
        })
        
        # Trigger callbacks
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"Event callback error: {e}")
        
        # Save event frame if significant
        if event.confidence > 0.8 and event.image_data is not None:
            self._save_event_frame(event)
    
    def _save_event_frame(self, event: VideoEvent):
        """Save frame for significant events"""
        filename = f"event_{event.timestamp.strftime('%Y%m%d_%H%M%S')}_{event.event_type.value}.jpg"
        filepath = self.recording_path / filename
        
        if event.image_data is not None:
            cv2.imwrite(str(filepath), event.image_data)
            event.metadata['saved_frame'] = str(filepath)
    
    def start_recording(self, filename: Optional[str] = None):
        """Start video recording"""
        if self.is_recording:
            return
        
        if filename is None:
            filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        filepath = self.recording_path / filename
        
        # Get video properties
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(filepath), fourcc, fps, (width, height)
        )
        
        self.is_recording = True
        print(f"Recording started: {filepath}")
    
    def stop_recording(self):
        """Stop video recording"""
        if not self.is_recording:
            return
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        self.is_recording = False
        print("Recording stopped")
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame"""
        if self.capture and self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                return frame
        return None
    
    def register_event_callback(self, callback: Callable):
        """Register callback for events"""
        self.event_callbacks.append(callback)
    
    @weave.op()
    def _log_statistics(self):
        """Log monitoring statistics"""
        weave.log({
            'video_stats': {
                'frames_processed': self.stats['frames_processed'],
                'events_detected': self.stats['events_detected'],
                'is_recording': self.is_recording,
                'buffer_size': self.frame_buffer.qsize()
            }
        })

class ColorChangeAnalyzer:
    """Analyze color changes in video frames"""
    
    def __init__(self, threshold: float = 30.0):
        self.threshold = threshold
        self.reference_colors = {}
        self.color_history = []
    
    def analyze(self, prev_frame: np.ndarray, curr_frame: np.ndarray, 
                frame_number: int) -> List[VideoEvent]:
        """Analyze color changes between frames"""
        events = []
        
        # Define regions of interest (could be configurable)
        roi = (200, 200, 600, 400)  # x, y, width, height
        
        # Extract ROI
        prev_roi = prev_frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        curr_roi = curr_frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        
        # Calculate mean colors
        prev_mean = cv2.mean(prev_roi)[:3]
        curr_mean = cv2.mean(curr_roi)[:3]
        
        # Calculate color difference
        color_diff = np.sqrt(sum((c - p)**2 for c, p in zip(curr_mean, prev_mean)))
        
        # Detect significant change
        if color_diff > self.threshold:
            # Determine color transition
            color_desc = self._describe_color_change(prev_mean, curr_mean)
            
            event = VideoEvent(
                timestamp=datetime.now(),
                event_type=EventType.COLOR_CHANGE,
                description=color_desc,
                confidence=min(color_diff / 100, 1.0),
                frame_number=frame_number,
                region_of_interest=roi,
                image_data=curr_frame.copy(),
                metadata={
                    'prev_color': prev_mean,
                    'curr_color': curr_mean,
                    'difference': color_diff
                }
            )
            events.append(event)
        
        # Update history
        self.color_history.append({
            'frame': frame_number,
            'color': curr_mean,
            'timestamp': datetime.now()
        })
        
        # Keep only recent history
        if len(self.color_history) > 100:
            self.color_history.pop(0)
        
        return events
    
    def _describe_color_change(self, prev_color: Tuple, curr_color: Tuple) -> str:
        """Describe the color change in words"""
        prev_name = self._get_color_name(prev_color)
        curr_name = self._get_color_name(curr_color)
        
        return f"Color change from {prev_name} to {curr_name}"
    
    def _get_color_name(self, bgr: Tuple) -> str:
        """Get approximate color name from BGR values"""
        b, g, r = bgr
        
        # Simple color classification
        if r > 150 and g < 100 and b < 100:
            return "red"
        elif r > 150 and g > 150 and b < 100:
            return "yellow"
        elif r < 100 and g > 150 and b < 100:
            return "green"
        elif r < 100 and g < 100 and b > 150:
            return "blue"
        elif r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif abs(r - g) < 20 and abs(g - b) < 20:
            return "gray"
        else:
            return "mixed"

class MotionDetector:
    """Detect motion in video frames"""
    
    def __init__(self, threshold: float = 25.0):
        self.threshold = threshold
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.motion_history = []
    
    def analyze(self, frame: np.ndarray, frame_number: int) -> List[VideoEvent]:
        """Analyze motion in frame"""
        events = []
        
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Analyze significant contours
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                
                event = VideoEvent(
                    timestamp=datetime.now(),
                    event_type=EventType.MOTION_DETECTED,
                    description=f"Motion detected in region ({x}, {y}, {w}, {h})",
                    confidence=min(area / 5000, 1.0),
                    frame_number=frame_number,
                    region_of_interest=(x, y, w, h),
                    metadata={
                        'area': area,
                        'center': (x + w//2, y + h//2)
                    }
                )
                events.append(event)
        
        return events

class ObjectDetector:
    """Detect specific objects in frames using ML"""
    
    def __init__(self):
        # Simplified object detection (would use YOLO or similar in production)
        self.target_objects = [
            'beaker', 'flask', 'syringe', 'hand', 'stirrer'
        ]
        # In production, load actual ML model here
    
    def analyze(self, frame: np.ndarray, frame_number: int) -> List[VideoEvent]:
        """Detect objects in frame"""
        events = []
        
        # Simplified detection (replace with actual ML inference)
        # This is a placeholder for demonstration
        
        # Simulate detection of a beaker
        if frame_number % 50 == 0:  # Simulate periodic detection
            event = VideoEvent(
                timestamp=datetime.now(),
                event_type=EventType.OBJECT_DETECTED,
                description="Beaker detected in frame",
                confidence=0.85,
                frame_number=frame_number,
                region_of_interest=(300, 200, 150, 200),
                metadata={
                    'object_class': 'beaker',
                    'detection_score': 0.85
                }
            )
            events.append(event)
        
        return events

class AnomalyDetector:
    """Detect anomalies in experiment video"""
    
    def __init__(self):
        self.expected_patterns = {
            'color_sequence': ['clear', 'yellow', 'red', 'clear'],
            'motion_regions': [(200, 200, 600, 400)],
            'duration_limits': {
                'stirring': (600, 1200),  # 10-20 minutes
                'color_change': (30, 300)  # 30s-5min
            }
        }
        self.pattern_history = []
    
    def analyze(self, frame: np.ndarray, frame_number: int, 
                recent_events: List[VideoEvent]) -> List[VideoEvent]:
        """Detect anomalies based on expected patterns"""
        events = []
        
        # Check for unexpected event sequences
        if recent_events:
            for event in recent_events:
                if event.event_type == EventType.COLOR_CHANGE:
                    # Check if color change is expected
                    if not self._is_expected_color_change(event):
                        anomaly_event = VideoEvent(
                            timestamp=datetime.now(),
                            event_type=EventType.ANOMALY,
                            description=f"Unexpected color change: {event.description}",
                            confidence=0.7,
                            frame_number=frame_number,
                            image_data=frame.copy(),
                            metadata={
                                'original_event': event.description,
                                'expected_pattern': self.expected_patterns['color_sequence']
                            }
                        )
                        events.append(anomaly_event)
        
        return events
    
    def _is_expected_color_change(self, event: VideoEvent) -> bool:
        """Check if color change matches expected pattern"""
        # Simplified check - in production would be more sophisticated
        color_name = event.description.split()[-1]
        return color_name in self.expected_patterns['color_sequence']

# Example usage with integration
class VideoExperimentMonitor:
    """High-level video monitoring for experiments"""
    
    def __init__(self):
        self.video_system = VideoMonitoringSystem(enable_ml=True)
        self.experiment_phases = []
        self.critical_events = []
        
        # Register event handler
        self.video_system.register_event_callback(self.handle_video_event)
    
    def handle_video_event(self, event: VideoEvent):
        """Handle video events for experiment monitoring"""
        print(f"Video Event: {event.event_type.value} - {event.description}")
        
        # Log critical events
        if event.event_type in [EventType.ANOMALY, EventType.SAFETY_VIOLATION]:
            self.critical_events.append(event)
            
            # Trigger safety protocols if needed
            if event.event_type == EventType.SAFETY_VIOLATION:
                self.trigger_safety_protocol(event)
        
        # Track experiment phases
        if event.event_type == EventType.COLOR_CHANGE:
            self.track_experiment_phase(event)
    
    def trigger_safety_protocol(self, event: VideoEvent):
        """Trigger safety protocols based on video detection"""
        print(f"SAFETY ALERT: {event.description}")
        # In production, would integrate with safety systems
    
    def track_experiment_phase(self, event: VideoEvent):
        """Track experiment phases based on visual cues"""
        self.experiment_phases.append({
            'timestamp': event.timestamp,
            'phase': event.description,
            'confidence': event.confidence
        })
    
    def start_monitoring(self):
        """Start experiment monitoring"""
        self.video_system.start_monitoring()
        self.video_system.start_recording()
    
    def stop_monitoring(self):
        """Stop experiment monitoring"""
        self.video_system.stop_monitoring()
        
        # Generate report
        report = {
            'total_events': self.video_system.stats['events_detected'],
            'critical_events': len(self.critical_events),
            'experiment_phases': self.experiment_phases
        }
        
        return report

if __name__ == "__main__":
    # Demo video monitoring
    monitor = VideoExperimentMonitor()
    
    try:
        print("Starting video monitoring...")
        monitor.start_monitoring()
        
        # Run for demonstration
        time.sleep(30)
        
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        report = monitor.stop_monitoring()
        print(f"\nMonitoring Report: {json.dumps(report, indent=2, default=str)}")