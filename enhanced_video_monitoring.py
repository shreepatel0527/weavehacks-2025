"""
Enhanced video monitoring system with advanced ML capabilities
"""
import cv2
import numpy as np
import threading
import queue
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import base64
import io
from PIL import Image
import weave
import torch
import torchvision.transforms as transforms
from collections import deque
import logging

# Try to import YOLO for object detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available. Install with: pip install ultralytics")

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

@dataclass
class TrackedObject:
    """Object being tracked across frames"""
    id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    first_seen: int
    last_seen: int
    positions: List[Tuple[int, int]]
    
    def update(self, bbox: Tuple[int, int, int, int], frame_num: int):
        """Update tracked object"""
        self.bbox = bbox
        self.last_seen = frame_num
        center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
        self.positions.append(center)

class EnhancedColorAnalyzer:
    """Advanced color analysis with HSV and Lab color spaces"""
    
    def __init__(self, sensitivity: float = 15.0):
        self.sensitivity = sensitivity
        self.color_history = deque(maxlen=30)
        self.reference_colors = {}
        self.color_profiles = self._load_color_profiles()
        
    def _load_color_profiles(self) -> Dict[str, Dict]:
        """Load known color profiles for experiments"""
        return {
            'gold_nanoparticles': {
                'initial': {'h': 30, 's': 20, 'v': 80},  # Light yellow
                'intermediate': {'h': 0, 's': 100, 'v': 50},  # Red
                'final': {'h': 340, 's': 80, 'v': 30}  # Dark red
            },
            'silver_nanoparticles': {
                'initial': {'h': 0, 's': 0, 'v': 90},  # Clear
                'intermediate': {'h': 60, 's': 50, 'v': 60},  # Yellow
                'final': {'h': 30, 's': 30, 'v': 40}  # Brown
            }
        }
    
    def analyze(self, frame: np.ndarray, roi: Optional[Tuple] = None) -> Dict[str, Any]:
        """Analyze color in frame or ROI"""
        # Extract ROI if specified
        if roi:
            x, y, w, h = roi
            analysis_region = frame[y:y+h, x:x+w]
        else:
            # Use center region
            h, w = frame.shape[:2]
            analysis_region = frame[h//4:3*h//4, w//4:3*w//4]
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(analysis_region, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(analysis_region, cv2.COLOR_BGR2LAB)
        
        # Calculate color statistics
        hsv_mean = cv2.mean(hsv)[:3]
        lab_mean = cv2.mean(lab)[:3]
        
        # Calculate dominant color
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        dominant_hue = np.argmax(hist)
        
        # Store in history
        color_data = {
            'hsv': hsv_mean,
            'lab': lab_mean,
            'dominant_hue': int(dominant_hue),
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
        
        # Match against known profiles
        matched_phase = self._match_color_profile(hsv_mean)
        
        return {
            'current_color': color_data,
            'change_detected': change_detected,
            'change_magnitude': change_magnitude,
            'matched_phase': matched_phase,
            'color_trajectory': self._analyze_trajectory()
        }
    
    def _match_color_profile(self, hsv_color: Tuple) -> Optional[str]:
        """Match current color against known profiles"""
        best_match = None
        min_distance = float('inf')
        
        for experiment, phases in self.color_profiles.items():
            for phase, target in phases.items():
                # Calculate HSV distance
                distance = np.sqrt(
                    (hsv_color[0] - target['h']) ** 2 +
                    (hsv_color[1] - target['s']) ** 2 +
                    (hsv_color[2] - target['v']) ** 2
                )
                
                if distance < min_distance and distance < 30:
                    min_distance = distance
                    best_match = f"{experiment}_{phase}"
        
        return best_match
    
    def _analyze_trajectory(self) -> Dict[str, Any]:
        """Analyze color change trajectory"""
        if len(self.color_history) < 5:
            return {'trend': 'insufficient_data'}
        
        # Get recent hue values
        recent_hues = [c['dominant_hue'] for c in list(self.color_history)[-10:]]
        
        # Calculate trend
        hue_diff = recent_hues[-1] - recent_hues[0]
        
        if abs(hue_diff) < 5:
            trend = 'stable'
        elif hue_diff > 0:
            trend = 'shifting_red'
        else:
            trend = 'shifting_blue'
        
        return {
            'trend': trend,
            'rate': abs(hue_diff) / len(recent_hues)
        }

class EnhancedMotionDetector:
    """Motion detection with optical flow and blob tracking"""
    
    def __init__(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True
        )
        self.previous_frame = None
        self.flow_threshold = 2.0
        self.min_area = 500
        
        # Optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
    
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect motion in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Remove shadows
        fg_mask[fg_mask == 127] = 0
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and analyze contours
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
        
        # Optical flow analysis
        flow_data = None
        if self.previous_frame is not None:
            flow_data = self._analyze_optical_flow(self.previous_frame, gray)
        
        self.previous_frame = gray.copy()
        
        return {
            'motion_detected': len(motion_regions) > 0,
            'motion_regions': motion_regions,
            'total_motion_area': total_motion_area,
            'motion_mask': fg_mask,
            'optical_flow': flow_data
        }
    
    def _analyze_optical_flow(self, prev_gray: np.ndarray, 
                            curr_gray: np.ndarray) -> Dict[str, Any]:
        """Analyze optical flow between frames"""
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calculate flow statistics
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Detect significant flow
        significant_flow = magnitude > self.flow_threshold
        flow_direction = np.mean(angle[significant_flow]) if np.any(significant_flow) else 0
        flow_magnitude = np.mean(magnitude[significant_flow]) if np.any(significant_flow) else 0
        
        return {
            'average_magnitude': float(flow_magnitude),
            'primary_direction': float(flow_direction),
            'flow_coverage': float(np.sum(significant_flow) / significant_flow.size)
        }

class MLObjectDetector:
    """ML-based object detection for lab equipment and materials"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Lab-specific classes
        self.lab_classes = [
            'beaker', 'flask', 'pipette', 'syringe', 'bottle',
            'stirrer', 'hot_plate', 'scale', 'thermometer',
            'gloves', 'hand', 'liquid', 'solid', 'gas_bubbles'
        ]
        
        # Object tracking
        self.tracked_objects = {}
        self.next_object_id = 1
        self.iou_threshold = 0.5
        
        # Load model
        if YOLO_AVAILABLE:
            if model_path and Path(model_path).exists():
                self.model = YOLO(model_path)
            else:
                # Use pre-trained model
                self.model = YOLO('yolov8n.pt')
        else:
            self.model = None
            
    def detect(self, frame: np.ndarray, frame_number: int) -> List[Dict[str, Any]]:
        """Detect objects in frame"""
        if self.model is None:
            return []
        
        # Run detection
        results = self.model(frame, verbose=False)
        
        detections = []
        current_frame_objects = []
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    # Extract detection info
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Get class name
                    class_name = self.model.names.get(class_id, 'unknown')
                    
                    # Filter for lab-relevant objects
                    if confidence > 0.5:
                        bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                        
                        detection = {
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': bbox,
                            'frame_number': frame_number
                        }
                        
                        # Track object
                        tracked_id = self._update_tracking(detection, frame_number)
                        detection['track_id'] = tracked_id
                        
                        detections.append(detection)
                        current_frame_objects.append(tracked_id)
        
        # Update tracking state
        self._cleanup_lost_tracks(current_frame_objects, frame_number)
        
        return detections
    
    def _update_tracking(self, detection: Dict, frame_number: int) -> int:
        """Update object tracking"""
        bbox = detection['bbox']
        
        # Find matching tracked object
        best_match_id = None
        best_iou = 0
        
        for obj_id, tracked in self.tracked_objects.items():
            if tracked.class_name == detection['class']:
                iou = self._calculate_iou(bbox, tracked.bbox)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_match_id = obj_id
        
        if best_match_id:
            # Update existing track
            self.tracked_objects[best_match_id].update(bbox, frame_number)
            return best_match_id
        else:
            # Create new track
            new_id = self.next_object_id
            self.next_object_id += 1
            
            center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
            self.tracked_objects[new_id] = TrackedObject(
                id=new_id,
                class_name=detection['class'],
                bbox=bbox,
                confidence=detection['confidence'],
                first_seen=frame_number,
                last_seen=frame_number,
                positions=[center]
            )
            
            return new_id
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate Intersection over Union"""
        x1_1, y1_1, w1, h1 = bbox1
        x1_2, y1_2, w2, h2 = bbox2
        
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 < xi1 or yi2 < yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _cleanup_lost_tracks(self, current_objects: List[int], frame_number: int):
        """Remove lost tracks"""
        lost_threshold = 10  # frames
        
        to_remove = []
        for obj_id, tracked in self.tracked_objects.items():
            if obj_id not in current_objects:
                if frame_number - tracked.last_seen > lost_threshold:
                    to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]

class SpecializedDetectors:
    """Specialized detectors for lab-specific events"""
    
    def __init__(self):
        self.liquid_detector = LiquidLevelDetector()
        self.bubble_detector = BubbleDetector()
        self.crystal_detector = CrystallizationDetector()
    
    def detect_all(self, frame: np.ndarray) -> Dict[str, Any]:
        """Run all specialized detectors"""
        results = {
            'liquid_level': self.liquid_detector.detect(frame),
            'bubbles': self.bubble_detector.detect(frame),
            'crystallization': self.crystal_detector.detect(frame)
        }
        
        return results

class LiquidLevelDetector:
    """Detect liquid levels in containers"""
    
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect liquid level"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find horizontal lines (potential liquid surfaces)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 100,
            minLineLength=100, maxLineGap=10
        )
        
        liquid_levels = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Check if line is mostly horizontal
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
                if angle < np.pi / 6:  # Less than 30 degrees
                    liquid_levels.append({
                        'line': (x1, y1, x2, y2),
                        'y_position': (y1 + y2) // 2,
                        'confidence': 0.7
                    })
        
        return {
            'levels_detected': len(liquid_levels),
            'liquid_levels': liquid_levels
        }

class BubbleDetector:
    """Detect gas bubbles in liquids"""
    
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect bubbles"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect circles (bubbles)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=5, maxRadius=50
        )
        
        bubbles = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                bubbles.append({
                    'center': (int(x), int(y)),
                    'radius': int(r),
                    'area': np.pi * r * r
                })
        
        return {
            'bubble_count': len(bubbles),
            'bubbles': bubbles,
            'total_bubble_area': sum(b['area'] for b in bubbles)
        }

class CrystallizationDetector:
    """Detect crystallization patterns"""
    
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect crystallization"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Detect sharp edges (crystal boundaries)
        edges = cv2.Canny(enhanced, 100, 200)
        
        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Analyze contours for crystal-like shapes
        crystals = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum crystal size
                # Approximate polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Crystals often have angular shapes
                if len(approx) >= 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    crystals.append({
                        'bbox': (x, y, w, h),
                        'vertices': len(approx),
                        'area': area,
                        'aspect_ratio': w / h if h > 0 else 0
                    })
        
        return {
            'crystal_count': len(crystals),
            'crystals': crystals,
            'crystallization_detected': len(crystals) > 5
        }

class EnhancedVideoMonitoringSystem:
    """Complete enhanced video monitoring system"""
    
    def __init__(self, camera_index: int = 0, 
                 enable_ml: bool = True,
                 enable_recording: bool = True):
        self.camera_index = camera_index
        self.enable_ml = enable_ml
        self.enable_recording = enable_recording
        
        # Core components
        self.color_analyzer = EnhancedColorAnalyzer()
        self.motion_detector = EnhancedMotionDetector()
        self.object_detector = MLObjectDetector() if enable_ml else None
        self.specialized_detectors = SpecializedDetectors()
        
        # Video capture
        self.capture = None
        self.is_monitoring = False
        self.frame_queue = queue.Queue(maxsize=30)
        self.event_queue = queue.Queue(maxsize=100)
        
        # Recording
        self.video_writer = None
        self.recording_path = Path("recordings")
        self.recording_path.mkdir(exist_ok=True)
        
        # Frame processing
        self.frame_number = 0
        self.skip_frames = 2  # Process every Nth frame for performance
        
        # Event handling
        self.event_callbacks = []
        self.event_history = deque(maxlen=1000)
        
        # Threading
        self._stop_event = threading.Event()
        self._threads = []
        
        # Logging
        self.logger = logging.getLogger('video_monitoring')
        
        # Initialize W&B
        weave.init('enhanced-video-monitoring')
    
    def start(self):
        """Start video monitoring"""
        if self.is_monitoring:
            return
        
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
                raise RuntimeError("Cannot open any camera")
        
        # Configure camera
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_monitoring = True
        self._stop_event.clear()
        
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
        
        # Start recording if enabled
        if self.enable_recording:
            self.start_recording()
        
        self.logger.info("Video monitoring started")
    
    def stop(self):
        """Stop video monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self._stop_event.set()
        
        # Wait for threads
        for thread in self._threads:
            thread.join(timeout=2.0)
        
        # Release resources
        if self.capture:
            self.capture.release()
        
        if self.video_writer:
            self.video_writer.release()
        
        self.logger.info("Video monitoring stopped")
    
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
                        # Drop frame if queue is full
                        pass
                else:
                    self.logger.warning("Failed to capture frame")
                    time.sleep(0.1)
            else:
                time.sleep(0.1)
    
    def _process_loop(self):
        """Process captured frames"""
        while not self._stop_event.is_set():
            try:
                # Get frame with timeout
                frame_data = self.frame_queue.get(timeout=0.5)
                frame_num, frame = frame_data
                
                # Skip frames for performance
                if frame_num % self.skip_frames != 0:
                    # Still record all frames
                    if self.video_writer:
                        self.video_writer.write(frame)
                    continue
                
                # Process frame
                events = self._process_frame(frame, frame_num)
                
                # Handle events
                for event in events:
                    self._handle_event(event)
                
                # Record frame
                if self.video_writer:
                    # Annotate frame with events
                    annotated = self._annotate_frame(frame, events)
                    self.video_writer.write(annotated)
                
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
                description=f"Color change detected (magnitude: {color_result['change_magnitude']:.1f})",
                confidence=0.8,
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
                    description=f"Motion in region {region['bbox']}",
                    confidence=0.7,
                    frame_number=frame_num,
                    region_of_interest=region['bbox'],
                    metadata=motion_result
                )
                events.append(event)
        
        # Object detection
        if self.object_detector:
            detections = self.object_detector.detect(frame, frame_num)
            for detection in detections:
                event = VideoEvent(
                    timestamp=datetime.now(),
                    event_type=EventType.OBJECT_DETECTED,
                    description=f"{detection['class']} detected",
                    confidence=detection['confidence'],
                    frame_number=frame_num,
                    region_of_interest=detection['bbox'],
                    metadata=detection
                )
                events.append(event)
        
        # Specialized detection
        special_results = self.specialized_detectors.detect_all(frame)
        
        # Check for bubbles
        if special_results['bubbles']['bubble_count'] > 10:
            event = VideoEvent(
                timestamp=datetime.now(),
                event_type=EventType.BUBBLE_FORMATION,
                description=f"{special_results['bubbles']['bubble_count']} bubbles detected",
                confidence=0.8,
                frame_number=frame_num,
                metadata=special_results['bubbles']
            )
            events.append(event)
        
        # Check for crystallization
        if special_results['crystallization']['crystallization_detected']:
            event = VideoEvent(
                timestamp=datetime.now(),
                event_type=EventType.CRYSTALLIZATION,
                description="Crystallization pattern detected",
                confidence=0.7,
                frame_number=frame_num,
                metadata=special_results['crystallization']
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
        
        # Log to W&B
        weave.log({
            'video_event': {
                'type': event.event_type.value,
                'description': event.description,
                'confidence': event.confidence,
                'frame': event.frame_number
            }
        })
        
        # Call callbacks
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Event callback error: {e}")
    
    def _annotate_frame(self, frame: np.ndarray, 
                       events: List[VideoEvent]) -> np.ndarray:
        """Annotate frame with event information"""
        annotated = frame.copy()
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(annotated, timestamp, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add event annotations
        y_offset = 60
        for event in events[:5]:  # Limit to prevent cluttering
            # Draw bounding box if available
            if event.region_of_interest:
                x, y, w, h = event.region_of_interest
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add label
                label = f"{event.event_type.value}: {event.confidence:.2f}"
                cv2.putText(annotated, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Add event text
            cv2.putText(annotated, event.description, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += 25
        
        return annotated
    
    def start_recording(self, filename: Optional[str] = None):
        """Start video recording"""
        if self.video_writer:
            return
        
        if filename is None:
            filename = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        filepath = self.recording_path / filename
        
        # Get video properties
        if self.capture:
            fps = int(self.capture.get(cv2.CAP_PROP_FPS))
            width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            fps, width, height = 30, 1280, 720
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(filepath), fourcc, fps, (width, height)
        )
        
        self.logger.info(f"Recording started: {filepath}")
    
    def stop_recording(self):
        """Stop video recording"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.logger.info("Recording stopped")
    
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
            'frames_processed': self.frame_number,
            'monitoring_duration': self.frame_number / 30.0  # Assuming 30 FPS
        }
    
    @weave.op()
    def _log_statistics(self):
        """Log monitoring statistics"""
        stats = self.get_event_summary()
        
        weave.log({
            'video_stats': {
                'frames': self.frame_number,
                'events': stats['total_events'],
                'queue_size': self.frame_queue.qsize()
            }
        })

# Example usage
def demo_enhanced_video():
    """Demonstrate enhanced video monitoring"""
    system = EnhancedVideoMonitoringSystem(
        camera_index=0,
        enable_ml=True,
        enable_recording=True
    )
    
    def event_handler(event: VideoEvent):
        """Handle video events"""
        print(f"Event: {event.event_type.value} - {event.description}")
        
        # Special handling for safety events
        if event.event_type == EventType.SAFETY_VIOLATION:
            print("SAFETY ALERT!")
    
    system.register_callback(event_handler)
    
    try:
        print("Starting enhanced video monitoring...")
        system.start()
        
        # Run for demo
        time.sleep(30)
        
        # Get summary
        summary = system.get_event_summary()
        print(f"\nSummary: {json.dumps(summary, indent=2)}")
        
    finally:
        system.stop()
        print("Video monitoring stopped")

if __name__ == "__main__":
    demo_enhanced_video()