"""
Thread-safe real-time safety monitoring system with improved synchronization
"""
import asyncio
import threading
import queue
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import weave
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
import logging

class SafetyLevel(Enum):
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class SafetyEvent:
    timestamp: datetime
    level: SafetyLevel
    parameter: str
    value: float
    threshold: float
    message: str

class ThreadSafeBuffer:
    """Thread-safe circular buffer for parameter history"""
    
    def __init__(self, maxlen: int = 100):
        self._buffer = deque(maxlen=maxlen)
        self._lock = threading.RLock()
    
    def append(self, item):
        """Add item to buffer"""
        with self._lock:
            self._buffer.append(item)
    
    def get_latest(self, n: int = 1) -> List:
        """Get latest n items"""
        with self._lock:
            return list(self._buffer)[-n:]
    
    def get_all(self) -> List:
        """Get all items"""
        with self._lock:
            return list(self._buffer)
    
    def clear(self):
        """Clear buffer"""
        with self._lock:
            self._buffer.clear()
    
    @property
    def length(self) -> int:
        """Get buffer length"""
        with self._lock:
            return len(self._buffer)

class ThreadSafeRealtimeSafetyMonitor:
    """Thread-safe continuous safety monitoring with real-time alerts"""
    
    def __init__(self, config_path: Optional[Path] = None, check_interval: float = 1.0):
        # Configuration
        self.config = self._load_config(config_path)
        self.check_interval = check_interval
        
        # Thread synchronization
        self._state_lock = threading.RLock()
        self._is_monitoring = False
        self._monitoring_thread = None
        self._stop_event = threading.Event()
        
        # Thread-safe queues and buffers
        self.event_queue = queue.Queue(maxsize=1000)
        self.command_queue = queue.Queue(maxsize=100)
        
        # Thread-safe parameter history
        self.parameter_history = {
            'temperature': ThreadSafeBuffer(100),
            'pressure': ThreadSafeBuffer(100),
            'nitrogen': ThreadSafeBuffer(100),
            'oxygen': ThreadSafeBuffer(100),
            'stirring_rpm': ThreadSafeBuffer(100),
            'ph': ThreadSafeBuffer(100)
        }
        
        # Thread-safe state
        self._current_readings = {}
        self._current_safety_level = SafetyLevel.SAFE
        self._readings_lock = threading.RLock()
        
        # Alert management
        self._alert_callbacks = []
        self._alert_lock = threading.RLock()
        self._last_alert_times = {}
        self._alert_cooldown = timedelta(seconds=60)
        
        # Statistics
        self._stats = {
            'total_readings': 0,
            'total_alerts': 0,
            'alerts_by_level': {level: 0 for level in SafetyLevel},
            'start_time': None
        }
        self._stats_lock = threading.RLock()
        
        # Setup logging
        self.logger = logging.getLogger('safety_monitor')
        
        # Initialize W&B
        weave.init('realtime-safety-monitor')
    
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load configuration with defaults"""
        default_config = {
            'thresholds': {
                'temperature': {'min': 0, 'max': 30, 'critical': 35},
                'pressure': {'min': 90, 'max': 110, 'critical': 120},
                'nitrogen': {'min': 75, 'max': 85, 'critical': 90},
                'oxygen': {'min': 19, 'max': 23, 'critical': 25},
                'stirring_rpm': {'min': 800, 'max': 1200, 'critical': 1500},
                'ph': {'min': 6.0, 'max': 8.0, 'critical': 9.0}
            },
            'sensor_file': 'Prototype-1/sensor_data.json',
            'alert_cooldown': 60,
            'emergency_stop_enabled': True
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        
        return default_config
    
    @property
    def is_monitoring(self) -> bool:
        """Thread-safe check if monitoring"""
        with self._state_lock:
            return self._is_monitoring
    
    @property
    def current_safety_level(self) -> SafetyLevel:
        """Thread-safe get current safety level"""
        with self._readings_lock:
            return self._current_safety_level
    
    @property
    def current_readings(self) -> Dict[str, float]:
        """Thread-safe get current readings"""
        with self._readings_lock:
            return dict(self._current_readings)
    
    def start_monitoring(self):
        """Start safety monitoring in background thread"""
        with self._state_lock:
            if self._is_monitoring:
                self.logger.warning("Monitoring already started")
                return
            
            self._is_monitoring = True
            self._stop_event.clear()
            
            # Update stats
            with self._stats_lock:
                self._stats['start_time'] = datetime.now()
            
            # Start monitoring thread
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="SafetyMonitorThread",
                daemon=True
            )
            self._monitoring_thread.start()
            
            # Start command processor thread
            self._command_thread = threading.Thread(
                target=self._process_commands,
                name="SafetyCommandThread",
                daemon=True
            )
            self._command_thread.start()
            
            self.logger.info("Safety monitoring started")
    
    def stop_monitoring(self, timeout: float = 5.0):
        """Stop safety monitoring gracefully"""
        with self._state_lock:
            if not self._is_monitoring:
                return
            
            self._is_monitoring = False
            self._stop_event.set()
        
        # Wait for threads to finish
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout)
            if self._monitoring_thread.is_alive():
                self.logger.warning("Monitoring thread did not stop gracefully")
        
        if hasattr(self, '_command_thread') and self._command_thread:
            self._command_thread.join(timeout)
        
        self.logger.info("Safety monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop - runs in separate thread"""
        self.logger.info("Monitoring loop started")
        
        while not self._stop_event.is_set():
            try:
                # Read sensor data
                readings = self._read_sensors()
                
                # Update current readings
                with self._readings_lock:
                    self._current_readings = readings
                
                # Store in history
                timestamp = datetime.now()
                for param, value in readings.items():
                    if param in self.parameter_history:
                        self.parameter_history[param].append({
                            'timestamp': timestamp,
                            'value': value
                        })
                
                # Check safety violations
                violations = self._check_violations(readings)
                
                # Determine safety level
                safety_level = self._determine_safety_level(violations)
                
                # Update current safety level
                with self._readings_lock:
                    self._current_safety_level = safety_level
                
                # Handle violations
                if violations:
                    self._handle_violations(violations, safety_level)
                
                # Update statistics
                with self._stats_lock:
                    self._stats['total_readings'] += 1
                
                # Log periodically
                if self._stats['total_readings'] % 10 == 0:
                    self._log_status()
                
                # Sleep for interval
                self._stop_event.wait(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self._stop_event.wait(1.0)  # Brief pause on error
    
    def _process_commands(self):
        """Process commands from command queue"""
        while not self._stop_event.is_set():
            try:
                # Get command with timeout
                command = self.command_queue.get(timeout=0.5)
                
                if command['type'] == 'update_threshold':
                    self._update_threshold(
                        command['parameter'],
                        command['threshold_type'],
                        command['value']
                    )
                elif command['type'] == 'emergency_stop':
                    self._execute_emergency_stop()
                elif command['type'] == 'reset_alerts':
                    self._reset_alerts()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing command: {e}")
    
    def _read_sensors(self) -> Dict[str, float]:
        """Read sensor data from file or generate test data"""
        sensor_file = Path(self.config['sensor_file'])
        
        if sensor_file.exists():
            try:
                with open(sensor_file, 'r') as f:
                    data = json.load(f)
                    return {
                        'temperature': data.get('temperature', 23.0),
                        'pressure': data.get('pressure', 101.3),
                        'nitrogen': data.get('nitrogen', 78.0),
                        'oxygen': data.get('oxygen', 21.0),
                        'stirring_rpm': data.get('stirring_rpm', 1000),
                        'ph': data.get('ph', 7.0)
                    }
            except Exception as e:
                self.logger.error(f"Error reading sensor file: {e}")
        
        # Generate test data with some variation
        base_values = {
            'temperature': 23.0,
            'pressure': 101.3,
            'nitrogen': 78.0,
            'oxygen': 21.0,
            'stirring_rpm': 1000,
            'ph': 7.0
        }
        
        # Add random variation
        return {
            param: value + np.random.normal(0, value * 0.01)
            for param, value in base_values.items()
        }
    
    def _check_violations(self, readings: Dict[str, float]) -> List[Dict]:
        """Check for threshold violations"""
        violations = []
        thresholds = self.config['thresholds']
        
        for param, value in readings.items():
            if param not in thresholds:
                continue
            
            limits = thresholds[param]
            
            # Check critical threshold
            if 'critical' in limits:
                if value > limits['critical'] or (param == 'oxygen' and value < 18):
                    violations.append({
                        'parameter': param,
                        'value': value,
                        'threshold': limits['critical'],
                        'type': 'critical',
                        'message': f"CRITICAL: {param} at {value:.2f} exceeds critical threshold"
                    })
                    continue
            
            # Check normal range
            if value < limits['min']:
                violations.append({
                    'parameter': param,
                    'value': value,
                    'threshold': limits['min'],
                    'type': 'low',
                    'message': f"WARNING: {param} at {value:.2f} below minimum"
                })
            elif value > limits['max']:
                violations.append({
                    'parameter': param,
                    'value': value,
                    'threshold': limits['max'],
                    'type': 'high',
                    'message': f"WARNING: {param} at {value:.2f} above maximum"
                })
        
        return violations
    
    def _determine_safety_level(self, violations: List[Dict]) -> SafetyLevel:
        """Determine overall safety level"""
        if not violations:
            return SafetyLevel.SAFE
        
        # Check for critical violations
        critical_count = sum(1 for v in violations if v['type'] == 'critical')
        
        if critical_count >= 2:
            return SafetyLevel.EMERGENCY
        elif critical_count == 1:
            return SafetyLevel.CRITICAL
        elif len(violations) >= 3:
            return SafetyLevel.WARNING
        else:
            return SafetyLevel.WARNING
    
    def _handle_violations(self, violations: List[Dict], safety_level: SafetyLevel):
        """Handle safety violations"""
        for violation in violations:
            # Create safety event
            event = SafetyEvent(
                timestamp=datetime.now(),
                level=safety_level,
                parameter=violation['parameter'],
                value=violation['value'],
                threshold=violation['threshold'],
                message=violation['message']
            )
            
            # Add to event queue
            try:
                self.event_queue.put_nowait(event)
            except queue.Full:
                self.logger.warning("Event queue full, dropping event")
            
            # Check if we should send alert
            if self._should_send_alert(violation['parameter']):
                self._send_alert(event)
            
            # Update statistics
            with self._stats_lock:
                self._stats['total_alerts'] += 1
                self._stats['alerts_by_level'][safety_level] += 1
        
        # Handle emergency
        if safety_level == SafetyLevel.EMERGENCY and self.config['emergency_stop_enabled']:
            self._execute_emergency_stop()
    
    def _should_send_alert(self, parameter: str) -> bool:
        """Check if alert should be sent (cooldown logic)"""
        now = datetime.now()
        
        if parameter in self._last_alert_times:
            time_since_last = now - self._last_alert_times[parameter]
            if time_since_last < self._alert_cooldown:
                return False
        
        self._last_alert_times[parameter] = now
        return True
    
    def _send_alert(self, event: SafetyEvent):
        """Send alert to registered callbacks"""
        with self._alert_lock:
            callbacks = list(self._alert_callbacks)
        
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def _execute_emergency_stop(self):
        """Execute emergency stop procedure"""
        self.logger.critical("EMERGENCY STOP INITIATED")
        
        # Create emergency event
        event = SafetyEvent(
            timestamp=datetime.now(),
            level=SafetyLevel.EMERGENCY,
            parameter='system',
            value=0,
            threshold=0,
            message="EMERGENCY STOP - Multiple critical violations"
        )
        
        # Send to all callbacks
        self._send_alert(event)
        
        # Log to W&B
        weave.log({
            'emergency_stop': {
                'timestamp': datetime.now().isoformat(),
                'safety_level': SafetyLevel.EMERGENCY.value
            }
        })
    
    @weave.op()
    def _log_status(self):
        """Log current status"""
        with self._stats_lock:
            stats = dict(self._stats)
        
        with self._readings_lock:
            readings = dict(self._current_readings)
            safety_level = self._current_safety_level
        
        weave.log({
            'safety_status': {
                'timestamp': datetime.now().isoformat(),
                'safety_level': safety_level.value,
                'readings': readings,
                'total_readings': stats['total_readings'],
                'total_alerts': stats['total_alerts']
            }
        })
    
    def register_alert_callback(self, callback: Callable[[SafetyEvent], None]):
        """Register callback for alerts"""
        with self._alert_lock:
            self._alert_callbacks.append(callback)
    
    def unregister_alert_callback(self, callback: Callable[[SafetyEvent], None]):
        """Unregister alert callback"""
        with self._alert_lock:
            if callback in self._alert_callbacks:
                self._alert_callbacks.remove(callback)
    
    def send_command(self, command: Dict):
        """Send command to monitoring thread"""
        try:
            self.command_queue.put_nowait(command)
        except queue.Full:
            self.logger.warning("Command queue full")
    
    def update_threshold(self, parameter: str, threshold_type: str, value: float):
        """Update threshold value"""
        self.send_command({
            'type': 'update_threshold',
            'parameter': parameter,
            'threshold_type': threshold_type,
            'value': value
        })
    
    def _update_threshold(self, parameter: str, threshold_type: str, value: float):
        """Internal threshold update"""
        if parameter in self.config['thresholds']:
            self.config['thresholds'][parameter][threshold_type] = value
            self.logger.info(f"Updated {parameter} {threshold_type} to {value}")
    
    def get_statistics(self) -> Dict:
        """Get monitoring statistics"""
        with self._stats_lock:
            stats = dict(self._stats)
        
        # Calculate uptime
        if stats['start_time']:
            stats['uptime'] = (datetime.now() - stats['start_time']).total_seconds()
        
        return stats
    
    def get_parameter_history(self, parameter: str, limit: int = 100) -> List[Dict]:
        """Get parameter history"""
        if parameter in self.parameter_history:
            return self.parameter_history[parameter].get_latest(limit)
        return []
    
    def reset_alerts(self):
        """Reset alert cooldowns"""
        self.send_command({'type': 'reset_alerts'})
    
    def _reset_alerts(self):
        """Internal alert reset"""
        self._last_alert_times.clear()
        self.logger.info("Alert cooldowns reset")
    
    @contextmanager
    def monitoring_context(self):
        """Context manager for monitoring"""
        self.start_monitoring()
        try:
            yield self
        finally:
            self.stop_monitoring()

# Example usage
def demo_thread_safe_monitor():
    """Demonstrate thread-safe monitoring"""
    monitor = ThreadSafeRealtimeSafetyMonitor(check_interval=0.5)
    
    # Register alert callback
    def alert_handler(event: SafetyEvent):
        print(f"ALERT: {event.level.value} - {event.message}")
    
    monitor.register_alert_callback(alert_handler)
    
    # Use context manager
    with monitor.monitoring_context():
        print("Monitoring started...")
        
        # Simulate some time
        time.sleep(5)
        
        # Get current status
        print(f"Current safety level: {monitor.current_safety_level.value}")
        print(f"Current readings: {monitor.current_readings}")
        
        # Get statistics
        stats = monitor.get_statistics()
        print(f"Statistics: {stats}")
    
    print("Monitoring stopped")

if __name__ == "__main__":
    demo_thread_safe_monitor()