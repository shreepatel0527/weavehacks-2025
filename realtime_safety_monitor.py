"""
Real-time safety monitoring system with continuous background processing
"""
import asyncio
import threading
import queue
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple, Optional
import numpy as np
import weave
from dataclasses import dataclass
from enum import Enum

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

class RealtimeSafetyMonitor:
    """Continuous safety monitoring with real-time alerts"""
    
    def __init__(self, config_path: Optional[Path] = None):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.event_queue = queue.Queue()
        self.alert_callbacks = []
        
        # Data buffers
        self.parameter_history = {
            'temperature': deque(maxlen=100),
            'pressure': deque(maxlen=100),
            'nitrogen': deque(maxlen=100),
            'oxygen': deque(maxlen=100)
        }
        
        # Alert state tracking
        self.active_alerts = {}
        self.alert_history = deque(maxlen=50)
        self.last_notification = {}
        self.notification_cooldown = 60  # seconds
        
        # Sensor data source
        self.sensor_data = []
        self.sensor_index = 0
        self._load_sensor_data()
        
        # Statistical tracking
        self.stats = {
            'readings_processed': 0,
            'alerts_triggered': 0,
            'start_time': None
        }
        
        # Initialize W&B
        weave.init('safety-monitoring')
    
    def _load_config(self, config_path: Optional[Path]) -> dict:
        """Load safety configuration"""
        default_config = {
            'temperature': {
                'min_safe': 15.0, 'max_safe': 35.0,
                'min_warning': 10.0, 'max_warning': 40.0,
                'min_critical': 5.0, 'max_critical': 45.0,
                'units': '°C'
            },
            'pressure': {
                'min_safe': 95.0, 'max_safe': 110.0,
                'min_warning': 90.0, 'max_warning': 115.0,
                'min_critical': 85.0, 'max_critical': 120.0,
                'units': 'kPa'
            },
            'nitrogen': {
                'min_safe': 75.0, 'max_safe': 85.0,
                'min_warning': 70.0, 'max_warning': 90.0,
                'min_critical': 65.0, 'max_critical': 95.0,
                'units': '%'
            },
            'oxygen': {
                'min_safe': 19.0, 'max_safe': 23.0,
                'min_warning': 18.0, 'max_warning': 24.0,
                'min_critical': 17.0, 'max_critical': 25.0,
                'units': '%'
            }
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    for param in default_config:
                        if param in loaded_config:
                            default_config[param].update(loaded_config[param])
            except Exception as e:
                print(f"Error loading config: {e}, using defaults")
        
        return default_config
    
    def _load_sensor_data(self):
        """Load sensor data for simulation"""
        try:
            sensor_file = Path("Prototype-1/sensor_data.json")
            if sensor_file.exists():
                with open(sensor_file, 'r') as f:
                    for line in f:
                        self.sensor_data.append(json.loads(line.strip()))
                print(f"Loaded {len(self.sensor_data)} sensor readings")
        except Exception as e:
            print(f"Could not load sensor data: {e}")
    
    @weave.op()
    def check_parameter_safety(self, parameter: str, value: float) -> Tuple[SafetyLevel, str]:
        """Check safety level for a parameter"""
        config = self.config[parameter]
        
        # Check critical levels
        if value < config['min_critical'] or value > config['max_critical']:
            return SafetyLevel.CRITICAL, f"{parameter} at critical level: {value}{config['units']}"
        
        # Check warning levels
        if value < config['min_warning'] or value > config['max_warning']:
            return SafetyLevel.WARNING, f"{parameter} approaching limits: {value}{config['units']}"
        
        # Check safe levels
        if value < config['min_safe'] or value > config['max_safe']:
            return SafetyLevel.WARNING, f"{parameter} outside safe range: {value}{config['units']}"
        
        return SafetyLevel.SAFE, f"{parameter} normal: {value}{config['units']}"
    
    @weave.op()
    def process_sensor_reading(self, reading: dict):
        """Process a single sensor reading"""
        sensor_type = reading.get('sensor_type')
        value = reading.get('value')
        timestamp = datetime.fromisoformat(reading.get('timestamp'))
        
        if sensor_type in self.parameter_history:
            # Add to history
            self.parameter_history[sensor_type].append({
                'timestamp': timestamp,
                'value': value
            })
            
            # Check safety
            level, message = self.check_parameter_safety(sensor_type, value)
            
            # Handle alerts
            if level != SafetyLevel.SAFE:
                self.handle_alert(sensor_type, value, level, message, timestamp)
            else:
                # Clear any active alerts for this parameter
                if sensor_type in self.active_alerts:
                    del self.active_alerts[sensor_type]
            
            # Update stats
            self.stats['readings_processed'] += 1
            
            # Log to W&B
            if self.stats['readings_processed'] % 10 == 0:  # Log every 10 readings
                self.log_status()
    
    def handle_alert(self, parameter: str, value: float, level: SafetyLevel, 
                    message: str, timestamp: datetime):
        """Handle safety alerts"""
        # Create event
        event = SafetyEvent(
            timestamp=timestamp,
            level=level,
            parameter=parameter,
            value=value,
            threshold=self.config[parameter]['max_safe'] if value > self.config[parameter]['max_safe'] 
                     else self.config[parameter]['min_safe'],
            message=message
        )
        
        # Add to queue
        self.event_queue.put(event)
        
        # Track active alerts
        self.active_alerts[parameter] = event
        self.alert_history.append(event)
        self.stats['alerts_triggered'] += 1
        
        # Check if notification needed
        if self.should_notify(parameter, level):
            self.send_notification(event)
        
        # Log to W&B
        weave.log({
            'safety_alert': {
                'parameter': parameter,
                'value': value,
                'level': level.value,
                'message': message,
                'timestamp': timestamp.isoformat()
            }
        })
    
    def should_notify(self, parameter: str, level: SafetyLevel) -> bool:
        """Check if notification should be sent"""
        # Always notify for critical/emergency
        if level in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY]:
            return True
        
        # Check cooldown for warnings
        if parameter in self.last_notification:
            time_since_last = datetime.now() - self.last_notification[parameter]
            if time_since_last.total_seconds() < self.notification_cooldown:
                return False
        
        return True
    
    def send_notification(self, event: SafetyEvent):
        """Send notification for safety event"""
        self.last_notification[event.parameter] = datetime.now()
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"Error in alert callback: {e}")
        
        # Default console notification
        if event.level == SafetyLevel.CRITICAL:
            print(f"\n🚨 CRITICAL ALERT: {event.message}")
        elif event.level == SafetyLevel.WARNING:
            print(f"\n⚠️  WARNING: {event.message}")
    
    def monitoring_loop(self):
        """Main monitoring loop running in background thread"""
        print("Safety monitoring started")
        self.stats['start_time'] = datetime.now()
        
        while self.is_monitoring:
            try:
                # Get next sensor reading
                if self.sensor_data and self.sensor_index < len(self.sensor_data):
                    reading = self.sensor_data[self.sensor_index]
                    self.process_sensor_reading(reading)
                    self.sensor_index += 1
                    
                    # Loop back to start if we reach the end
                    if self.sensor_index >= len(self.sensor_data):
                        self.sensor_index = 0
                else:
                    # Generate simulated data if no real data
                    reading = self.generate_simulated_reading()
                    self.process_sensor_reading(reading)
                
                # Simulate real-time delay
                time.sleep(0.1)  # 10Hz monitoring
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    def generate_simulated_reading(self) -> dict:
        """Generate simulated sensor reading"""
        import random
        
        parameters = ['temperature', 'pressure', 'nitrogen', 'oxygen']
        parameter = random.choice(parameters)
        
        # Generate value with some randomness
        if parameter == 'temperature':
            value = random.gauss(25, 5)
        elif parameter == 'pressure':
            value = random.gauss(101, 5)
        elif parameter == 'nitrogen':
            value = random.gauss(80, 5)
        else:  # oxygen
            value = random.gauss(21, 1)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'sensor_type': parameter,
            'value': value,
            'units': self.config[parameter]['units']
        }
    
    @weave.op()
    def log_status(self):
        """Log current status to W&B"""
        current_values = {}
        trends = {}
        
        for param, history in self.parameter_history.items():
            if history:
                current_values[param] = history[-1]['value']
                
                # Calculate trend
                if len(history) >= 10:
                    recent_values = [h['value'] for h in list(history)[-10:]]
                    trend = np.polyfit(range(10), recent_values, 1)[0]
                    trends[param] = 'increasing' if trend > 0.1 else 'decreasing' if trend < -0.1 else 'stable'
        
        weave.log({
            'monitoring_status': {
                'current_values': current_values,
                'trends': trends,
                'active_alerts': len(self.active_alerts),
                'readings_processed': self.stats['readings_processed'],
                'uptime_seconds': (datetime.now() - self.stats['start_time']).total_seconds() 
                                if self.stats['start_time'] else 0
            }
        })
    
    def start(self):
        """Start continuous monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self.monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            print("Real-time safety monitoring activated")
    
    def stop(self):
        """Stop monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
        print("Safety monitoring stopped")
    
    def register_alert_callback(self, callback):
        """Register callback for alerts"""
        self.alert_callbacks.append(callback)
    
    def get_current_status(self) -> dict:
        """Get current monitoring status"""
        status = {
            'is_monitoring': self.is_monitoring,
            'current_values': {},
            'active_alerts': {},
            'statistics': self.stats.copy()
        }
        
        # Get latest values
        for param, history in self.parameter_history.items():
            if history:
                latest = history[-1]
                status['current_values'][param] = {
                    'value': latest['value'],
                    'timestamp': latest['timestamp'].isoformat(),
                    'units': self.config[param]['units']
                }
        
        # Get active alerts
        for param, alert in self.active_alerts.items():
            status['active_alerts'][param] = {
                'level': alert.level.value,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat()
            }
        
        return status
    
    def get_parameter_history(self, parameter: str, duration_minutes: int = 10) -> List[dict]:
        """Get recent history for a parameter"""
        if parameter not in self.parameter_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        history = []
        
        for item in self.parameter_history[parameter]:
            if item['timestamp'] > cutoff_time:
                history.append({
                    'timestamp': item['timestamp'].isoformat(),
                    'value': item['value']
                })
        
        return history

# Example usage with alert handling
def demo_alert_handler(event: SafetyEvent):
    """Example alert handler"""
    if event.level == SafetyLevel.CRITICAL:
        print(f"🚨 TAKING EMERGENCY ACTION: Shutting down {event.parameter} system!")
        # Here you would trigger actual safety systems
    elif event.level == SafetyLevel.WARNING:
        print(f"⚠️  Adjusting {event.parameter} controls...")
        # Here you would make adjustments

if __name__ == "__main__":
    # Create monitor
    monitor = RealtimeSafetyMonitor()
    
    # Register alert handler
    monitor.register_alert_callback(demo_alert_handler)
    
    # Start monitoring
    monitor.start()
    
    try:
        # Keep running and periodically show status
        while True:
            time.sleep(5)
            status = monitor.get_current_status()
            print(f"\n📊 Status: {status['statistics']['readings_processed']} readings, "
                  f"{len(status['active_alerts'])} active alerts")
            
            if status['current_values']:
                print("Current values:")
                for param, data in status['current_values'].items():
                    print(f"  {param}: {data['value']:.1f} {data['units']}")
    
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        monitor.stop()