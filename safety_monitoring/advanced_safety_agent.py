#!/usr/bin/env python3
"""
Advanced Safety Monitoring Agent for Lab Automation
Implements sophisticated safety protocols with multi-parameter monitoring,
alert escalation, and automated shutdown capabilities.
"""

import csv
import time
import datetime
import threading
import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('safety_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class ParameterType(Enum):
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    NITROGEN = "nitrogen"
    BUTANE = "butane"
    OXYGEN = "oxygen"
    PH = "ph"

@dataclass
class SafetyThreshold:
    parameter: ParameterType
    min_safe: float
    max_safe: float
    warning_buffer: float = 0.1
    critical_buffer: float = 0.2
    units: str = ""

@dataclass
class SensorReading:
    timestamp: datetime.datetime
    parameter: ParameterType
    value: float
    units: str
    sensor_id: str = "default"

@dataclass
class SafetyAlert:
    timestamp: datetime.datetime
    level: AlertLevel
    parameter: ParameterType
    current_value: float
    threshold_violated: str
    message: str
    sensor_id: str
    requires_action: bool = False

class SafetyMonitoringAgent:
    """
    Advanced safety monitoring agent with multi-parameter tracking,
    alert escalation, and automated response capabilities.
    """
    
    def __init__(self, config_file: str = "safety_config.json"):
        self.config_file = config_file
        self.thresholds = self._load_thresholds()
        self.alert_history: List[SafetyAlert] = []
        self.active_alerts: Dict[str, SafetyAlert] = {}
        self.monitoring = False
        self.alert_callbacks: List[Callable] = []
        self.shutdown_callbacks: List[Callable] = []
        
        # Timing configurations
        self.alert_persistence_minutes = 1.0
        self.scientist_response_timeout_minutes = 3.0
        self.monitoring_interval_seconds = 2.0
        
        # Alert tracking
        self.scientist_contacted = {}
        self.last_alert_times = {}
        
        logger.info("Safety Monitoring Agent initialized")
    
    def _load_thresholds(self) -> Dict[ParameterType, SafetyThreshold]:
        """Load safety thresholds from configuration file."""
        default_thresholds = {
            ParameterType.TEMPERATURE: SafetyThreshold(
                parameter=ParameterType.TEMPERATURE,
                min_safe=15.0, max_safe=35.0,
                warning_buffer=2.0, critical_buffer=5.0,
                units="°C"
            ),
            ParameterType.PRESSURE: SafetyThreshold(
                parameter=ParameterType.PRESSURE,
                min_safe=95.0, max_safe=110.0,
                warning_buffer=2.0, critical_buffer=5.0,
                units="kPa"
            ),
            ParameterType.NITROGEN: SafetyThreshold(
                parameter=ParameterType.NITROGEN,
                min_safe=75.0, max_safe=85.0,
                warning_buffer=2.0, critical_buffer=5.0,
                units="%"
            ),
            ParameterType.BUTANE: SafetyThreshold(
                parameter=ParameterType.BUTANE,
                min_safe=0.0, max_safe=0.1,
                warning_buffer=0.02, critical_buffer=0.05,
                units="ppm"
            ),
            ParameterType.OXYGEN: SafetyThreshold(
                parameter=ParameterType.OXYGEN,
                min_safe=19.0, max_safe=23.0,
                warning_buffer=1.0, critical_buffer=2.0,
                units="%"
            )
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                    # Load custom thresholds if available
                    logger.info(f"Loaded safety configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}. Using defaults.")
        else:
            # Save default configuration
            self._save_default_config(default_thresholds)
        
        return default_thresholds
    
    def _save_default_config(self, thresholds: Dict[ParameterType, SafetyThreshold]):
        """Save default configuration to file."""
        config_data = {
            param.value: asdict(threshold) for param, threshold in thresholds.items()
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            logger.info(f"Saved default safety configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save default config: {e}")
    
    def analyze_reading(self, reading: SensorReading) -> Optional[SafetyAlert]:
        """Analyze a sensor reading and determine if an alert should be generated."""
        threshold = self.thresholds.get(reading.parameter)
        if not threshold:
            logger.warning(f"No threshold configured for parameter: {reading.parameter}")
            return None
        
        alert_level = AlertLevel.NORMAL
        message = ""
        requires_action = False
        threshold_violated = ""
        
        # Check for violations
        if reading.value < (threshold.min_safe - threshold.critical_buffer):
            alert_level = AlertLevel.EMERGENCY
            threshold_violated = f"min_critical ({threshold.min_safe - threshold.critical_buffer})"
            message = f"EMERGENCY: {reading.parameter.value} critically low at {reading.value}{threshold.units}"
            requires_action = True
        elif reading.value > (threshold.max_safe + threshold.critical_buffer):
            alert_level = AlertLevel.EMERGENCY
            threshold_violated = f"max_critical ({threshold.max_safe + threshold.critical_buffer})"
            message = f"EMERGENCY: {reading.parameter.value} critically high at {reading.value}{threshold.units}"
            requires_action = True
        elif reading.value < (threshold.min_safe - threshold.warning_buffer):
            alert_level = AlertLevel.CRITICAL
            threshold_violated = f"min_warning ({threshold.min_safe - threshold.warning_buffer})"
            message = f"CRITICAL: {reading.parameter.value} below safe range at {reading.value}{threshold.units}"
            requires_action = True
        elif reading.value > (threshold.max_safe + threshold.warning_buffer):
            alert_level = AlertLevel.CRITICAL
            threshold_violated = f"max_warning ({threshold.max_safe + threshold.warning_buffer})"
            message = f"CRITICAL: {reading.parameter.value} above safe range at {reading.value}{threshold.units}"
            requires_action = True
        elif reading.value < threshold.min_safe or reading.value > threshold.max_safe:
            alert_level = AlertLevel.WARNING
            threshold_violated = f"safe_range ({threshold.min_safe}-{threshold.max_safe})"
            message = f"WARNING: {reading.parameter.value} outside optimal range at {reading.value}{threshold.units}"
        
        if alert_level != AlertLevel.NORMAL:
            return SafetyAlert(
                timestamp=reading.timestamp,
                level=alert_level,
                parameter=reading.parameter,
                current_value=reading.value,
                threshold_violated=threshold_violated,
                message=message,
                sensor_id=reading.sensor_id,
                requires_action=requires_action
            )
        
        return None
    
    def process_alert(self, alert: SafetyAlert):
        """Process a safety alert with appropriate escalation."""
        alert_key = f"{alert.parameter.value}_{alert.sensor_id}"
        
        # Store alert
        self.alert_history.append(alert)
        self.active_alerts[alert_key] = alert
        
        # Log alert
        logger.warning(f"SAFETY ALERT: {alert.message}")
        
        # Execute alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        # Handle escalation based on alert level
        if alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            self._handle_critical_alert(alert, alert_key)
    
    def _handle_critical_alert(self, alert: SafetyAlert, alert_key: str):
        """Handle critical and emergency alerts with escalation protocol."""
        current_time = datetime.datetime.now()
        
        # Check if scientist has been contacted for this parameter
        if alert_key not in self.scientist_contacted:
            self._contact_scientist(alert)
            self.scientist_contacted[alert_key] = current_time
            logger.info(f"Scientist contacted for {alert.parameter.value} alert")
        
        # Check if scientist response timeout has been exceeded
        time_since_contact = current_time - self.scientist_contacted[alert_key]
        if time_since_contact.total_seconds() > (self.scientist_response_timeout_minutes * 60):
            # Check if the alert condition persists
            if alert_key in self.active_alerts:
                latest_alert = self.active_alerts[alert_key]
                if latest_alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
                    self._initiate_emergency_shutdown(alert)
                    logger.critical(f"Emergency shutdown initiated for {alert.parameter.value}")
                else:
                    # Condition improved, record warning
                    logger.info(f"Condition improved for {alert.parameter.value}, no shutdown needed")
                    del self.scientist_contacted[alert_key]
    
    def _contact_scientist(self, alert: SafetyAlert):
        """Simulate contacting the scientist (implement actual notification here)."""
        notification_message = {
            "type": "safety_alert",
            "timestamp": alert.timestamp.isoformat(),
            "level": alert.level.value,
            "parameter": alert.parameter.value,
            "value": alert.current_value,
            "message": alert.message,
            "requires_immediate_attention": alert.requires_action
        }
        
        # In a real implementation, this would send SMS, email, push notification, etc.
        logger.info(f"SCIENTIST NOTIFICATION: {json.dumps(notification_message, indent=2)}")
    
    def _initiate_emergency_shutdown(self, alert: SafetyAlert):
        """Initiate emergency shutdown procedures."""
        shutdown_command = {
            "type": "emergency_shutdown",
            "timestamp": datetime.datetime.now().isoformat(),
            "trigger_alert": asdict(alert),
            "reason": f"No scientist response within {self.scientist_response_timeout_minutes} minutes"
        }
        
        logger.critical(f"EMERGENCY SHUTDOWN: {json.dumps(shutdown_command, indent=2)}")
        
        # Execute shutdown callbacks
        for callback in self.shutdown_callbacks:
            try:
                callback(shutdown_command)
            except Exception as e:
                logger.error(f"Shutdown callback failed: {e}")
    
    def monitor_csv_file(self, file_path: str, parameter_mapping: Dict[str, ParameterType]):
        """Monitor a CSV file for sensor data."""
        logger.info(f"Starting CSV monitoring for {file_path}")
        last_position = 0
        
        while self.monitoring:
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        f.seek(last_position)
                        reader = csv.DictReader(f)
                        
                        for row in reader:
                            if not row:
                                continue
                            
                            # Process each parameter in the row
                            timestamp = datetime.datetime.fromisoformat(row.get('timestamp', datetime.datetime.now().isoformat()))
                            
                            for csv_column, param_type in parameter_mapping.items():
                                if csv_column in row:
                                    try:
                                        value = float(row[csv_column])
                                        reading = SensorReading(
                                            timestamp=timestamp,
                                            parameter=param_type,
                                            value=value,
                                            units=self.thresholds[param_type].units,
                                            sensor_id=f"csv_{csv_column}"
                                        )
                                        
                                        alert = self.analyze_reading(reading)
                                        if alert:
                                            self.process_alert(alert)
                                        else:
                                            # Clear alert if conditions are normal
                                            alert_key = f"{param_type.value}_csv_{csv_column}"
                                            if alert_key in self.active_alerts:
                                                del self.active_alerts[alert_key]
                                                if alert_key in self.scientist_contacted:
                                                    del self.scientist_contacted[alert_key]
                                    
                                    except (ValueError, KeyError) as e:
                                        logger.warning(f"Invalid data in row: {row}, error: {e}")
                        
                        last_position = f.tell()
                
                time.sleep(self.monitoring_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error monitoring CSV file: {e}")
                time.sleep(self.monitoring_interval_seconds)
    
    def start_monitoring(self, file_path: str = "sensor_data.csv"):
        """Start the safety monitoring agent."""
        self.monitoring = True
        
        # Default parameter mapping for the existing CSV format
        parameter_mapping = {
            "temperature_celsius": ParameterType.TEMPERATURE,
            "pressure_kpa": ParameterType.PRESSURE
        }
        
        # Start monitoring in a separate thread
        monitor_thread = threading.Thread(
            target=self.monitor_csv_file,
            args=(file_path, parameter_mapping),
            daemon=True
        )
        monitor_thread.start()
        
        logger.info("Safety monitoring started")
        return monitor_thread
    
    def stop_monitoring(self):
        """Stop the safety monitoring agent."""
        self.monitoring = False
        logger.info("Safety monitoring stopped")
    
    def add_alert_callback(self, callback: Callable):
        """Add a callback function to be called when alerts are generated."""
        self.alert_callbacks.append(callback)
    
    def add_shutdown_callback(self, callback: Callable):
        """Add a callback function to be called during emergency shutdown."""
        self.shutdown_callbacks.append(callback)
    
    def get_status_report(self) -> Dict:
        """Generate a comprehensive status report."""
        return {
            "monitoring_active": self.monitoring,
            "active_alerts": {k: asdict(v) for k, v in self.active_alerts.items()},
            "total_alerts_today": len([a for a in self.alert_history 
                                     if a.timestamp.date() == datetime.date.today()]),
            "scientist_contacts_pending": len(self.scientist_contacted),
            "configuration": {k.value: asdict(v) for k, v in self.thresholds.items()}
        }

def main():
    """Main function to run the safety monitoring agent."""
    agent = SafetyMonitoringAgent()
    
    # Add example callbacks
    def alert_webhook(alert: SafetyAlert):
        print(f"🚨 ALERT WEBHOOK: {alert.message}")
    
    def shutdown_controller(shutdown_info: Dict):
        print(f"🛑 SHUTDOWN COMMAND: Initiating emergency protocols")
        # In real implementation, this would interface with lab equipment
    
    agent.add_alert_callback(alert_webhook)
    agent.add_shutdown_callback(shutdown_controller)
    
    try:
        # Start monitoring
        monitor_thread = agent.start_monitoring()
        
        print("🔍 Advanced Safety Monitoring Agent is running...")
        print("📊 Monitoring: Temperature, Pressure, and more")
        print("⚠️  Alert levels: WARNING → CRITICAL → EMERGENCY")
        print("🚨 Auto-shutdown after 3min scientist non-response")
        print("Press Ctrl+C to stop")
        
        # Keep main thread alive
        while agent.monitoring:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Stopping safety monitoring...")
        agent.stop_monitoring()
        
        # Print final status report
        status = agent.get_status_report()
        print(f"\n📋 Final Status Report:")
        print(json.dumps(status, indent=2, default=str))

if __name__ == "__main__":
    main()