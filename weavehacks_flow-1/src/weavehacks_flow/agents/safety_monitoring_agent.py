#!/usr/bin/env python3
"""
Enhanced Safety Monitoring Agent for Lab Automation
Re-implemented using advanced sensor data collection and experiment-specific safety protocols.
Based on Prototype-1 code with integrated experiment management and real-time monitoring.
"""

import sys
import os
import threading
import time
import logging
import json
import datetime
import queue
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import random
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('safety_monitoring_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import components from Prototype-1 if available
try:
    # Try to import from current directory first
    from sensor_data_module import SensorDataCollector, SensorReading, SensorType, SensorConfig
    from experiment_config import ExperimentManager, ExperimentConfig
    from advanced_safety_agent import SafetyMonitoringAgent, SafetyAlert, ParameterType, SafetyThreshold
    PROTOTYPE1_AVAILABLE = True
    logger.info("Using Prototype-1 components")
except ImportError:
    # Fallback to local implementations
    PROTOTYPE1_AVAILABLE = False
    logger.warning("Prototype-1 components not available, using local implementations")

# Local implementations for when Prototype-1 components are not available
if not PROTOTYPE1_AVAILABLE:
    class SensorType(Enum):
        TEMPERATURE = "temperature"
        PRESSURE = "pressure"
        PH = "ph"
        HUMIDITY = "humidity"

    class ParameterType(Enum):
        TEMPERATURE = "temperature"
        PRESSURE = "pressure"
        NITROGEN = "nitrogen"
        OXYGEN = "oxygen"
        BUTANE = "butane"
        PH = "ph"

    class AlertLevel(Enum):
        NORMAL = "normal"
        WARNING = "warning"
        CRITICAL = "critical"
        EMERGENCY = "emergency"

    @dataclass
    class SensorReading:
        timestamp: datetime
        sensor_type: SensorType
        value: float
        units: str
        sensor_id: str = "default"
        location: str = "lab"

    @dataclass
    class SafetyAlert:
        timestamp: datetime
        level: AlertLevel
        parameter: ParameterType
        current_value: float
        threshold_violated: str
        message: str
        sensor_id: str
        requires_action: bool = False

    @dataclass
    class SafetyThreshold:
        parameter: ParameterType
        min_safe: float
        max_safe: float
        warning_buffer: float = 0.1
        critical_buffer: float = 0.2
        units: str = ""

class EnhancedSafetyMonitoringAgent:
    """
    Enhanced Safety Monitoring Agent with experiment-specific protocols,
    real-time sensor integration, and advanced alert escalation.
    """
    
    def __init__(self, config_file: str = "enhanced_safety_config.json"):
        self.config_file = config_file
        self.monitoring = False
        self.alert_history: List[SafetyAlert] = []
        self.active_alerts: Dict[str, SafetyAlert] = {}
        self.alert_callbacks: List[Callable] = []
        self.shutdown_callbacks: List[Callable] = []
        
        # Experiment and sensor integration
        self.current_experiment = None
        self.experiment_thresholds = {}
        self.sensor_queue = queue.Queue()
        
        # Timing configurations
        self.alert_persistence_minutes = 1.0
        self.scientist_response_timeout_minutes = 3.0
        self.monitoring_interval_seconds = 2.0
        
        # Alert tracking
        self.scientist_contacted = {}
        self.last_alert_times = {}
        
        # Initialize with default experiment configurations
        self._initialize_experiment_configs()
        
        logger.info("Enhanced Safety Monitoring Agent initialized")
    
    def _initialize_experiment_configs(self):
        """Initialize experiment-specific safety configurations."""
        self.experiment_configs = {
            "gold_nanoparticle_room_temp": {
                "name": "Gold Nanoparticle Synthesis (Room Temp)",
                "temperature": {
                    "min_safe": 20.0, "max_safe": 25.0,
                    "warning_buffer": 2.0, "critical_buffer": 5.0,
                    "units": "°C"
                },
                "pressure": {
                    "min_safe": 100.0, "max_safe": 102.0,
                    "warning_buffer": 1.0, "critical_buffer": 3.0,
                    "units": "kPa"
                },
                "duration_hours": 1.5,
                "special_notes": "Monitor color change: deep red → faint yellow → clear"
            },
            "gold_nanoparticle_ice_bath": {
                "name": "Gold Nanoparticle Synthesis (Ice Bath)",
                "temperature": {
                    "min_safe": 0.0, "max_safe": 5.0,
                    "warning_buffer": 1.0, "critical_buffer": 3.0,
                    "units": "°C"
                },
                "pressure": {
                    "min_safe": 100.0, "max_safe": 102.0,
                    "warning_buffer": 1.0, "critical_buffer": 3.0,
                    "units": "kPa"
                },
                "duration_hours": 1.0,
                "special_notes": "Cool to 0°C in ice bath over 30 min with stirring"
            },
            "gold_nanoparticle_stirring": {
                "name": "Gold Nanoparticle Synthesis (Vigorous Stirring)",
                "temperature": {
                    "min_safe": 25.0, "max_safe": 35.0,
                    "warning_buffer": 3.0, "critical_buffer": 8.0,
                    "units": "°C"
                },
                "pressure": {
                    "min_safe": 100.0, "max_safe": 103.0,
                    "warning_buffer": 1.5, "critical_buffer": 4.0,
                    "units": "kPa"
                },
                "duration_hours": 0.25,
                "special_notes": "Stir vigorously (~1100 rpm) for ~15 min"
            },
            "overnight_stirring": {
                "name": "Overnight Stirring Under N₂",
                "temperature": {
                    "min_safe": 18.0, "max_safe": 28.0,
                    "warning_buffer": 2.0, "critical_buffer": 5.0,
                    "units": "°C"
                },
                "pressure": {
                    "min_safe": 99.0, "max_safe": 103.0,
                    "warning_buffer": 2.0, "critical_buffer": 5.0,
                    "units": "kPa"
                },
                "duration_hours": 12.0,
                "special_notes": "Stir overnight under N₂ atmosphere - REQUIRES SAFETY MONITORING"
            }
        }
        
        logger.info(f"Initialized {len(self.experiment_configs)} experiment configurations")
    
    def set_experiment(self, experiment_id: str) -> bool:
        """Set the current experiment and update safety thresholds."""
        if experiment_id in self.experiment_configs:
            self.current_experiment = experiment_id
            self._update_safety_thresholds()
            logger.info(f"Set current experiment to: {self.experiment_configs[experiment_id]['name']}")
            return True
        else:
            logger.error(f"Unknown experiment ID: {experiment_id}")
            return False
    
    def _update_safety_thresholds(self):
        """Update safety thresholds based on current experiment."""
        if not self.current_experiment:
            return
        
        config = self.experiment_configs[self.current_experiment]
        
        # Update temperature thresholds
        if PROTOTYPE1_AVAILABLE:
            temp_threshold = SafetyThreshold(
                parameter=ParameterType.TEMPERATURE,
                min_safe=config["temperature"]["min_safe"],
                max_safe=config["temperature"]["max_safe"],
                warning_buffer=config["temperature"]["warning_buffer"],
                critical_buffer=config["temperature"]["critical_buffer"],
                units=config["temperature"]["units"]
            )
            
            pressure_threshold = SafetyThreshold(
                parameter=ParameterType.PRESSURE,
                min_safe=config["pressure"]["min_safe"],
                max_safe=config["pressure"]["max_safe"],
                warning_buffer=config["pressure"]["warning_buffer"],
                critical_buffer=config["pressure"]["critical_buffer"],
                units=config["pressure"]["units"]
            )
        else:
            # Use local implementation
            temp_threshold = SafetyThreshold(
                parameter=ParameterType.TEMPERATURE,
                min_safe=config["temperature"]["min_safe"],
                max_safe=config["temperature"]["max_safe"],
                warning_buffer=config["temperature"]["warning_buffer"],
                critical_buffer=config["temperature"]["critical_buffer"],
                units=config["temperature"]["units"]
            )
            
            pressure_threshold = SafetyThreshold(
                parameter=ParameterType.PRESSURE,
                min_safe=config["pressure"]["min_safe"],
                max_safe=config["pressure"]["max_safe"],
                warning_buffer=config["pressure"]["warning_buffer"],
                critical_buffer=config["pressure"]["critical_buffer"],
                units=config["pressure"]["units"]
            )
        
        self.experiment_thresholds = {
            ParameterType.TEMPERATURE: temp_threshold,
            ParameterType.PRESSURE: pressure_threshold
        }
        
        logger.info(f"Updated safety thresholds for experiment: {config['name']}")
        logger.info(f"Temperature: {temp_threshold.min_safe}-{temp_threshold.max_safe}{temp_threshold.units}")
        logger.info(f"Pressure: {pressure_threshold.min_safe}-{pressure_threshold.max_safe}{pressure_threshold.units}")
    
    def analyze_sensor_reading(self, reading: SensorReading) -> Optional[SafetyAlert]:
        """Analyze a sensor reading against current experiment thresholds."""
        if not self.current_experiment or not self.experiment_thresholds:
            logger.warning("No experiment set or thresholds configured")
            return None
        
        # Map sensor type to parameter type
        parameter_map = {
            SensorType.TEMPERATURE: ParameterType.TEMPERATURE,
            SensorType.PRESSURE: ParameterType.PRESSURE
        }
        
        parameter_type = parameter_map.get(reading.sensor_type)
        if not parameter_type or parameter_type not in self.experiment_thresholds:
            return None
        
        threshold = self.experiment_thresholds[parameter_type]
        
        # Determine alert level
        alert_level = AlertLevel.NORMAL
        message = ""
        requires_action = False
        threshold_violated = ""
        
        # Check for violations
        if reading.value < (threshold.min_safe - threshold.critical_buffer):
            alert_level = AlertLevel.EMERGENCY
            threshold_violated = f"min_critical ({threshold.min_safe - threshold.critical_buffer})"
            message = f"EMERGENCY: {reading.sensor_type.value} critically low at {reading.value}{threshold.units}"
            requires_action = True
        elif reading.value > (threshold.max_safe + threshold.critical_buffer):
            alert_level = AlertLevel.EMERGENCY
            threshold_violated = f"max_critical ({threshold.max_safe + threshold.critical_buffer})"
            message = f"EMERGENCY: {reading.sensor_type.value} critically high at {reading.value}{threshold.units}"
            requires_action = True
        elif reading.value < (threshold.min_safe - threshold.warning_buffer):
            alert_level = AlertLevel.CRITICAL
            threshold_violated = f"min_warning ({threshold.min_safe - threshold.warning_buffer})"
            message = f"CRITICAL: {reading.sensor_type.value} below safe range at {reading.value}{threshold.units}"
            requires_action = True
        elif reading.value > (threshold.max_safe + threshold.warning_buffer):
            alert_level = AlertLevel.CRITICAL
            threshold_violated = f"max_warning ({threshold.max_safe + threshold.warning_buffer})"
            message = f"CRITICAL: {reading.sensor_type.value} above safe range at {reading.value}{threshold.units}"
            requires_action = True
        elif reading.value < threshold.min_safe or reading.value > threshold.max_safe:
            alert_level = AlertLevel.WARNING
            threshold_violated = f"safe_range ({threshold.min_safe}-{threshold.max_safe})"
            message = f"WARNING: {reading.sensor_type.value} outside optimal range at {reading.value}{threshold.units}"
        
        if alert_level != AlertLevel.NORMAL:
            return SafetyAlert(
                timestamp=reading.timestamp,
                level=alert_level,
                parameter=parameter_type,
                current_value=reading.value,
                threshold_violated=threshold_violated,
                message=message,
                sensor_id=reading.sensor_id,
                requires_action=requires_action
            )
        
        return None
    
    def process_sensor_reading(self, reading: SensorReading):
        """Process a sensor reading and generate alerts if necessary."""
        if not self.monitoring:
            return
        
        alert = self.analyze_sensor_reading(reading)
        if alert:
            self.process_alert(alert)
    
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
        current_time = datetime.now()
        
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
            "experiment": self.current_experiment,
            "requires_immediate_attention": alert.requires_action
        }
        
        # In a real implementation, this would send SMS, email, push notification, etc.
        logger.info(f"SCIENTIST NOTIFICATION: {json.dumps(notification_message, indent=2)}")
    
    def _initiate_emergency_shutdown(self, alert: SafetyAlert):
        """Initiate emergency shutdown procedures."""
        shutdown_command = {
            "type": "emergency_shutdown",
            "timestamp": datetime.now().isoformat(),
            "trigger_alert": asdict(alert),
            "experiment": self.current_experiment,
            "reason": f"No scientist response within {self.scientist_response_timeout_minutes} minutes"
        }
        
        logger.critical(f"EMERGENCY SHUTDOWN: {json.dumps(shutdown_command, indent=2)}")
        
        # Execute shutdown callbacks
        for callback in self.shutdown_callbacks:
            try:
                callback(shutdown_command)
            except Exception as e:
                logger.error(f"Shutdown callback failed: {e}")
    
    def simulate_sensor_data(self):
        """Simulate realistic sensor data based on current experiment."""
        if not self.current_experiment:
            return
        
        config = self.experiment_configs[self.current_experiment]
        current_time = datetime.now()
        
        # Generate temperature reading
        temp_config = config["temperature"]
        temp_center = (temp_config["min_safe"] + temp_config["max_safe"]) / 2
        temp_noise = random.uniform(-2.0, 2.0)
        temp_value = temp_center + temp_noise
        
        # Occasionally generate out-of-range readings for testing
        if random.random() < 0.1:  # 10% chance
            if random.random() < 0.5:
                temp_value = temp_config["min_safe"] - random.uniform(1.0, 5.0)
            else:
                temp_value = temp_config["max_safe"] + random.uniform(1.0, 5.0)
        
        temp_reading = SensorReading(
            timestamp=current_time,
            sensor_type=SensorType.TEMPERATURE,
            value=round(temp_value, 2),
            units=temp_config["units"],
            sensor_id="temp_001"
        )
        
        # Generate pressure reading
        pressure_config = config["pressure"]
        pressure_center = (pressure_config["min_safe"] + pressure_config["max_safe"]) / 2
        pressure_noise = random.uniform(-1.0, 1.0)
        pressure_value = pressure_center + pressure_noise
        
        # Occasionally generate out-of-range readings for testing
        if random.random() < 0.08:  # 8% chance
            if random.random() < 0.5:
                pressure_value = pressure_config["min_safe"] - random.uniform(1.0, 3.0)
            else:
                pressure_value = pressure_config["max_safe"] + random.uniform(1.0, 3.0)
        
        pressure_reading = SensorReading(
            timestamp=current_time,
            sensor_type=SensorType.PRESSURE,
            value=round(pressure_value, 2),
            units=pressure_config["units"],
            sensor_id="press_001"
        )
        
        # Process readings
        self.process_sensor_reading(temp_reading)
        self.process_sensor_reading(pressure_reading)
        
        return [temp_reading, pressure_reading]
    
    def start_monitoring(self):
        """Start the safety monitoring system."""
        if self.monitoring:
            logger.warning("Safety monitoring already active")
            return
        
        if not self.current_experiment:
            logger.error("Cannot start monitoring without setting an experiment")
            return
        
        self.monitoring = True
        
        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        monitor_thread.start()
        
        logger.info("Enhanced safety monitoring started")
        return monitor_thread
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Safety monitoring loop started")
        
        while self.monitoring:
            try:
                # Simulate sensor data (replace with actual sensor integration)
                readings = self.simulate_sensor_data()
                
                if readings:
                    for reading in readings:
                        logger.debug(f"Processed reading: {reading.sensor_type.value} = {reading.value}{reading.units}")
                
                time.sleep(self.monitoring_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    def stop_monitoring(self):
        """Stop the safety monitoring system."""
        self.monitoring = False
        logger.info("Enhanced safety monitoring stopped")
    
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
            "current_experiment": self.current_experiment,
            "experiment_name": self.experiment_configs.get(self.current_experiment, {}).get("name", "None") if self.current_experiment else "None",
            "active_alerts": {k: asdict(v) for k, v in self.active_alerts.items()},
            "total_alerts_today": len([a for a in self.alert_history 
                                     if a.timestamp.date() == datetime.now().date()]),
            "scientist_contacts_pending": len(self.scientist_contacted),
            "available_experiments": list(self.experiment_configs.keys())
        }
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict]:
        """Get recent safety alerts."""
        recent_alerts = self.alert_history[-count:] if self.alert_history else []
        return [
            {
                "timestamp": alert.timestamp.isoformat(),
                "level": alert.level.value,
                "parameter": alert.parameter.value,
                "value": alert.current_value,
                "message": alert.message,
                "sensor_id": alert.sensor_id
            }
            for alert in recent_alerts
        ]

def main():
    """Main function to run the enhanced safety monitoring agent."""
    agent = EnhancedSafetyMonitoringAgent()
    
    # Add example callbacks
    def alert_webhook(alert: SafetyAlert):
        print(f"🚨 ALERT WEBHOOK: {alert.message}")
    
    def shutdown_controller(shutdown_info: Dict):
        print(f"🛑 SHUTDOWN COMMAND: Initiating emergency protocols")
        # In real implementation, this would interface with lab equipment
    
    agent.add_alert_callback(alert_webhook)
    agent.add_shutdown_callback(shutdown_controller)
    
    # Demo different experiments
    experiments = [
        "gold_nanoparticle_room_temp",
        "gold_nanoparticle_ice_bath", 
        "gold_nanoparticle_stirring",
        "overnight_stirring"
    ]
    
    try:
        print("🔬 Enhanced Safety Monitoring Agent Demo")
        print("=" * 50)
        
        for i, exp_id in enumerate(experiments):
            print(f"\n🧪 Demo {i+1}: {agent.experiment_configs[exp_id]['name']}")
            print("-" * 40)
            
            # Set experiment
            agent.set_experiment(exp_id)
            
            # Start monitoring
            monitor_thread = agent.start_monitoring()
            
            print(f"📊 Monitoring for 30 seconds...")
            print(f"⚠️  Alert levels: WARNING → CRITICAL → EMERGENCY")
            print(f"🚨 Auto-shutdown after {agent.scientist_response_timeout_minutes}min scientist non-response")
            
            # Monitor for 30 seconds
            time.sleep(30)
            
            # Stop monitoring
            agent.stop_monitoring()
            
            # Print status
            status = agent.get_status_report()
            alerts = agent.get_recent_alerts(5)
            
            print(f"\n📋 Experiment Summary:")
            print(f"   Total alerts: {len(alerts)}")
            print(f"   Active alerts: {len(status['active_alerts'])}")
            print(f"   Contacts pending: {status['scientist_contacts_pending']}")
            
            if alerts:
                print(f"   Last alert: {alerts[-1]['message']}")
            
            print()
            
            if i < len(experiments) - 1:
                print("⏸️  Press Ctrl+C to stop demo or wait for next experiment...")
                time.sleep(5)
        
        print("✅ Demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo stopped by user")
        agent.stop_monitoring()
        
        # Print final status
        final_status = agent.get_status_report()
        print(f"\n📋 Final Status Report:")
        print(json.dumps(final_status, indent=2, default=str))

if __name__ == "__main__":
    main()