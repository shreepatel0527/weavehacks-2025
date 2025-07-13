#!/usr/bin/env python3
"""
Safety Integration Module for Prototype-1
Integrates sensor data collection with advanced safety monitoring.
"""

import sys
import os
import threading
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional

from sensor_data_module import SensorDataCollector, SensorReading, SensorType
from experiment_config import ExperimentManager, ExperimentConfig
from advanced_safety_agent import SafetyMonitoringAgent, SafetyAlert, ParameterType, SafetyThreshold

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegratedLabSafetySystem:
    """
    Integrated lab safety system that combines real-time sensor data collection
    with advanced safety monitoring and alerting.
    """
    
    def __init__(self):
        self.sensor_collector = SensorDataCollector()
        self.safety_agent = SafetyMonitoringAgent()
        self.experiment_manager = ExperimentManager()
        self.running = False
        
        # Mapping between sensor types and safety parameters
        self.sensor_to_parameter_map = {
            SensorType.TEMPERATURE: ParameterType.TEMPERATURE,
            SensorType.PRESSURE: ParameterType.PRESSURE
        }
        
        # Setup integration
        self._setup_integration()
        
        logger.info("Integrated Lab Safety System initialized")
    
    def _setup_integration(self):
        """Setup integration between sensor collector and safety agent."""
        # Add sensor data callback to feed safety monitoring
        self.sensor_collector.add_callback(self._process_sensor_for_safety)
        
        # Add safety alert callbacks
        self.safety_agent.add_alert_callback(self._handle_safety_alert)
        self.safety_agent.add_shutdown_callback(self._handle_emergency_shutdown)
    
    def _process_sensor_for_safety(self, reading: SensorReading):
        """Process sensor reading for safety analysis."""
        # Convert sensor reading to safety monitoring format
        parameter_type = self.sensor_to_parameter_map.get(reading.sensor_type)
        
        if parameter_type:
            # Create safety sensor reading
            from advanced_safety_agent import SensorReading as SafetySensorReading
            
            safety_reading = SafetySensorReading(
                timestamp=reading.timestamp,
                parameter=parameter_type,
                value=reading.value,
                units=reading.units,
                sensor_id=reading.sensor_id
            )
            
            # Analyze for safety concerns
            alert = self.safety_agent.analyze_reading(safety_reading)
            if alert:
                self.safety_agent.process_alert(alert)
    
    def _handle_safety_alert(self, alert: SafetyAlert):
        """Handle safety alerts with enhanced logging and notifications."""
        alert_msg = (f"🚨 SAFETY ALERT: {alert.level.value.upper()} - "
                    f"{alert.parameter.value} = {alert.current_value}{self.safety_agent.thresholds[alert.parameter].units} "
                    f"(Threshold: {alert.threshold_violated})")
        
        print(alert_msg)
        logger.warning(alert_msg)
        
        # Additional alert handling logic can be added here
        # e.g., send notifications, update UI, etc.
    
    def _handle_emergency_shutdown(self, shutdown_info: Dict):
        """Handle emergency shutdown procedures."""
        shutdown_msg = f"🛑 EMERGENCY SHUTDOWN INITIATED: {shutdown_info['reason']}"
        print(shutdown_msg)
        logger.critical(shutdown_msg)
        
        # In a real lab environment, this would:
        # - Shut down equipment
        # - Alert all personnel
        # - Log to emergency systems
        # - Contact emergency services if needed
    
    def start_system(self):
        """Start the integrated lab safety system."""
        if self.running:
            logger.warning("System already running")
            return
        
        self.running = True
        
        print("🔬 Starting Integrated Lab Safety System...")
        print("📊 Sensor Collection: Temperature, Pressure")
        print("🛡️  Safety Monitoring: Multi-parameter threshold analysis")
        print("🚨 Alert System: WARNING → CRITICAL → EMERGENCY escalation")
        print("⏰ Auto-shutdown: 3-minute scientist response timeout")
        
        # Start sensor data collection
        self.sensor_collector.start_collection()
        
        # Start safety monitoring (it will monitor sensor data via callbacks)
        self.safety_agent.monitoring = True
        
        logger.info("Integrated Lab Safety System started")
    
    def stop_system(self):
        """Stop the integrated lab safety system."""
        self.running = False
        
        print("⏹️  Stopping Integrated Lab Safety System...")
        
        # Stop sensor collection
        self.sensor_collector.stop_collection()
        
        # Stop safety monitoring
        self.safety_agent.stop_monitoring()
        
        logger.info("Integrated Lab Safety System stopped")
    
    def set_experiment(self, experiment_id: str) -> bool:
        """Set the current experiment and update safety thresholds accordingly."""
        if self.experiment_manager.set_current_experiment(experiment_id):
            self._update_safety_thresholds()
            return True
        return False
    
    def _update_safety_thresholds(self):
        """Update safety monitoring thresholds based on current experiment."""
        current_exp = self.experiment_manager.get_current_experiment()
        if not current_exp:
            return
        
        # Update temperature threshold
        temp_threshold = SafetyThreshold(
            parameter=ParameterType.TEMPERATURE,
            min_safe=current_exp.temperature_range.min_safe,
            max_safe=current_exp.temperature_range.max_safe,
            warning_buffer=current_exp.temperature_range.warning_buffer,
            critical_buffer=current_exp.temperature_range.critical_buffer,
            units=current_exp.temperature_range.units
        )
        
        # Update pressure threshold
        pressure_threshold = SafetyThreshold(
            parameter=ParameterType.PRESSURE,
            min_safe=current_exp.pressure_range.min_safe,
            max_safe=current_exp.pressure_range.max_safe,
            warning_buffer=current_exp.pressure_range.warning_buffer,
            critical_buffer=current_exp.pressure_range.critical_buffer,
            units=current_exp.pressure_range.units
        )
        
        # Update safety agent thresholds
        self.safety_agent.thresholds[ParameterType.TEMPERATURE] = temp_threshold
        self.safety_agent.thresholds[ParameterType.PRESSURE] = pressure_threshold
        
        # Also update sensor data collector ranges to match experiment
        self.sensor_collector.update_sensor_ranges_for_experiment(current_exp)
        
        logger.info(f"Updated safety thresholds for experiment: {current_exp.name}")
        logger.info(f"Temperature: {temp_threshold.min_safe}-{temp_threshold.max_safe}{temp_threshold.units}")
        logger.info(f"Pressure: {pressure_threshold.min_safe}-{pressure_threshold.max_safe}{pressure_threshold.units}")
    
    def get_available_experiments(self) -> List[str]:
        """Get list of available experiments."""
        return self.experiment_manager.get_experiment_list()
    
    def get_current_experiment_info(self) -> Optional[Dict]:
        """Get information about the current experiment."""
        current_exp = self.experiment_manager.get_current_experiment()
        if not current_exp:
            return None
        
        return {
            "name": current_exp.name,
            "type": current_exp.experiment_type.value,
            "description": current_exp.description,
            "temperature_range": f"{current_exp.temperature_range.min_safe}-{current_exp.temperature_range.max_safe}{current_exp.temperature_range.units}",
            "pressure_range": f"{current_exp.pressure_range.min_safe}-{current_exp.pressure_range.max_safe}{current_exp.pressure_range.units}",
            "duration_hours": current_exp.duration_hours,
            "special_notes": current_exp.special_notes
        }
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        sensor_status = self.sensor_collector.get_status_report()
        safety_status = self.safety_agent.get_status_report()
        
        return {
            "system_running": self.running,
            "sensor_system": sensor_status,
            "safety_system": safety_status,
            "integration_active": True,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict]:
        """Get recent safety alerts."""
        recent_alerts = self.safety_agent.alert_history[-count:]
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
    """Main function to run the integrated system."""
    system = IntegratedLabSafetySystem()
    
    try:
        system.start_system()
        
        print("\n" + "="*60)
        print("🧪 INTEGRATED LAB SAFETY SYSTEM ACTIVE")
        print("="*60)
        print("Real-time monitoring of lab conditions with automated safety protocols")
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        # Status reporting loop
        while system.running:
            time.sleep(30)  # Report every 30 seconds
            
            status = system.get_system_status()
            recent_alerts = system.get_recent_alerts(count=3)
            
            print(f"\n📊 SYSTEM STATUS - {datetime.now().strftime('%H:%M:%S')}")
            print(f"   Sensors Active: {status['sensor_system']['active_sensors']}")
            print(f"   Queue Size: {status['sensor_system']['queue_size']}")
            print(f"   Active Alerts: {len(status['safety_system']['active_alerts'])}")
            print(f"   Recent Alerts: {len(recent_alerts)}")
            
            if recent_alerts:
                print("   Last Alert:", recent_alerts[-1]['message'])
            
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("🛑 SHUTDOWN INITIATED BY USER")
        print("="*60)
        
        system.stop_system()
        
        # Final status report
        final_status = system.get_system_status()
        recent_alerts = system.get_recent_alerts()
        
        print(f"\n📋 FINAL SYSTEM REPORT")
        print(f"   Total Sensor Readings: {final_status['sensor_system']['history_size']}")
        print(f"   Total Safety Alerts: {len(recent_alerts)}")
        print(f"   System Runtime: Safe shutdown completed")
        print("="*60)

if __name__ == "__main__":
    main()
