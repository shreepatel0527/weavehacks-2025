#!/usr/bin/env python3
"""
Sensor Data Collection Module for Lab Automation
Handles real-time sensor data collection, streaming, and integration with safety monitoring.
"""

import csv
import time
import datetime
import threading
import json
import logging
import os
import queue
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Generator
from enum import Enum
import random
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SensorType(Enum):
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    PH = "ph"
    HUMIDITY = "humidity"

@dataclass
class SensorReading:
    timestamp: datetime.datetime
    sensor_type: SensorType
    value: float
    units: str
    sensor_id: str = "default"
    location: str = "lab"

@dataclass
class SensorConfig:
    sensor_id: str
    sensor_type: SensorType
    units: str
    min_value: float
    max_value: float
    sampling_rate_seconds: float = 2.0
    enabled: bool = True

class SensorDataCollector:
    """
    Advanced sensor data collection system with real-time streaming
    and integration capabilities for lab automation.
    """
    
    def __init__(self, config_file: str = "sensor_config.json"):
        self.config_file = config_file
        self.sensors = self._load_sensor_config()
        self.data_queue = queue.Queue()
        self.callbacks: List[Callable] = []
        self.collecting = False
        self.data_history: List[SensorReading] = []
        
        # File storage settings
        self.csv_file = "sensor_data.csv"
        self.json_file = "sensor_data.json"
        
        logger.info("Sensor Data Collector initialized")
    
    def update_sensor_ranges_for_experiment(self, experiment_config):
        """Update sensor ranges based on the current experiment configuration."""
        if hasattr(experiment_config, 'temperature_range'):
            # Update temperature sensor
            if "temp_001" in self.sensors:
                temp_sensor = self.sensors["temp_001"]
                temp_sensor.min_value = experiment_config.temperature_range.min_safe
                temp_sensor.max_value = experiment_config.temperature_range.max_safe
                logger.info(f"Updated temperature sensor range: {temp_sensor.min_value}-{temp_sensor.max_value}{temp_sensor.units}")
        
        if hasattr(experiment_config, 'pressure_range'):
            # Update pressure sensor
            if "press_001" in self.sensors:
                pressure_sensor = self.sensors["press_001"]
                pressure_sensor.min_value = experiment_config.pressure_range.min_safe
                pressure_sensor.max_value = experiment_config.pressure_range.max_safe
                logger.info(f"Updated pressure sensor range: {pressure_sensor.min_value}-{pressure_sensor.max_value}{pressure_sensor.units}")
    
    def _load_sensor_config(self) -> Dict[str, SensorConfig]:
        """Load sensor configuration from file."""
        # Default sensors with wide ranges - these will be updated based on experiment selection
        default_sensors = {
            "temp_001": SensorConfig(
                sensor_id="temp_001",
                sensor_type=SensorType.TEMPERATURE,
                units="°C",
                min_value=0.0,   # Will be updated based on experiment
                max_value=50.0,  # Will be updated based on experiment
                sampling_rate_seconds=2.0
            ),
            "press_001": SensorConfig(
                sensor_id="press_001",
                sensor_type=SensorType.PRESSURE,
                units="kPa", 
                min_value=95.0,  # Will be updated based on experiment
                max_value=115.0, # Will be updated based on experiment
                sampling_rate_seconds=2.0
            )
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                    logger.info(f"Loaded sensor configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}. Using defaults.")
        else:
            self._save_default_config(default_sensors)
        
        return default_sensors
    
    def _save_default_config(self, sensors: Dict[str, SensorConfig]):
        """Save default configuration to file."""
        config_data = {
            sensor_id: asdict(config) for sensor_id, config in sensors.items()
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            logger.info(f"Saved default sensor configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save default config: {e}")
    
    def _simulate_sensor_reading(self, sensor_config: SensorConfig) -> SensorReading:
        """Simulate realistic sensor reading based on experiment type and time progression."""
        current_time = datetime.datetime.now()
        
        # Get experiment-specific behavior
        experiment_behavior = self._get_experiment_behavior(sensor_config, current_time)
        
        # Apply experiment-specific patterns
        base_value = experiment_behavior['base_value']
        noise_factor = experiment_behavior['noise_factor']
        trend = experiment_behavior.get('trend', 0)
        
        # Add realistic noise
        range_span = sensor_config.max_value - sensor_config.min_value
        noise = random.uniform(-range_span * noise_factor, range_span * noise_factor)
        
        # Apply trend and noise
        value = base_value + trend + noise
        
        # Occasionally generate readings outside safe ranges for testing safety alerts
        if random.random() < experiment_behavior.get('alert_probability', 0.05):
            if random.random() < 0.5:
                value = sensor_config.min_value - random.uniform(0.5, 2.0)
            else:
                value = sensor_config.max_value + random.uniform(0.5, 2.0)
        
        return SensorReading(
            timestamp=current_time,
            sensor_type=sensor_config.sensor_type,
            value=round(value, 2),
            units=sensor_config.units,
            sensor_id=sensor_config.sensor_id
        )
    
    def _get_experiment_behavior(self, sensor_config: SensorConfig, current_time: datetime.datetime) -> Dict:
        """Get experiment-specific sensor behavior patterns."""
        # Get time since start of collection (simulate experiment progression)
        if not hasattr(self, '_collection_start_time'):
            self._collection_start_time = current_time
        
        elapsed_minutes = (current_time - self._collection_start_time).total_seconds() / 60
        
        # Default behavior
        range_center = (sensor_config.min_value + sensor_config.max_value) / 2
        behavior = {
            'base_value': range_center,
            'noise_factor': 0.1,
            'trend': 0,
            'alert_probability': 0.03
        }
        
        # Experiment-specific patterns based on sensor ID and type
        if sensor_config.sensor_type == SensorType.TEMPERATURE:
            behavior.update(self._get_temperature_behavior(sensor_config, elapsed_minutes))
        elif sensor_config.sensor_type == SensorType.PRESSURE:
            behavior.update(self._get_pressure_behavior(sensor_config, elapsed_minutes))
        
        return behavior
    
    def _get_temperature_behavior(self, sensor_config: SensorConfig, elapsed_minutes: float) -> Dict:
        """Get temperature-specific behavior based on experiment type."""
        range_center = (sensor_config.min_value + sensor_config.max_value) / 2
        
        # Ice bath experiment (0-5°C)
        if sensor_config.max_value <= 10:
            # Cooling pattern: starts higher, drops to ice bath temp
            if elapsed_minutes < 5:
                # Initial cooling phase
                base_temp = max(sensor_config.min_value + 3, 
                               25 - (elapsed_minutes * 4))  # Cool from room temp
                return {
                    'base_value': base_temp,
                    'noise_factor': 0.15,
                    'trend': -0.5 if elapsed_minutes < 3 else 0,
                    'alert_probability': 0.02
                }
            else:
                # Stable ice bath phase
                return {
                    'base_value': sensor_config.min_value + 1.5,
                    'noise_factor': 0.08,
                    'trend': random.uniform(-0.1, 0.1),
                    'alert_probability': 0.01
                }
        
        # Room temperature experiment (20-25°C)
        elif 15 <= sensor_config.min_value <= 25 and sensor_config.max_value <= 30:
            return {
                'base_value': range_center + random.uniform(-1, 1),
                'noise_factor': 0.12,
                'trend': random.uniform(-0.2, 0.2),
                'alert_probability': 0.03
            }
        
        # Elevated temperature experiment (25-35°C)
        elif sensor_config.min_value >= 20 and sensor_config.max_value >= 30:
            # Heating pattern during vigorous stirring
            if elapsed_minutes < 10:
                # Gradual heating phase
                base_temp = min(sensor_config.max_value - 2,
                               sensor_config.min_value + (elapsed_minutes * 0.8))
                return {
                    'base_value': base_temp,
                    'noise_factor': 0.2,
                    'trend': 0.3 if elapsed_minutes < 8 else -0.1,
                    'alert_probability': 0.06
                }
            else:
                # Stable elevated temperature
                return {
                    'base_value': sensor_config.max_value - 3,
                    'noise_factor': 0.15,
                    'trend': random.uniform(-0.3, 0.3),
                    'alert_probability': 0.04
                }
        
        # Overnight experiment (18-28°C)
        else:
            # Stable with minor fluctuations
            return {
                'base_value': range_center + random.uniform(-2, 2),
                'noise_factor': 0.1,
                'trend': random.uniform(-0.1, 0.1),
                'alert_probability': 0.02
            }
    
    def _get_pressure_behavior(self, sensor_config: SensorConfig, elapsed_minutes: float) -> Dict:
        """Get pressure-specific behavior based on experiment type."""
        range_center = (sensor_config.min_value + sensor_config.max_value) / 2
        
        # Normal atmospheric pressure experiments
        if sensor_config.min_value >= 99 and sensor_config.max_value <= 105:
            return {
                'base_value': range_center + random.uniform(-0.5, 0.5),
                'noise_factor': 0.08,
                'trend': random.uniform(-0.05, 0.05),
                'alert_probability': 0.02
            }
        
        # Experiments with slight pressure variation (stirring, N2 atmosphere)
        else:
            # Slight pressure changes due to stirring or gas atmosphere
            pressure_wave = 0.3 * math.sin(elapsed_minutes * 0.5)  # Gentle oscillation
            return {
                'base_value': range_center + pressure_wave,
                'noise_factor': 0.12,
                'trend': random.uniform(-0.1, 0.1),
                'alert_probability': 0.03
            }
    
    def add_callback(self, callback: Callable[[SensorReading], None]):
        """Add a callback function to be called when new sensor data is available."""
        self.callbacks.append(callback)
    
    def start_collection(self):
        """Start sensor data collection."""
        if self.collecting:
            logger.warning("Sensor collection already running")
            return
        
        self.collecting = True
        
        # Initialize CSV file with headers
        self._initialize_csv_file()
        
        # Start collection threads for each sensor
        for sensor_config in self.sensors.values():
            if sensor_config.enabled:
                thread = threading.Thread(
                    target=self._collect_sensor_data,
                    args=(sensor_config,),
                    daemon=True
                )
                thread.start()
        
        # Start data processing thread
        processing_thread = threading.Thread(
            target=self._process_data_queue,
            daemon=True
        )
        processing_thread.start()
        
        logger.info("Sensor data collection started")
    
    def stop_collection(self):
        """Stop sensor data collection."""
        self.collecting = False
        logger.info("Sensor data collection stopped")
    
    def _initialize_csv_file(self):
        """Initialize CSV file with headers."""
        try:
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'sensor_id', 'sensor_type', 'value', 'units', 'location'
                ])
            logger.info(f"Initialized CSV file: {self.csv_file}")
        except Exception as e:
            logger.error(f"Failed to initialize CSV file: {e}")
    
    def _collect_sensor_data(self, sensor_config: SensorConfig):
        """Collect data from a specific sensor."""
        logger.info(f"Starting data collection for sensor: {sensor_config.sensor_id}")
        
        while self.collecting:
            try:
                # Simulate sensor reading (replace with actual sensor interface)
                reading = self._simulate_sensor_reading(sensor_config)
                
                # Add to queue for processing
                self.data_queue.put(reading)
                
                # Sleep based on sensor sampling rate
                time.sleep(sensor_config.sampling_rate_seconds)
                
            except Exception as e:
                logger.error(f"Error collecting data from {sensor_config.sensor_id}: {e}")
                time.sleep(1)  # Brief pause before retry
    
    def _process_data_queue(self):
        """Process sensor data from the queue."""
        while self.collecting:
            try:
                # Get data from queue (with timeout to allow checking self.collecting)
                reading = self.data_queue.get(timeout=1)
                
                # Store in history
                self.data_history.append(reading)
                
                # Keep only last 1000 readings in memory
                if len(self.data_history) > 1000:
                    self.data_history = self.data_history[-1000:]
                
                # Save to CSV
                self._save_to_csv(reading)
                
                # Save to JSON
                self._save_to_json(reading)
                
                # Execute callbacks
                for callback in self.callbacks:
                    try:
                        callback(reading)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                # Mark task as done
                self.data_queue.task_done()
                
            except queue.Empty:
                continue  # No data available, continue loop
            except Exception as e:
                logger.error(f"Error processing data queue: {e}")
    
    def _save_to_csv(self, reading: SensorReading):
        """Save sensor reading to CSV file."""
        try:
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    reading.timestamp.isoformat(),
                    reading.sensor_id,
                    reading.sensor_type.value,
                    reading.value,
                    reading.units,
                    reading.location
                ])
        except Exception as e:
            logger.error(f"Failed to save to CSV: {e}")
    
    def _save_to_json(self, reading: SensorReading):
        """Save sensor reading to JSON file (append mode)."""
        try:
            data_entry = {
                "timestamp": reading.timestamp.isoformat(),
                "sensor_id": reading.sensor_id,
                "sensor_type": reading.sensor_type.value,
                "value": reading.value,
                "units": reading.units,
                "location": reading.location
            }
            
            # Append to JSON file
            with open(self.json_file, 'a') as f:
                f.write(json.dumps(data_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to save to JSON: {e}")
    
    def get_latest_readings(self, sensor_type: Optional[SensorType] = None, 
                          count: int = 10) -> List[SensorReading]:
        """Get the latest sensor readings."""
        readings = self.data_history
        
        if sensor_type:
            readings = [r for r in readings if r.sensor_type == sensor_type]
        
        return readings[-count:] if readings else []
    
    def get_status_report(self) -> Dict:
        """Generate a status report."""
        active_sensors = [s for s in self.sensors.values() if s.enabled]
        recent_readings = self.get_latest_readings(count=5)
        
        return {
            "collecting": self.collecting,
            "active_sensors": len(active_sensors),
            "total_sensors": len(self.sensors),
            "queue_size": self.data_queue.qsize(),
            "history_size": len(self.data_history),
            "recent_readings": [asdict(r) for r in recent_readings],
            "csv_file": self.csv_file,
            "json_file": self.json_file
        }

def main():
    """Main function for testing the sensor data collector."""
    collector = SensorDataCollector()
    
    # Add a simple callback to print readings
    def print_reading(reading: SensorReading):
        print(f"📊 {reading.sensor_id}: {reading.value}{reading.units} "
              f"({reading.sensor_type.value}) at {reading.timestamp.strftime('%H:%M:%S')}")
    
    collector.add_callback(print_reading)
    
    try:
        print("🔬 Starting sensor data collection...")
        print("📈 Simulating lab sensors: Temperature, Pressure")
        print("💾 Data saved to sensor_data.csv and sensor_data.json")
        print("Press Ctrl+C to stop")
        
        collector.start_collection()
        
        # Keep main thread alive and print status every 30 seconds
        while collector.collecting:
            time.sleep(30)
            status = collector.get_status_report()
            print(f"\n📋 Status: {status['queue_size']} queued, "
                  f"{status['history_size']} in history")
            
    except KeyboardInterrupt:
        print("\n⏹️  Stopping sensor data collection...")
        collector.stop_collection()
        
        # Print final status
        status = collector.get_status_report()
        print(f"\n📊 Final Status Report:")
        print(json.dumps(status, indent=2, default=str))

if __name__ == "__main__":
    main()