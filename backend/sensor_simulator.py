#!/usr/bin/env python3
"""
WeaveHacks 2025 - Sensor Data Simulator
Generates realistic temperature and pressure data for lab experiments
"""

import asyncio
import random
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import weave

class SensorType(Enum):
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    GAS_LEVEL = "gas_level"
    PH = "ph"

@dataclass
class SensorReading:
    sensor_id: str
    sensor_type: SensorType
    value: float
    units: str
    timestamp: datetime
    location: str = "lab"
    experiment_id: Optional[str] = None

class ExperimentProfile:
    """Defines expected sensor behavior for different experiment types"""
    
    def __init__(self, name: str, temp_range: tuple, pressure_range: tuple, duration_hours: int):
        self.name = name
        self.temp_range = temp_range  # (min, max) in Celsius
        self.pressure_range = pressure_range  # (min, max) in kPa
        self.duration_hours = duration_hours
        self.start_time = None
    
    def get_expected_temperature(self, elapsed_minutes: int) -> float:
        """Get expected temperature based on experiment progress"""
        temp_min, temp_max = self.temp_range
        
        # Simulate heating and cooling cycles
        if elapsed_minutes < 30:  # Initial heating
            progress = elapsed_minutes / 30
            return temp_min + (temp_max - temp_min) * progress
        elif elapsed_minutes < 120:  # Stable high temperature
            return temp_max + random.uniform(-2, 2)
        elif elapsed_minutes < 180:  # Cooling phase
            progress = (elapsed_minutes - 120) / 60
            return temp_max - (temp_max - temp_min) * 0.7 * progress
        else:  # Final stable phase
            return temp_min + (temp_max - temp_min) * 0.3 + random.uniform(-1, 1)
    
    def get_expected_pressure(self, elapsed_minutes: int) -> float:
        """Get expected pressure based on experiment progress"""
        pressure_min, pressure_max = self.pressure_range
        
        # Pressure follows temperature with some lag
        if elapsed_minutes < 45:  # Gradual pressure increase
            progress = elapsed_minutes / 45
            return pressure_min + (pressure_max - pressure_min) * progress
        elif elapsed_minutes < 150:  # High pressure phase
            return pressure_max + random.uniform(-0.5, 0.5)
        else:  # Pressure release
            progress = min(1.0, (elapsed_minutes - 150) / 90)
            return pressure_max - (pressure_max - pressure_min) * 0.8 * progress

class SensorSimulator:
    """Simulates realistic sensor data for lab experiments"""
    
    def __init__(self):
        self.sensors = {
            "TEMP_001": {"type": SensorType.TEMPERATURE, "location": "reaction_vessel"},
            "TEMP_002": {"type": SensorType.TEMPERATURE, "location": "fume_hood"},
            "PRESS_001": {"type": SensorType.PRESSURE, "location": "reaction_vessel"},
            "PRESS_002": {"type": SensorType.PRESSURE, "location": "nitrogen_line"},
            "GAS_001": {"type": SensorType.GAS_LEVEL, "location": "fume_hood"},
        }
        
        self.experiment_profiles = {
            "nanoparticle_synthesis": ExperimentProfile(
                "Au Nanoparticle Synthesis",
                temp_range=(20, 85),  # Room temp to 85°C
                pressure_range=(100, 105),  # Slightly above atmospheric
                duration_hours=4
            ),
            "high_temp_reaction": ExperimentProfile(
                "High Temperature Reaction",
                temp_range=(25, 150),
                pressure_range=(98, 110),
                duration_hours=6
            ),
            "pressure_reaction": ExperimentProfile(
                "Pressure Reaction",
                temp_range=(40, 80),
                pressure_range=(100, 120),
                duration_hours=8
            )
        }
        
        self.current_profile = None
        self.experiment_start_time = None
        self.data_queue = queue.Queue()
        self.is_running = False
        self.simulation_thread = None
        
        # Safety thresholds
        self.safety_thresholds = {
            SensorType.TEMPERATURE: {"warning": 90, "critical": 100},
            SensorType.PRESSURE: {"warning": 115, "critical": 125},
            SensorType.GAS_LEVEL: {"warning": 1000, "critical": 5000}  # ppm
        }
    
    @weave.op()
    def start_experiment(self, experiment_type: str, experiment_id: str):
        """Start sensor simulation for a specific experiment"""
        if experiment_type not in self.experiment_profiles:
            return False, f"Unknown experiment type: {experiment_type}"
        
        self.current_profile = self.experiment_profiles[experiment_type]
        self.experiment_start_time = datetime.now()
        self.current_profile.start_time = self.experiment_start_time
        self.current_experiment_id = experiment_id
        
        if not self.is_running:
            self.start_simulation()
        
        return True, f"Started simulation for {experiment_type}"
    
    def start_simulation(self):
        """Start the sensor data generation thread"""
        if self.is_running:
            return
        
        self.is_running = True
        self.simulation_thread = threading.Thread(target=self._simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
    
    def stop_simulation(self):
        """Stop the sensor simulation"""
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2)
    
    def _simulation_loop(self):
        """Main simulation loop running in background thread"""
        while self.is_running:
            try:
                # Generate sensor readings every 5 seconds
                readings = self._generate_readings()
                for reading in readings:
                    self.data_queue.put(reading)
                
                time.sleep(5)  # 5-second intervals
                
            except Exception as e:
                print(f"Simulation error: {e}")
                time.sleep(1)
    
    def _generate_readings(self) -> List[SensorReading]:
        """Generate current sensor readings"""
        readings = []
        current_time = datetime.now()
        
        if not self.current_profile or not self.experiment_start_time:
            # No active experiment - generate baseline readings
            return self._generate_baseline_readings(current_time)
        
        # Calculate elapsed time
        elapsed = current_time - self.experiment_start_time
        elapsed_minutes = elapsed.total_seconds() / 60
        
        # Generate temperature readings
        for sensor_id, sensor_info in self.sensors.items():
            if sensor_info["type"] == SensorType.TEMPERATURE:
                expected_temp = self.current_profile.get_expected_temperature(elapsed_minutes)
                
                # Add realistic noise and sensor-specific offsets
                if sensor_info["location"] == "fume_hood":
                    # Fume hood is typically cooler
                    actual_temp = expected_temp - 5 + random.uniform(-2, 2)
                else:
                    actual_temp = expected_temp + random.uniform(-1.5, 1.5)
                
                readings.append(SensorReading(
                    sensor_id=sensor_id,
                    sensor_type=SensorType.TEMPERATURE,
                    value=round(actual_temp, 1),
                    units="°C",
                    timestamp=current_time,
                    location=sensor_info["location"],
                    experiment_id=getattr(self, 'current_experiment_id', None)
                ))
        
        # Generate pressure readings
        for sensor_id, sensor_info in self.sensors.items():
            if sensor_info["type"] == SensorType.PRESSURE:
                expected_pressure = self.current_profile.get_expected_pressure(elapsed_minutes)
                
                # Add realistic pressure variations
                if sensor_info["location"] == "nitrogen_line":
                    # Nitrogen line pressure is more stable
                    actual_pressure = expected_pressure + random.uniform(-0.2, 0.2)
                else:
                    actual_pressure = expected_pressure + random.uniform(-0.8, 0.8)
                
                readings.append(SensorReading(
                    sensor_id=sensor_id,
                    sensor_type=SensorType.PRESSURE,
                    value=round(actual_pressure, 2),
                    units="kPa",
                    timestamp=current_time,
                    location=sensor_info["location"],
                    experiment_id=getattr(self, 'current_experiment_id', None)
                ))
        
        # Generate gas level readings
        for sensor_id, sensor_info in self.sensors.items():
            if sensor_info["type"] == SensorType.GAS_LEVEL:
                # Gas levels vary based on experiment activity
                base_level = 200 + elapsed_minutes * 2  # Gradual increase
                actual_level = base_level + random.uniform(-50, 100)
                
                readings.append(SensorReading(
                    sensor_id=sensor_id,
                    sensor_type=SensorType.GAS_LEVEL,
                    value=round(actual_level, 0),
                    units="ppm",
                    timestamp=current_time,
                    location=sensor_info["location"],
                    experiment_id=getattr(self, 'current_experiment_id', None)
                ))
        
        return readings
    
    def _generate_baseline_readings(self, current_time: datetime) -> List[SensorReading]:
        """Generate baseline sensor readings when no experiment is active"""
        readings = []
        
        for sensor_id, sensor_info in self.sensors.items():
            if sensor_info["type"] == SensorType.TEMPERATURE:
                # Room temperature with small variations
                temp = 22 + random.uniform(-2, 3)
                readings.append(SensorReading(
                    sensor_id=sensor_id,
                    sensor_type=SensorType.TEMPERATURE,
                    value=round(temp, 1),
                    units="°C",
                    timestamp=current_time,
                    location=sensor_info["location"]
                ))
            
            elif sensor_info["type"] == SensorType.PRESSURE:
                # Atmospheric pressure with small variations
                pressure = 101.3 + random.uniform(-0.5, 0.5)
                readings.append(SensorReading(
                    sensor_id=sensor_id,
                    sensor_type=SensorType.PRESSURE,
                    value=round(pressure, 2),
                    units="kPa",
                    timestamp=current_time,
                    location=sensor_info["location"]
                ))
            
            elif sensor_info["type"] == SensorType.GAS_LEVEL:
                # Low background gas levels
                gas_level = 150 + random.uniform(-20, 30)
                readings.append(SensorReading(
                    sensor_id=sensor_id,
                    sensor_type=SensorType.GAS_LEVEL,
                    value=round(gas_level, 0),
                    units="ppm",
                    timestamp=current_time,
                    location=sensor_info["location"]
                ))
        
        return readings
    
    def get_recent_readings(self, count: int = 50) -> List[Dict]:
        """Get recent sensor readings from the queue"""
        readings = []
        
        # Drain the queue up to count items
        for _ in range(min(count, self.data_queue.qsize())):
            try:
                reading = self.data_queue.get_nowait()
                readings.append({
                    "sensor_id": reading.sensor_id,
                    "sensor_type": reading.sensor_type.value,
                    "value": reading.value,
                    "units": reading.units,
                    "timestamp": reading.timestamp.isoformat(),
                    "location": reading.location,
                    "experiment_id": reading.experiment_id
                })
            except queue.Empty:
                break
        
        return readings
    
    def check_safety_thresholds(self, readings: List[SensorReading]) -> List[Dict]:
        """Check if any readings exceed safety thresholds"""
        alerts = []
        
        for reading in readings:
            if reading.sensor_type in self.safety_thresholds:
                thresholds = self.safety_thresholds[reading.sensor_type]
                
                if reading.value >= thresholds["critical"]:
                    alerts.append({
                        "sensor_id": reading.sensor_id,
                        "parameter": reading.sensor_type.value,
                        "value": reading.value,
                        "threshold": thresholds["critical"],
                        "severity": "critical",
                        "message": f"{reading.sensor_type.value.title()} critically high: {reading.value} {reading.units}"
                    })
                elif reading.value >= thresholds["warning"]:
                    alerts.append({
                        "sensor_id": reading.sensor_id,
                        "parameter": reading.sensor_type.value,
                        "value": reading.value,
                        "threshold": thresholds["warning"],
                        "severity": "warning",
                        "message": f"{reading.sensor_type.value.title()} elevated: {reading.value} {reading.units}"
                    })
        
        return alerts
    
    def get_experiment_status(self) -> Dict:
        """Get current experiment status"""
        if not self.current_profile or not self.experiment_start_time:
            return {
                "active": False,
                "experiment_type": None,
                "elapsed_time": 0,
                "progress_percent": 0
            }
        
        elapsed = datetime.now() - self.experiment_start_time
        elapsed_hours = elapsed.total_seconds() / 3600
        progress = min(100, (elapsed_hours / self.current_profile.duration_hours) * 100)
        
        return {
            "active": True,
            "experiment_type": self.current_profile.name,
            "elapsed_time": elapsed.total_seconds(),
            "elapsed_hours": elapsed_hours,
            "progress_percent": progress,
            "expected_duration_hours": self.current_profile.duration_hours
        }

# Global simulator instance
sensor_simulator = SensorSimulator()