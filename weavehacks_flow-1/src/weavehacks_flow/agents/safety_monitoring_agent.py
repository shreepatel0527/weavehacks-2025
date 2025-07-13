import json
import time
from datetime import datetime
from random import randint
import weave
import wandb

def safe_wandb_log(data: dict):
    """Safely log to wandb, handling cases where wandb is not initialized"""
    try:
        wandb.log(data)
    except wandb.errors.UsageError:
        # wandb not initialized, try to initialize minimally
        try:
            wandb.init(project="lab-assistant-agents", mode="disabled")
            wandb.log(data)
        except Exception:
            # If all else fails, just skip logging
            pass
    except Exception:
        # Any other wandb error, skip logging
        pass
from pathlib import Path

class SafetyMonitoringAgent:
    def __init__(self):
        # Load safety configuration
        self.load_safety_config()
        self.sensor_data_file = None
        self.sensor_data = []
        self.current_index = 0
        self.monitoring_active = False
        self.safety_events = []
        
    def load_safety_config(self):
        """Load safety thresholds from configuration file"""
        try:
            config_path = Path(__file__).parent.parent.parent.parent.parent / "Prototype-1" / "safety_config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            self.safety_thresholds = {
                "temperature": {
                    "min": config["temperature"]["min_safe"],
                    "max": config["temperature"]["max_safe"],
                    "warning_buffer": config["temperature"]["warning_buffer"],
                    "units": config["temperature"]["units"]
                },
                "pressure": {
                    "min": config["pressure"]["min_safe"],
                    "max": config["pressure"]["max_safe"],
                    "warning_buffer": config["pressure"]["warning_buffer"],
                    "units": config["pressure"]["units"]
                },
                "nitrogen": {
                    "min": config["nitrogen"]["min_safe"],
                    "max": config["nitrogen"]["max_safe"],
                    "warning_buffer": config["nitrogen"]["warning_buffer"],
                    "units": config["nitrogen"]["units"]
                },
                "oxygen": {
                    "min": config["oxygen"]["min_safe"],
                    "max": config["oxygen"]["max_safe"],
                    "warning_buffer": config["oxygen"]["warning_buffer"],
                    "units": config["oxygen"]["units"]
                }
            }
        except Exception as e:
            print(f"Warning: Could not load safety config: {e}")
            # Fallback to default values
            self.safety_thresholds = {
                "temperature": {"min": 15, "max": 35, "warning_buffer": 2, "units": "°C"},
                "pressure": {"min": 95, "max": 110, "warning_buffer": 2, "units": "kPa"},
                "nitrogen": {"min": 75, "max": 85, "warning_buffer": 2, "units": "%"},
                "oxygen": {"min": 19, "max": 23, "warning_buffer": 1, "units": "%"}
            }
    
    def load_sensor_data(self, use_real_data=True):
        """Load sensor data from file or simulate"""
        if use_real_data:
            try:
                sensor_file = Path(__file__).parent.parent.parent.parent.parent / "Prototype-1" / "sensor_data.json"
                with open(sensor_file, 'r') as f:
                    # Read all lines as separate JSON objects
                    self.sensor_data = []
                    for line in f:
                        self.sensor_data.append(json.loads(line.strip()))
                print(f"Loaded {len(self.sensor_data)} sensor readings")
                return True
            except Exception as e:
                print(f"Warning: Could not load sensor data: {e}")
                return False
        return False

    @weave.op()
    def monitor_parameters(self, use_real_data=True):
        """Monitor safety parameters from sensor stream"""
        # Try to load real sensor data if not already loaded
        if not self.sensor_data and use_real_data:
            self.load_sensor_data(use_real_data)
        
        # Get current readings
        if self.sensor_data and self.current_index < len(self.sensor_data):
            # Use real sensor data
            current_readings = {}
            
            # Collect readings from the next few data points to get all sensors
            for i in range(min(4, len(self.sensor_data) - self.current_index)):
                reading = self.sensor_data[self.current_index + i]
                sensor_type = reading.get("sensor_type", "")
                if sensor_type in ["temperature", "pressure", "nitrogen", "oxygen"]:
                    current_readings[sensor_type] = reading.get("value", 0)
            
            self.current_temperature = current_readings.get("temperature", self.get_temperature())
            self.current_pressure = current_readings.get("pressure", self.get_pressure())
            self.current_nitrogen = current_readings.get("nitrogen", self.get_nitrogen())
            self.current_oxygen = current_readings.get("oxygen", self.get_oxygen())
            
            # Advance the index for next reading
            self.current_index += 4
        else:
            # Fallback to simulated data
            self.current_temperature = self.get_temperature()
            self.current_pressure = self.get_pressure()
            self.current_nitrogen = self.get_nitrogen()
            self.current_oxygen = self.get_oxygen()
        
        # Log to W&B
        safe_wandb_log({
            'safety_monitoring': {
                'temperature': self.current_temperature,
                'pressure': self.current_pressure,
                'nitrogen': self.current_nitrogen,
                'oxygen': self.current_oxygen,
                'timestamp': datetime.now().isoformat()
            }
        })

    def get_temperature(self):
        """Simulate getting temperature from a sensor"""
        return randint(20, 40)

    def get_pressure(self):
        """Simulate getting pressure from a sensor"""
        return randint(90, 115)
    
    def get_nitrogen(self):
        """Simulate getting nitrogen level from a sensor"""
        return randint(70, 90)
    
    def get_oxygen(self):
        """Simulate getting oxygen level from a sensor"""
        return randint(18, 24)

    @weave.op()
    def is_safe(self):
        """Check if all parameters are within safe ranges"""
        temp_safe = (self.safety_thresholds["temperature"]["min"] <= self.current_temperature <= 
                    self.safety_thresholds["temperature"]["max"])
        
        pressure_safe = (self.safety_thresholds["pressure"]["min"] <= self.current_pressure <= 
                        self.safety_thresholds["pressure"]["max"])
        
        nitrogen_safe = (self.safety_thresholds["nitrogen"]["min"] <= self.current_nitrogen <= 
                        self.safety_thresholds["nitrogen"]["max"])
        
        oxygen_safe = (self.safety_thresholds["oxygen"]["min"] <= self.current_oxygen <= 
                      self.safety_thresholds["oxygen"]["max"])
        
        all_safe = temp_safe and pressure_safe and nitrogen_safe and oxygen_safe
        
        # Log safety status
        safe_wandb_log({
            'safety_check': {
                'temperature_safe': temp_safe,
                'pressure_safe': pressure_safe,
                'nitrogen_safe': nitrogen_safe,
                'oxygen_safe': oxygen_safe,
                'overall_safe': all_safe
            }
        })
        
        return all_safe
    
    @weave.op()
    def check_warning_levels(self):
        """Check if any parameters are approaching unsafe levels"""
        warnings = []
        
        # Temperature warnings
        temp_min_warn = self.safety_thresholds["temperature"]["min"] + self.safety_thresholds["temperature"]["warning_buffer"]
        temp_max_warn = self.safety_thresholds["temperature"]["max"] - self.safety_thresholds["temperature"]["warning_buffer"]
        
        if self.current_temperature <= temp_min_warn:
            warnings.append(f"Temperature approaching minimum safe level: {self.current_temperature}°C")
        elif self.current_temperature >= temp_max_warn:
            warnings.append(f"Temperature approaching maximum safe level: {self.current_temperature}°C")
        
        # Pressure warnings
        press_min_warn = self.safety_thresholds["pressure"]["min"] + self.safety_thresholds["pressure"]["warning_buffer"]
        press_max_warn = self.safety_thresholds["pressure"]["max"] - self.safety_thresholds["pressure"]["warning_buffer"]
        
        if self.current_pressure <= press_min_warn:
            warnings.append(f"Pressure approaching minimum safe level: {self.current_pressure}kPa")
        elif self.current_pressure >= press_max_warn:
            warnings.append(f"Pressure approaching maximum safe level: {self.current_pressure}kPa")
        
        return warnings

    @weave.op()
    def notify_scientist(self):
        """Notify scientist of safety concerns"""
        print("\n" + "="*50)
        print("⚠️  SAFETY ALERT! ⚠️")
        print("="*50)
        print(f"Current Temperature: {self.current_temperature}{self.safety_thresholds['temperature']['units']}")
        print(f"  Safe range: {self.safety_thresholds['temperature']['min']}-{self.safety_thresholds['temperature']['max']}{self.safety_thresholds['temperature']['units']}")
        print(f"Current Pressure: {self.current_pressure}{self.safety_thresholds['pressure']['units']}")
        print(f"  Safe range: {self.safety_thresholds['pressure']['min']}-{self.safety_thresholds['pressure']['max']}{self.safety_thresholds['pressure']['units']}")
        print(f"Current Nitrogen: {self.current_nitrogen}{self.safety_thresholds['nitrogen']['units']}")
        print(f"  Safe range: {self.safety_thresholds['nitrogen']['min']}-{self.safety_thresholds['nitrogen']['max']}{self.safety_thresholds['nitrogen']['units']}")
        print(f"Current Oxygen: {self.current_oxygen}{self.safety_thresholds['oxygen']['units']}")
        print(f"  Safe range: {self.safety_thresholds['oxygen']['min']}-{self.safety_thresholds['oxygen']['max']}{self.safety_thresholds['oxygen']['units']}")
        
        # Check for warnings
        warnings = self.check_warning_levels()
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  - {warning}")
        
        print("="*50 + "\n")
        
        # Log alert to W&B
        safe_wandb_log({
            'safety_alert': {
                'timestamp': datetime.now().isoformat(),
                'temperature': self.current_temperature,
                'pressure': self.current_pressure,
                'nitrogen': self.current_nitrogen,
                'oxygen': self.current_oxygen,
                'warnings': warnings
            }
        })
        
        # Record safety event
        self.safety_events.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'safety_alert',
            'parameters': {
                'temperature': self.current_temperature,
                'pressure': self.current_pressure,
                'nitrogen': self.current_nitrogen,
                'oxygen': self.current_oxygen
            }
        })