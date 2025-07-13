from random import randint

class SafetyMonitoringAgent:
    def __init__(self):
        self.safety_thresholds = {
            "temperature": 100,  # Example threshold
            "pressure": 50       # Example threshold
        }

    def monitor_parameters(self):
        # Simulated parameter readings
        self.current_temperature = self.get_temperature()
        self.current_pressure = self.get_pressure()

    def get_temperature(self):
        # Simulate getting temperature from a sensor
        return randint(80, 120)

    def get_pressure(self):
        # Simulate getting pressure from a sensor
        return randint(30, 70)

    def is_safe(self):
        return (self.current_temperature <= self.safety_thresholds["temperature"] and
                self.current_pressure <= self.safety_thresholds["pressure"])

    def notify_scientist(self):
        print("Warning: Safety parameters exceeded!")
        print(f"Current Temperature: {self.current_temperature}, Threshold: {self.safety_thresholds['temperature']}")
        print(f"Current Pressure: {self.current_pressure}, Threshold: {self.safety_thresholds['pressure']}")