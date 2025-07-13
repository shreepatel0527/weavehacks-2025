#!/usr/bin/env python3
"""
Demo Data Generator for Different Experiment Types
Generates sample sensor data files for demonstration purposes.
"""

import csv
import datetime
import random
import math
import time
from experiment_config import ExperimentManager

def generate_experiment_data(experiment_id: str, duration_minutes: int = 30, filename: str = None):
    """Generate realistic sensor data for a specific experiment."""
    
    manager = ExperimentManager()
    experiment = manager.get_experiment_config(experiment_id)
    
    if not experiment:
        print(f"Experiment {experiment_id} not found!")
        return
    
    if not filename:
        filename = f"demo_data_{experiment_id}.csv"
    
    print(f"Generating {duration_minutes} minutes of data for: {experiment.name}")
    print(f"Temperature range: {experiment.temperature_range.min_safe}-{experiment.temperature_range.max_safe}°C")
    print(f"Pressure range: {experiment.pressure_range.min_safe}-{experiment.pressure_range.max_safe} kPa")
    
    # Create CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'sensor_id', 'sensor_type', 'value', 'units', 'location'])
        
        start_time = datetime.datetime.now()
        
        for minute in range(duration_minutes):
            current_time = start_time + datetime.timedelta(minutes=minute)
            
            # Generate temperature data
            temp_data = generate_temperature_for_experiment(experiment, minute)
            writer.writerow([
                current_time.isoformat(),
                'temp_001',
                'temperature',
                temp_data,
                '°C',
                'lab'
            ])
            
            # Generate pressure data
            pressure_data = generate_pressure_for_experiment(experiment, minute)
            writer.writerow([
                current_time.isoformat(),
                'press_001', 
                'pressure',
                pressure_data,
                'kPa',
                'lab'
            ])
    
    print(f"Generated demo data: {filename}")

def generate_temperature_for_experiment(experiment, elapsed_minutes):
    """Generate realistic temperature data based on experiment type."""
    temp_range = experiment.temperature_range
    range_center = (temp_range.min_safe + temp_range.max_safe) / 2
    
    # Ice bath experiment (0-5°C)
    if temp_range.max_safe <= 10:
        if elapsed_minutes < 5:
            # Cooling from room temperature
            base_temp = max(temp_range.min_safe + 2, 25 - (elapsed_minutes * 4))
            noise = random.uniform(-0.5, 0.5)
        else:
            # Stable ice bath
            base_temp = temp_range.min_safe + 1.5
            noise = random.uniform(-0.3, 0.3)
        return round(base_temp + noise, 2)
    
    # Room temperature experiment (20-25°C)
    elif 15 <= temp_range.min_safe <= 25 and temp_range.max_safe <= 30:
        base_temp = range_center + random.uniform(-1, 1)
        noise = random.uniform(-0.8, 0.8)
        return round(base_temp + noise, 2)
    
    # Elevated temperature experiment (25-35°C)
    elif temp_range.min_safe >= 20 and temp_range.max_safe >= 30:
        if elapsed_minutes < 10:
            # Heating phase
            base_temp = min(temp_range.max_safe - 2, temp_range.min_safe + (elapsed_minutes * 0.8))
            noise = random.uniform(-1.0, 1.0)
        else:
            # Stable elevated temperature
            base_temp = temp_range.max_safe - 3
            noise = random.uniform(-1.5, 1.5)
        return round(base_temp + noise, 2)
    
    # Overnight experiment (18-28°C)
    else:
        base_temp = range_center + random.uniform(-2, 2)
        noise = random.uniform(-0.5, 0.5)
        return round(base_temp + noise, 2)

def generate_pressure_for_experiment(experiment, elapsed_minutes):
    """Generate realistic pressure data based on experiment type."""
    pressure_range = experiment.pressure_range
    range_center = (pressure_range.min_safe + pressure_range.max_safe) / 2
    
    # Add slight oscillation for stirring experiments
    if "stirring" in experiment.name.lower():
        oscillation = 0.3 * math.sin(elapsed_minutes * 0.5)
    else:
        oscillation = 0
    
    base_pressure = range_center + oscillation
    noise = random.uniform(-0.5, 0.5)
    
    return round(base_pressure + noise, 2)

def generate_all_experiment_demos():
    """Generate demo data for all experiment types."""
    experiments = [
        ("gold_nanoparticle_room_temp", 20),
        ("gold_nanoparticle_ice_bath", 15),
        ("gold_nanoparticle_stirring", 10),
        ("overnight_stirring", 60),
        ("custom_experiment", 25)
    ]
    
    print("🧪 Generating demo data for all experiment types...")
    print("=" * 50)
    
    for exp_id, duration in experiments:
        generate_experiment_data(exp_id, duration)
        print()
    
    print("✅ All demo data files generated!")
    print("\nGenerated files:")
    for exp_id, _ in experiments:
        print(f"  - demo_data_{exp_id}.csv")

if __name__ == "__main__":
    generate_all_experiment_demos()