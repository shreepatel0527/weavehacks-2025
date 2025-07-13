#!/usr/bin/env python3
"""
WeaveHacks 2025 - Sensor Data Demo
Quick demo script to test sensor simulation and API endpoints
"""

import requests
import time
import json
from datetime import datetime

API_BASE_URL = "http://localhost:8000"

def test_sensor_simulation():
    """Test the sensor simulation system"""
    print("🧪 WeaveHacks Sensor Simulation Demo")
    print("=" * 50)
    
    # Check backend health
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Backend API is healthy")
        else:
            print("❌ Backend API not responding correctly")
            return
    except:
        print("❌ Cannot connect to backend API")
        print("   Start backend with: cd backend && uvicorn main:app --reload")
        return
    
    # Create test experiment
    experiment_id = f"sensor_demo_{int(time.time())}"
    print(f"\n📋 Creating test experiment: {experiment_id}")
    
    try:
        response = requests.post(f"{API_BASE_URL}/experiments", 
                               params={"experiment_id": experiment_id})
        if response.status_code == 200:
            print("✅ Experiment created successfully")
        else:
            print(f"❌ Failed to create experiment: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error creating experiment: {e}")
        return
    
    # Get available experiment types
    print("\n🔬 Available experiment types:")
    try:
        response = requests.get(f"{API_BASE_URL}/sensors/experiment-types")
        if response.status_code == 200:
            data = response.json()
            for exp_type, profile in data["profiles"].items():
                print(f"  • {exp_type}: {profile['name']}")
                print(f"    Temperature: {profile['temperature_range'][0]}-{profile['temperature_range'][1]}°C")
                print(f"    Pressure: {profile['pressure_range'][0]}-{profile['pressure_range'][1]} kPa")
                print(f"    Duration: {profile['duration_hours']} hours")
        else:
            print("❌ Could not fetch experiment types")
    except Exception as e:
        print(f"❌ Error fetching experiment types: {e}")
    
    # Start sensor monitoring
    print(f"\n🟢 Starting sensor monitoring for nanoparticle synthesis...")
    try:
        response = requests.post(f"{API_BASE_URL}/sensors/start-experiment",
                               params={"experiment_type": "nanoparticle_synthesis", 
                                      "experiment_id": experiment_id})
        if response.status_code == 200:
            print("✅ Sensor monitoring started!")
        else:
            print(f"❌ Failed to start monitoring: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error starting monitoring: {e}")
        return
    
    # Monitor sensor data for 60 seconds
    print("\n📊 Monitoring sensor data for 60 seconds...")
    print("=" * 50)
    
    for i in range(12):  # 12 iterations of 5 seconds each
        try:
            # Get sensor status
            response = requests.get(f"{API_BASE_URL}/sensors/status")
            if response.status_code == 200:
                data = response.json()
                
                # Display experiment status
                exp_status = data.get("experiment", {})
                if exp_status.get("active", False):
                    elapsed_hours = exp_status.get("elapsed_hours", 0)
                    progress = exp_status.get("progress_percent", 0)
                    print(f"\n⏱️  Time {i*5+5}s | Experiment Progress: {progress:.1f}% | Elapsed: {elapsed_hours:.2f}h")
                
                # Display recent readings
                recent_readings = data.get("recent_readings", [])
                if recent_readings:
                    # Group by sensor type
                    temp_readings = [r for r in recent_readings if r["sensor_type"] == "temperature"]
                    pressure_readings = [r for r in recent_readings if r["sensor_type"] == "pressure"]
                    gas_readings = [r for r in recent_readings if r["sensor_type"] == "gas_level"]
                    
                    print("🌡️  Temperature:")
                    for reading in temp_readings[-2:]:  # Show last 2 temperature readings
                        time_str = datetime.fromisoformat(reading["timestamp"]).strftime("%H:%M:%S")
                        print(f"   {reading['sensor_id']}: {reading['value']:.1f}°C ({reading['location']}) at {time_str}")
                    
                    print("📊 Pressure:")
                    for reading in pressure_readings[-2:]:  # Show last 2 pressure readings
                        time_str = datetime.fromisoformat(reading["timestamp"]).strftime("%H:%M:%S")
                        print(f"   {reading['sensor_id']}: {reading['value']:.2f} kPa ({reading['location']}) at {time_str}")
                    
                    if gas_readings:
                        print("💨 Gas Levels:")
                        for reading in gas_readings[-1:]:  # Show last gas reading
                            time_str = datetime.fromisoformat(reading["timestamp"]).strftime("%H:%M:%S")
                            print(f"   {reading['sensor_id']}: {reading['value']:.0f} ppm ({reading['location']}) at {time_str}")
                
                # Check for safety alerts
                safety_alerts = data.get("safety_alerts", [])
                if safety_alerts:
                    print("🚨 Safety Alerts:")
                    for alert in safety_alerts:
                        severity_icon = {"warning": "🟡", "critical": "🔴"}.get(alert["severity"], "⚪")
                        print(f"   {severity_icon} {alert['message']}")
                
            else:
                print(f"❌ Error getting sensor status: {response.status_code}")
        
        except Exception as e:
            print(f"❌ Error monitoring sensors: {e}")
        
        # Wait before next reading
        if i < 11:  # Don't wait after the last iteration
            time.sleep(5)
    
    # Stop monitoring
    print(f"\n🔴 Stopping sensor monitoring...")
    try:
        response = requests.post(f"{API_BASE_URL}/sensors/stop")
        if response.status_code == 200:
            print("✅ Sensor monitoring stopped")
        else:
            print(f"❌ Failed to stop monitoring: {response.status_code}")
    except Exception as e:
        print(f"❌ Error stopping monitoring: {e}")
    
    # Final sensor readings
    print(f"\n📈 Final sensor data summary:")
    try:
        response = requests.get(f"{API_BASE_URL}/sensors/readings", params={"count": 20})
        if response.status_code == 200:
            data = response.json()
            readings = data["readings"]
            
            if readings:
                # Calculate averages
                temp_values = [r["value"] for r in readings if r["sensor_type"] == "temperature"]
                pressure_values = [r["value"] for r in readings if r["sensor_type"] == "pressure"]
                
                if temp_values:
                    avg_temp = sum(temp_values) / len(temp_values)
                    max_temp = max(temp_values)
                    min_temp = min(temp_values)
                    print(f"   🌡️  Temperature: Avg {avg_temp:.1f}°C, Range {min_temp:.1f}-{max_temp:.1f}°C")
                
                if pressure_values:
                    avg_pressure = sum(pressure_values) / len(pressure_values)
                    max_pressure = max(pressure_values)
                    min_pressure = min(pressure_values)
                    print(f"   📊 Pressure: Avg {avg_pressure:.2f} kPa, Range {min_pressure:.2f}-{max_pressure:.2f} kPa")
                
                print(f"   📊 Total readings collected: {len(readings)}")
            else:
                print("   No readings collected")
    except Exception as e:
        print(f"❌ Error getting final readings: {e}")
    
    print("\n🎉 Sensor simulation demo completed!")
    print("\n💡 To see live data in the UI:")
    print("   1. Start the frontend: streamlit run integrated_app.py")
    print("   2. Go to the 'Safety Monitor' tab")
    print("   3. Create an experiment and start monitoring")

if __name__ == "__main__":
    test_sensor_simulation()