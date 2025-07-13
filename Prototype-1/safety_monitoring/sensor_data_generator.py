import csv
import time
import datetime
import random

FILE_NAME = 'sensor_data.csv'

def generate_data():
    """
    Simulates a sensor writing data (temperature and pressure) to a CSV file.
    
    This function appends a new row with a timestamp, a random temperature,
    and a random pressure reading to the specified file every few seconds.
    """
    print(f"Starting to generate sensor data in '{FILE_NAME}'.")
    print("Press Ctrl+C to stop.")
    
    # Write a header row. This will overwrite the file if it exists.
    with open(FILE_NAME, 'w', newline='') as f_write:
        writer = csv.writer(f_write)
        writer.writerow(['timestamp', 'temperature_celsius', 'pressure_kpa'])

    try:
        # Loop to continuously append new data
        while True:
            with open(FILE_NAME, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Get current timestamp
                timestamp = datetime.datetime.now().isoformat()
                
                # Simulate a temperature reading (in Celsius)
                temperature = 25.0 + round(random.uniform(-2.0, 3.0), 2)
                
                # Simulate a pressure reading (in kilopascals, kPa)
                # Sea level pressure is ~101.3 kPa. We'll fluctuate around that.
                pressure = 101.3 + round(random.uniform(-1.0, 1.5), 2)
                
                # Write the new data row
                writer.writerow([timestamp, temperature, pressure])
                print(f"Wrote: {timestamp}, {temperature}°C, {pressure} kPa")
                
            # Wait for a random interval before the next reading
            time.sleep(random.randint(2, 6))
            
    except KeyboardInterrupt:
        print("\nData generation stopped.")

if __name__ == "__main__":
    generate_data()


