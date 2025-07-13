import csv
import time
import os

FILE_NAME = 'sensor_data.csv'
HIGH_TEMP_THRESHOLD = 27.0      # Celsius
HIGH_PRESSURE_THRESHOLD = 102.5 # kilopascals (kPa)
POLL_INTERVAL_SECONDS = 5

def process_data(data):
    """
    Processes a single data point and checks for alarm conditions.
    
    Args:
        data (dict): A dictionary containing the sensor reading.
    """
    temp = data['temperature_celsius']
    pressure = data['pressure_kpa']
    print(f"Read: Timestamp={data['timestamp']}, Temp={temp}°C, Pressure={pressure} kPa")
    
    # Check if temperature exceeds the alarm threshold
    if temp > HIGH_TEMP_THRESHOLD:
        print(f"  -> ALARM! Temperature {temp}°C is above the threshold of {HIGH_TEMP_THRESHOLD}°C!")
        return "unsafe"
    
    # Check if pressure exceeds the alarm threshold
    if pressure > HIGH_PRESSURE_THRESHOLD:
        print(f"  -> ALARM! Pressure {pressure} kPa is above the threshold of {HIGH_PRESSURE_THRESHOLD} kPa!")
        return "unsafe"
    
    return "safe"

def monitor_sensor_file():
    """
    Monitors the sensor data file for new entries and processes them.
    
    This function periodically checks the file and only reads new lines
    that have been added since the last check using the file's seek position.
    """
    print(f"Monitoring '{FILE_NAME}' for new data every {POLL_INTERVAL_SECONDS} seconds.")
    print(f"Alarm thresholds: Temp > {HIGH_TEMP_THRESHOLD}°C, Pressure > {HIGH_PRESSURE_THRESHOLD} kPa. Press Ctrl+C to stop.")
    
    # We use the file's byte position to track what we've read.
    # This is more robust than counting lines, especially if the file is replaced.
    last_read_position = 0
    
    try:
        while True:
            try:
                with open(FILE_NAME, 'r', newline='') as csvfile:
                    # Move to the last known read position
                    csvfile.seek(last_read_position)
                    reader = csv.reader(csvfile)
                    
                    # Read all new lines from this position
                    for row in reader:
                        # Skip the header row if we are at the beginning of the file
                        if last_read_position == 0 and row == ['timestamp', 'temperature_celsius', 'pressure_kpa']:
                            continue
                        
                        if not row: continue # Skip potential empty rows

                        try:
                           # Unpack the three values from the row
                           timestamp, temp_str, pressure_str = row
                           data_point = {
                               "timestamp": timestamp,
                               "temperature_celsius": float(temp_str),
                               "pressure_kpa": float(pressure_str)
                           }
                           process_data(data_point)
                        except (ValueError, IndexError):
                           # This handles rows that don't have exactly 3 items or can't be converted to float
                           print(f"  -> Skipping malformed row: {row}")
                           continue
                    
                    # Update our position to the end of the file for the next check
                    last_read_position = csvfile.tell()

            except FileNotFoundError:
                print(f"Waiting for data file '{FILE_NAME}' to be created...")
                # If file is not found, reset position for when it's created
                last_read_position = 0
            except Exception as e:
                print(f"An error occurred: {e}")
            
            # Wait for the specified interval before checking again
            time.sleep(POLL_INTERVAL_SECONDS)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

if __name__ == "__main__":
    monitor_sensor_file()


