# This file contains utility functions that assist with various tasks throughout the project.

def format_data(data):
    """Formats the data for better readability."""
    return f"Formatted Data: {data}"

def log_event(event):
    """Logs an event with a timestamp."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {event}")

def validate_input(value, expected_type):
    """Validates the input value against the expected type."""
    if not isinstance(value, expected_type):
        raise ValueError(f"Expected value of type {expected_type}, got {type(value)}")
    return True