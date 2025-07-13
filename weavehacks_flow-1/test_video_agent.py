#!/usr/bin/env python3
"""Test script to verify video_agent attribute exists in ExperimentFlow"""
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from weavehacks_flow.main import ExperimentFlow
    
    # Create an instance
    print("Creating ExperimentFlow instance...")
    ef = ExperimentFlow()
    
    # Check for video_agent
    has_video_agent = hasattr(ef, 'video_agent')
    print(f"Has video_agent attribute: {has_video_agent}")
    
    if has_video_agent:
        print(f"video_agent type: {type(ef.video_agent)}")
        print(f"video_agent class: {ef.video_agent.__class__.__name__}")
        
        # Test camera availability
        camera_available = ef.video_agent.test_camera()
        print(f"Camera available: {camera_available}")
        
        # List all agents
        print("\nAll agents in ExperimentFlow:")
        agents = [attr for attr in dir(ef) if attr.endswith('_agent')]
        for agent in agents:
            print(f"  - {agent}: {type(getattr(ef, agent))}")
    else:
        print("ERROR: video_agent attribute not found!")
        print("\nAvailable attributes:")
        for attr in dir(ef):
            if not attr.startswith('_'):
                print(f"  - {attr}")
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()