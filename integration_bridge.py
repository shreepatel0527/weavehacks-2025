#!/usr/bin/env python3
"""
Integration bridge between Prototype-1 and weavehacks_flow-1
Handles import path resolution and component integration
"""

import sys
import os
from pathlib import Path

# Add both prototype directories to Python path
current_dir = Path(__file__).parent
prototype1_dir = current_dir / "Prototype-1"
prototype_1_dir = current_dir / "Prototype_1"
weavehacks_flow_dir = current_dir / "weavehacks_flow-1" / "src"

sys.path.insert(0, str(prototype1_dir))
sys.path.insert(0, str(prototype_1_dir))
sys.path.insert(0, str(weavehacks_flow_dir))

try:
    # Try to import from Prototype-1 first, then Prototype_1
    try:
        from claude_interface import ClaudeInterface
        from gemini_interface import GeminiInterface
        from speech_recognition_module import SpeechRecognizer
        print("✅ Imported from Prototype-1")
    except ImportError:
        # Try alternative path
        sys.path.insert(0, str(current_dir / "rohit_prototype"))
        from claude_interface import ClaudeInterface
        from gemini_interface import GeminiInterface
        from speech_recognition_module import SpeechRecognizer
        print("✅ Imported from rohit_prototype")

    # Import weavehacks_flow-1 agents
    try:
        from weavehacks_flow.agents.data_collection_agent import DataCollectionAgent
        from weavehacks_flow.agents.lab_control_agent import LabControlAgent
        from weavehacks_flow.agents.safety_monitoring_agent import EnhancedSafetyMonitoringAgent
        print("✅ Imported from weavehacks_flow-1")
    except ImportError as e:
        print(f"⚠️ Could not import weavehacks_flow agents: {e}")
        # Create fallback agents
        import weave
        
        class DataCollectionAgent:
            @weave.op()
            def record_data(self, prompt):
                print(f"Data Collection: {prompt}")
                return 0.0
        
        class LabControlAgent:
            def turn_on(self, instrument):
                print(f"Lab Control: {instrument} ON")
            
            def turn_off(self, instrument):
                print(f"Lab Control: {instrument} OFF")
        
        class EnhancedSafetyMonitoringAgent:
            def monitor_parameters(self):
                print("Safety: Monitoring parameters")
                
            def is_safe(self):
                return True
                
            def notify_scientist(self):
                print("Safety: Alert sent")

except ImportError as e:
    print(f"❌ Import error: {e}")
    # Create minimal fallback classes
    class ClaudeInterface:
        def send_message(self, message):
            return False, "Claude not available"
        
        def test_connection(self):
            return False
    
    class GeminiInterface:
        def send_message(self, message):
            return False, "Gemini not available"
            
        def test_connection(self):
            return False
    
    class SpeechRecognizer:
        def __init__(self, model_size="base"):
            self.model_size = model_size
            
        def record_and_transcribe(self, duration=5.0):
            return False, "Speech recognition not available"
    
    class DataCollectionAgent:
        def record_data(self, prompt):
            return 0.0
    
    class LabControlAgent:
        def turn_on(self, instrument):
            pass
        def turn_off(self, instrument):
            pass
    
    class EnhancedSafetyMonitoringAgent:
        def monitor_parameters(self):
            pass
        def is_safe(self):
            return True
        def notify_scientist(self):
            pass

# Export all classes for use in integrated_app.py
__all__ = [
    'ClaudeInterface',
    'GeminiInterface', 
    'SpeechRecognizer',
    'DataCollectionAgent',
    'LabControlAgent',
    'EnhancedSafetyMonitoringAgent'
]