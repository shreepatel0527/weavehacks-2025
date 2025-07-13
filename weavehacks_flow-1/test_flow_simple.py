#!/usr/bin/env python3
"""
Simple test of experiment flow with mocked inputs
"""
import sys
import os
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from weavehacks_flow.main import ExperimentFlow

# Mock the data input to avoid interactive prompts
def mock_record_data(prompt, use_voice=False):
    """Mock data collection to return test values"""
    data_map = {
        "gold": 0.1576,
        "water": 10.0,
        "TOAB": 0.25,
        "toluene": 10.0,
        "sulfur": 0.1659,
        "NaBH4": 0.1514,
        "ice-cold": 7.0,
        "final": 0.05
    }
    
    # Find which reagent is being asked for
    for key, value in data_map.items():
        if key.lower() in prompt.lower():
            print(f"[MOCK] {prompt} -> {value}")
            return value
    
    # Default value
    print(f"[MOCK] {prompt} -> 1.0 (default)")
    return 1.0

print("="*60)
print("TESTING EXPERIMENT FLOW WITH MOCKED INPUTS")
print("="*60)

try:
    # Create experiment flow
    flow = ExperimentFlow()
    
    # Mock the data collection
    flow.data_agent.record_data = Mock(side_effect=mock_record_data)
    
    # Test each step individually
    print("\nTesting individual steps...")
    
    # Step 1: Initialize
    print("\n1. Initialize experiment")
    flow.initialize_experiment()
    
    # Step 2: Weigh gold
    print("\n2. Weigh gold")
    flow.weigh_gold()
    print(f"   Gold mass: {flow.state.mass_gold}g")
    
    # Step 3: Calculate sulfur
    print("\n3. Calculate sulfur amount")
    result = flow.calculate_sulfur_amount()
    print(f"   Sulfur needed: {result:.4f}g")
    
    # Step 4: Weigh sulfur
    print("\n4. Weigh sulfur")
    flow.weigh_sulfur()
    print(f"   Sulfur mass: {flow.state.mass_sulfur}g")
    
    print("\n" + "="*60)
    print("INDIVIDUAL STEPS TEST SUCCESSFUL!")
    print("="*60)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)