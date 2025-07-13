#!/usr/bin/env python3
"""
Simple test to validate the current implementation works
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing imports...")
try:
    from weavehacks_flow.agents.data_collection_agent import DataCollectionAgent
    from weavehacks_flow.agents.lab_control_agent import LabControlAgent
    from weavehacks_flow.agents.safety_monitoring_agent import EnhancedSafetyMonitoringAgent
    from weavehacks_flow.agents.video_monitoring_agent import VideoMonitoringAgent
    from weavehacks_flow.agents.voice_recognition_agent import SpeechRecognizerAgent
    print("✓ All agent imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

print("\nTesting agent instantiation...")
try:
    data_agent = DataCollectionAgent()
    print("✓ DataCollectionAgent created")
    
    lab_agent = LabControlAgent()
    print("✓ LabControlAgent created")
    
    safety_agent = EnhancedSafetyMonitoringAgent()
    print("✓ EnhancedSafetyMonitoringAgent created")
    
    video_agent = VideoMonitoringAgent()
    print("✓ VideoMonitoringAgent created")
    
    voice_agent = SpeechRecognizerAgent()
    print("✓ SpeechRecognizerAgent created")
    
except Exception as e:
    print(f"✗ Agent creation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTesting basic agent methods...")
try:
    # Test lab control agent
    lab_agent.turn_on("centrifuge")
    assert lab_agent.is_on("centrifuge"), "Centrifuge should be on"
    lab_agent.turn_off("centrifuge")
    assert not lab_agent.is_on("centrifuge"), "Centrifuge should be off"
    print("✓ LabControlAgent methods work")
    
    # Test safety monitoring
    safety_agent.set_experiment("gold_nanoparticle_room_temp")
    print("✓ SafetyMonitoringAgent experiment set")
    
    # Test video agent
    has_camera = video_agent.test_camera()
    print(f"✓ VideoMonitoringAgent camera test: {'available' if has_camera else 'not available'}")
    
except Exception as e:
    print(f"✗ Method test error: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting chemistry calculations...")
try:
    from weavehacks_flow.utils.chemistry_calculations import (
        calculate_sulfur_amount,
        calculate_nabh4_amount,
        calculate_percent_yield
    )
    
    # Test sulfur calculation
    result = calculate_sulfur_amount(0.1576)
    assert 'mass_sulfur_g' in result, "Should return sulfur mass"
    print(f"✓ Sulfur calculation: {result['mass_sulfur_g']:.4f}g")
    
    # Test NaBH4 calculation
    result = calculate_nabh4_amount(0.1576)
    assert 'mass_nabh4_g' in result, "Should return NaBH4 mass"
    print(f"✓ NaBH4 calculation: {result['mass_nabh4_g']:.4f}g")
    
    # Test percent yield
    result = calculate_percent_yield(0.1576, 0.05)
    assert 'percent_yield' in result, "Should return percent yield"
    print(f"✓ Percent yield: {result['percent_yield']:.2f}%")
    
except Exception as e:
    print(f"✗ Chemistry calculation error: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting main experiment flow...")
try:
    from weavehacks_flow.main import ExperimentFlow
    
    flow = ExperimentFlow()
    print("✓ ExperimentFlow created successfully")
    
    # Check all agents are initialized
    assert hasattr(flow, 'data_agent'), "Should have data_agent"
    assert hasattr(flow, 'lab_agent'), "Should have lab_agent"
    assert hasattr(flow, 'safety_agent'), "Should have safety_agent"
    assert hasattr(flow, 'voice_agent'), "Should have voice_agent"
    assert hasattr(flow, 'video_agent'), "Should have video_agent"
    print("✓ All agents initialized in ExperimentFlow")
    
except Exception as e:
    print(f"✗ ExperimentFlow error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("BASIC FUNCTIONALITY TEST COMPLETE")
print("="*50)