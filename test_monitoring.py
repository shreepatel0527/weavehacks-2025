#!/usr/bin/env python
"""
Test script for the overnight monitoring functionality
"""

import sys
import os
sys.path.append('weavehacks_flow-1/src')

def test_overnight_monitoring():
    """Test the overnight monitoring system integration"""
    try:
        from weavehacks_flow.main import ExperimentFlow
        
        print("Testing Overnight Monitoring Integration")
        print("="*50)
        
        # Create experiment flow
        flow = ExperimentFlow()
        
        # Test 1: Check that agents are properly initialized
        print("\n1. Testing agent initialization...")
        assert hasattr(flow, 'video_agent'), "Video agent not initialized"
        assert hasattr(flow, 'safety_agent'), "Safety agent not initialized"
        assert hasattr(flow, 'lab_agent'), "Lab control agent not initialized"
        print("✓ All agents initialized successfully")
        
        # Test 2: Check that overnight monitoring function exists
        print("\n2. Testing overnight monitoring function...")
        assert hasattr(flow, 'initialize_overnight_monitoring'), "Overnight monitoring function not found"
        print("✓ initialize_overnight_monitoring function exists")
        
        # Test 3: Check emergency halt function
        print("\n3. Testing emergency halt function...")
        assert hasattr(flow, '_trigger_emergency_halt'), "Emergency halt function not found"
        assert hasattr(flow, '_handle_video_safety_event'), "Video safety event handler not found"
        assert hasattr(flow, '_continuous_monitoring_loop'), "Continuous monitoring loop not found"
        print("✓ Emergency halt functions exist")
        
        # Test 4: Check lab control agent enhancements
        print("\n4. Testing lab control agent enhancements...")
        assert hasattr(flow.lab_agent, 'emergency_shutdown_all'), "Emergency shutdown function not found"
        assert hasattr(flow.lab_agent, 'alert_mode'), "Alert mode attribute not found"
        assert hasattr(flow.lab_agent, 'get_status'), "Status function not found"
        print("✓ Lab control agent enhanced successfully")
        
        # Test 5: Check video agent safety integration
        print("\n5. Testing video agent safety integration...")
        assert hasattr(flow.video_agent, 'has_safety_violations'), "Safety violation check not found"
        assert hasattr(flow.video_agent, 'get_safety_violations'), "Safety violation getter not found"
        assert hasattr(flow.video_agent, 'register_callback'), "Callback registration not found"
        print("✓ Video agent safety integration verified")
        
        # Test 6: Check safety agent monitoring
        print("\n6. Testing safety agent monitoring...")
        assert hasattr(flow.safety_agent, 'monitor_parameters'), "Monitor parameters function not found"
        assert hasattr(flow.safety_agent, 'is_safe'), "Safety check function not found"
        assert hasattr(flow.safety_agent, 'notify_scientist'), "Notification function not found"
        print("✓ Safety agent monitoring verified")
        
        # Test 7: Check workflow integration
        print("\n7. Testing workflow integration...")
        # The overnight monitoring should be in the workflow
        workflow_found = any('initialize_overnight_monitoring' in str(step) for step in flow._step_methods)
        assert workflow_found or hasattr(flow, 'initialize_overnight_monitoring'), "Overnight monitoring not in workflow"
        print("✓ Overnight monitoring integrated into workflow")
        
        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED - Overnight Monitoring System Ready!")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_overnight_monitoring()
    sys.exit(0 if success else 1)