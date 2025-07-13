#!/usr/bin/env python3
"""
Test script to verify video monitoring integration
"""
import sys
import os
from pathlib import Path

# Add the necessary paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'weavehacks_flow-1/src'))

def test_video_agent_import():
    """Test importing the video monitoring agent"""
    try:
        from weavehacks_flow.agents.video_monitoring_agent import VideoMonitoringAgent
        print("✅ Video monitoring agent import successful")
        return True
    except Exception as e:
        print(f"❌ Video monitoring agent import failed: {e}")
        return False

def test_video_agent_creation():
    """Test creating a video monitoring agent"""
    try:
        from weavehacks_flow.agents.video_monitoring_agent import VideoMonitoringAgent
        agent = VideoMonitoringAgent()
        print("✅ Video monitoring agent creation successful")
        return True, agent
    except Exception as e:
        print(f"❌ Video monitoring agent creation failed: {e}")
        return False, None

def test_video_agent_functionality(agent):
    """Test basic video monitoring agent functionality"""
    try:
        # Test event summary
        summary = agent.get_event_summary()
        print(f"✅ Event summary: {summary}")
        
        # Test start monitoring (will fail gracefully without camera)
        result = agent.start_monitoring()
        print(f"✅ Start monitoring result: {result['status']}")
        
        # Test stop monitoring
        stop_result = agent.stop_monitoring()
        print(f"✅ Stop monitoring result: {stop_result['status']}")
        
        return True
    except Exception as e:
        print(f"❌ Video monitoring agent functionality test failed: {e}")
        return False

def test_main_flow_import():
    """Test importing the updated main flow"""
    try:
        from weavehacks_flow.main import ExperimentFlow
        print("✅ Main flow import successful")
        return True
    except Exception as e:
        print(f"❌ Main flow import failed: {e}")
        return False

def test_unified_ui_syntax():
    """Test unified lab assistant syntax"""
    try:
        import ast
        with open('unified_lab_assistant.py', 'r') as f:
            code = f.read()
        ast.parse(code)
        print("✅ Unified lab assistant syntax is valid")
        return True
    except Exception as e:
        print(f"❌ Unified lab assistant syntax error: {e}")
        return False

def test_requirements():
    """Test that required packages are available"""
    results = []
    
    # Test core packages
    packages = [
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('weave', 'Weave'),
        ('streamlit', 'Streamlit'),
        ('datetime', 'datetime'),
        ('pathlib', 'pathlib'),
        ('threading', 'threading'),
        ('queue', 'queue')
    ]
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name} available")
            results.append(True)
        except ImportError:
            print(f"❌ {name} not available")
            results.append(False)
    
    return all(results)

def main():
    """Run all tests"""
    print("🧪 Running Video Monitoring Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Requirements Check", test_requirements),
        ("Video Agent Import", test_video_agent_import),
        ("Video Agent Creation", lambda: test_video_agent_creation()[0]),
        ("Main Flow Import", test_main_flow_import),
        ("Unified UI Syntax", test_unified_ui_syntax),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append(False)
    
    # Test video agent functionality if creation succeeded
    if len(results) >= 2 and results[1]:  # Video Agent Creation passed
        print(f"\n🔍 Video Agent Functionality:")
        try:
            from weavehacks_flow.agents.video_monitoring_agent import VideoMonitoringAgent
            agent = VideoMonitoringAgent()
            func_result = test_video_agent_functionality(agent)
            results.append(func_result)
        except Exception as e:
            print(f"❌ Video Agent Functionality failed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Video monitoring integration is successful.")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed. Review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())