#!/usr/bin/env python3
"""
WeaveHacks 2025 - Integration Test Suite
Tests the connection between frontend and backend components
"""

import unittest
import requests
import time
import subprocess
import threading
from pathlib import Path
import weave

# Initialize Weave for test tracking
try:
    weave.init('weavehacks-integration-tests')
    print("Weave initialized for testing")
except Exception as e:
    print(f"Weave init failed (optional): {e}")

class IntegrationTestSuite(unittest.TestCase):
    """Test suite for full-stack integration"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.api_base = "http://localhost:8000"
        cls.test_experiment_id = "test_exp_001"
        
    def setUp(self):
        """Set up each test"""
        # Clean up any existing test experiments
        try:
            requests.delete(f"{self.api_base}/experiments/{self.test_experiment_id}")
        except:
            pass
    
    @weave.op()
    def test_01_backend_health(self):
        """Test backend API health check"""
        try:
            response = requests.get(f"{self.api_base}/health", timeout=5)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("status", data)
            self.assertEqual(data["status"], "healthy")
            print("✅ Backend health check passed")
        except requests.exceptions.ConnectionError:
            self.fail("❌ Backend API not available. Start with: uvicorn backend.main:app --reload")
    
    @weave.op()
    def test_02_experiment_creation(self):
        """Test experiment creation via API"""
        response = requests.post(f"{self.api_base}/experiments", 
                               params={"experiment_id": self.test_experiment_id})
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["experiment_id"], self.test_experiment_id)
        self.assertEqual(data["status"], "not_started")
        self.assertEqual(data["step_num"], 0)
        print("✅ Experiment creation test passed")
    
    @weave.op()
    def test_03_experiment_retrieval(self):
        """Test experiment data retrieval"""
        # First create an experiment
        requests.post(f"{self.api_base}/experiments", 
                     params={"experiment_id": self.test_experiment_id})
        
        # Then retrieve it
        response = requests.get(f"{self.api_base}/experiments/{self.test_experiment_id}")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["experiment_id"], self.test_experiment_id)
        print("✅ Experiment retrieval test passed")
    
    @weave.op()
    def test_04_data_recording(self):
        """Test data recording via API"""
        # Create experiment first
        requests.post(f"{self.api_base}/experiments", 
                     params={"experiment_id": self.test_experiment_id})
        
        # Record test data
        test_data = {
            "experiment_id": self.test_experiment_id,
            "data_type": "mass",
            "compound": "HAuCl4·3H2O",
            "value": 0.1576,
            "units": "g"
        }
        
        response = requests.post(f"{self.api_base}/data", json=test_data)
        self.assertEqual(response.status_code, 200)
        
        # Verify experiment was updated
        exp_response = requests.get(f"{self.api_base}/experiments/{self.test_experiment_id}")
        exp_data = exp_response.json()
        self.assertAlmostEqual(exp_data["mass_gold"], 0.1576, places=4)
        print("✅ Data recording test passed")
    
    @weave.op()
    def test_05_chemistry_calculations(self):
        """Test chemistry calculations via API"""
        # Create experiment and add gold mass
        requests.post(f"{self.api_base}/experiments", 
                     params={"experiment_id": self.test_experiment_id})
        
        # Test sulfur calculation
        calc_data = {
            "experiment_id": self.test_experiment_id,
            "gold_mass": 0.1576
        }
        
        response = requests.post(f"{self.api_base}/calculations/sulfur-amount", json=calc_data)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("mass_sulfur_needed_g", data)
        self.assertGreater(data["mass_sulfur_needed_g"], 0)
        print("✅ Chemistry calculations test passed")
    
    @weave.op()
    def test_06_safety_alerts(self):
        """Test safety alert system"""
        # Create experiment first
        requests.post(f"{self.api_base}/experiments", 
                     params={"experiment_id": self.test_experiment_id})
        
        # Create test safety alert
        alert_data = {
            "experiment_id": self.test_experiment_id,
            "parameter": "temperature",
            "value": 85.0,
            "threshold": 80.0,
            "severity": "warning"
        }
        
        response = requests.post(f"{self.api_base}/safety/alert", json=alert_data)
        self.assertEqual(response.status_code, 200)
        
        # Retrieve alerts
        alerts_response = requests.get(f"{self.api_base}/safety/alerts")
        self.assertEqual(alerts_response.status_code, 200)
        
        alerts = alerts_response.json()
        self.assertGreater(len(alerts), 0)
        print("✅ Safety alerts test passed")
    
    @weave.op()
    def test_07_integration_bridge(self):
        """Test integration bridge imports"""
        try:
            from integration_bridge import (
                ClaudeInterface,
                GeminiInterface,
                SpeechRecognizer,
                DataCollectionAgent,
                LabControlAgent,
                EnhancedSafetyMonitoringAgent
            )
            
            # Test instantiation
            data_agent = DataCollectionAgent()
            lab_agent = LabControlAgent()
            safety_agent = EnhancedSafetyMonitoringAgent()
            
            self.assertIsNotNone(data_agent)
            self.assertIsNotNone(lab_agent)
            self.assertIsNotNone(safety_agent)
            print("✅ Integration bridge test passed")
            
        except ImportError as e:
            self.fail(f"❌ Integration bridge import failed: {e}")
    
    @weave.op()
    def test_08_agent_functionality(self):
        """Test agent basic functionality"""
        from integration_bridge import DataCollectionAgent, LabControlAgent, EnhancedSafetyMonitoringAgent
        
        # Test data collection agent
        data_agent = DataCollectionAgent()
        result = data_agent.record_data("Test measurement")
        self.assertIsNotNone(result)
        
        # Test lab control agent
        lab_agent = LabControlAgent()
        lab_agent.turn_on("centrifuge")  # Should not raise exception
        lab_agent.turn_off("centrifuge")  # Should not raise exception
        
        # Test safety monitoring agent
        safety_agent = EnhancedSafetyMonitoringAgent()
        safety_agent.monitor_parameters()  # Should not raise exception
        is_safe = safety_agent.is_safe()
        self.assertIsInstance(is_safe, bool)
        
        print("✅ Agent functionality test passed")
    
    def tearDown(self):
        """Clean up after each test"""
        try:
            requests.delete(f"{self.api_base}/experiments/{self.test_experiment_id}")
        except:
            pass


class PerformanceTestSuite(unittest.TestCase):
    """Performance tests for the integrated system"""
    
    @classmethod
    def setUpClass(cls):
        cls.api_base = "http://localhost:8000"
    
    @weave.op()
    def test_response_times(self):
        """Test API response times"""
        import time
        
        # Test experiment creation time
        start_time = time.time()
        response = requests.post(f"{self.api_base}/experiments", 
                               params={"experiment_id": "perf_test_001"})
        creation_time = time.time() - start_time
        
        self.assertLess(creation_time, 1.0, "Experiment creation should be under 1 second")
        self.assertEqual(response.status_code, 200)
        
        # Test data recording time
        test_data = {
            "experiment_id": "perf_test_001",
            "data_type": "mass",
            "compound": "HAuCl4·3H2O",
            "value": 0.1576,
            "units": "g"
        }
        
        start_time = time.time()
        response = requests.post(f"{self.api_base}/data", json=test_data)
        recording_time = time.time() - start_time
        
        self.assertLess(recording_time, 0.5, "Data recording should be under 0.5 seconds")
        self.assertEqual(response.status_code, 200)
        
        print(f"✅ Performance test passed - Creation: {creation_time:.3f}s, Recording: {recording_time:.3f}s")
        
        # Cleanup
        requests.delete(f"{self.api_base}/experiments/perf_test_001")


def run_tests():
    """Run all integration tests"""
    print("🧪 Starting WeaveHacks Integration Test Suite")
    print("=" * 50)
    
    # Check if backend is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code != 200:
            print("❌ Backend API not healthy")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Backend API not running. Start with:")
        print("   cd backend && uvicorn main:app --reload")
        return False
    
    # Run integration tests
    suite = unittest.TestLoader().loadTestsFromTestCase(IntegrationTestSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n🎉 All integration tests passed!")
        
        # Run performance tests if integration tests pass
        print("\n🚀 Running performance tests...")
        perf_suite = unittest.TestLoader().loadTestsFromTestCase(PerformanceTestSuite)
        perf_result = runner.run(perf_suite)
        
        if perf_result.wasSuccessful():
            print("\n⚡ All performance tests passed!")
            return True
        else:
            print("\n⚠️ Some performance tests failed")
            return False
    else:
        print("\n❌ Some integration tests failed")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)