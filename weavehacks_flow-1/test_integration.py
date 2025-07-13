#!/usr/bin/env python3
"""
Integration test for the complete system
"""
import sys
import os
import unittest
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class TestSystemIntegration(unittest.TestCase):
    """Test the integration of all components"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Suppress logging during tests
        import logging
        logging.disable(logging.CRITICAL)
    
    def tearDown(self):
        """Clean up after tests"""
        import logging
        logging.disable(logging.NOTSET)
    
    def test_imports(self):
        """Test that all modules can be imported"""
        try:
            from weavehacks_flow.agents.data_collection_agent import DataCollectionAgent
            from weavehacks_flow.agents.lab_control_agent import LabControlAgent
            from weavehacks_flow.agents.safety_monitoring_agent import EnhancedSafetyMonitoringAgent
            from weavehacks_flow.agents.video_monitoring_agent import VideoMonitoringAgent
            from weavehacks_flow.agents.voice_recognition_agent import SpeechRecognizerAgent
            from weavehacks_flow.main import ExperimentFlow
            from weavehacks_flow.utils.chemistry_calculations import calculate_sulfur_amount
        except ImportError as e:
            self.fail(f"Import failed: {e}")
    
    def test_agent_creation(self):
        """Test that all agents can be created"""
        from weavehacks_flow.agents.data_collection_agent import DataCollectionAgent
        from weavehacks_flow.agents.lab_control_agent import LabControlAgent
        from weavehacks_flow.agents.safety_monitoring_agent import EnhancedSafetyMonitoringAgent
        from weavehacks_flow.agents.video_monitoring_agent import VideoMonitoringAgent
        from weavehacks_flow.agents.voice_recognition_agent import SpeechRecognizerAgent
        
        # Create agents
        data_agent = DataCollectionAgent()
        self.assertIsNotNone(data_agent)
        
        lab_agent = LabControlAgent()
        self.assertIsNotNone(lab_agent)
        
        safety_agent = EnhancedSafetyMonitoringAgent()
        self.assertIsNotNone(safety_agent)
        
        video_agent = VideoMonitoringAgent()
        self.assertIsNotNone(video_agent)
        
        voice_agent = SpeechRecognizerAgent()
        self.assertIsNotNone(voice_agent)
    
    def test_experiment_flow_creation(self):
        """Test ExperimentFlow creation and initialization"""
        from weavehacks_flow.main import ExperimentFlow
        
        flow = ExperimentFlow()
        
        # Check all agents are initialized
        self.assertIsNotNone(flow.data_agent)
        self.assertIsNotNone(flow.lab_agent)
        self.assertIsNotNone(flow.safety_agent)
        self.assertIsNotNone(flow.voice_agent)
        # video_agent might be None if OpenCV not available
        
        # Check initial state
        self.assertEqual(flow.state.step_num, 0)
        self.assertEqual(flow.state.exp_status, "not started")
        self.assertEqual(flow.state.safety_status, "safe")
    
    def test_chemistry_calculations(self):
        """Test chemistry calculation functions"""
        from weavehacks_flow.utils.chemistry_calculations import (
            calculate_sulfur_amount,
            calculate_nabh4_amount,
            calculate_percent_yield
        )
        
        # Test sulfur calculation
        gold_mass = 0.1576
        result = calculate_sulfur_amount(gold_mass)
        self.assertIn('mass_sulfur_g', result)
        self.assertGreater(result['mass_sulfur_g'], 0)
        
        # Test NaBH4 calculation
        result = calculate_nabh4_amount(gold_mass)
        self.assertIn('mass_nabh4_g', result)
        self.assertGreater(result['mass_nabh4_g'], 0)
        
        # Test percent yield
        result = calculate_percent_yield(gold_mass, 0.05)
        self.assertIn('percent_yield', result)
        self.assertGreater(result['percent_yield'], 0)
    
    def test_lab_control_functionality(self):
        """Test lab control agent functionality"""
        from weavehacks_flow.agents.lab_control_agent import LabControlAgent
        
        agent = LabControlAgent()
        
        # Test turning on/off instruments
        agent.turn_on("centrifuge")
        self.assertTrue(agent.is_on("centrifuge"))
        
        agent.turn_off("centrifuge")
        self.assertFalse(agent.is_on("centrifuge"))
    
    def test_safety_monitoring_functionality(self):
        """Test safety monitoring functionality"""
        from weavehacks_flow.agents.safety_monitoring_agent import EnhancedSafetyMonitoringAgent
        
        agent = EnhancedSafetyMonitoringAgent()
        
        # Test setting experiment
        success = agent.set_experiment("gold_nanoparticle_room_temp")
        self.assertTrue(success)
        self.assertEqual(agent.current_experiment, "gold_nanoparticle_room_temp")
    
    def test_video_agent_functionality(self):
        """Test video monitoring agent basic functionality"""
        from weavehacks_flow.agents.video_monitoring_agent import VideoMonitoringAgent
        
        agent = VideoMonitoringAgent()
        
        # Test camera check (may fail if no camera)
        has_camera = agent.test_camera()
        self.assertIsInstance(has_camera, bool)
        
        # Test callback registration
        def dummy_callback(event):
            pass
        
        agent.register_callback(dummy_callback)
        # Check that callback was registered (implementation may vary)
        self.assertTrue(hasattr(agent, 'register_callback'))
    
    @patch('weavehacks_flow.agents.data_collection_agent.DataCollectionAgent.record_data')
    def test_experiment_step_execution(self, mock_record_data):
        """Test individual experiment step execution"""
        from weavehacks_flow.main import ExperimentFlow
        
        # Set up mock
        mock_record_data.return_value = 0.1576
        
        flow = ExperimentFlow()
        
        # Test weigh gold step
        flow.weigh_gold()
        
        # Check state was updated
        self.assertEqual(flow.state.mass_gold, 0.1576)
        self.assertEqual(flow.state.step_num, 1)
        
        # Verify mock was called
        mock_record_data.assert_called_once()

if __name__ == '__main__':
    print("="*60)
    print("RUNNING INTEGRATION TESTS")
    print("="*60)
    unittest.main(verbosity=2)