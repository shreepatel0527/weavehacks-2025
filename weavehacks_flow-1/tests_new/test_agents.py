"""
Test suite for WeaveHacks Lab Assistant agents
"""
import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from weavehacks_flow.agents.data_collection_agent import DataCollectionAgent
from weavehacks_flow.agents.lab_control_agent import LabControlAgent
from weavehacks_flow.agents.safety_monitoring_agent import EnhancedSafetyMonitoringAgent as SafetyMonitoringAgent
from weavehacks_flow.utils.chemistry_calculations import (
    calculate_sulfur_amount,
    calculate_nabh4_amount,
    calculate_percent_yield,
    calculate_toab_ratio
)

class TestDataCollectionAgent(unittest.TestCase):
    def setUp(self):
        self.agent = DataCollectionAgent()
    
    @patch('builtins.input', return_value='0.1576')
    @patch('wandb.log')
    def test_record_data(self, mock_log, mock_input):
        """Test data recording functionality"""
        result = self.agent.record_data("Enter mass")
        self.assertEqual(result, 0.1576)
        mock_log.assert_called_once()
    
    @patch('builtins.input', return_value='Gold chloride')
    @patch('wandb.log')
    def test_clarify_reagent(self, mock_log, mock_input):
        """Test reagent clarification"""
        result = self.agent.clarify_reagent()
        self.assertEqual(result, 'Gold chloride')
        mock_log.assert_called_once()

class TestLabControlAgent(unittest.TestCase):
    def setUp(self):
        self.agent = LabControlAgent()
    
    @patch('wandb.log')
    def test_turn_on_instrument(self, mock_log):
        """Test turning on instruments"""
        self.agent.turn_on("centrifuge")
        self.assertTrue(self.agent.is_on("centrifuge"))
        self.assertEqual(mock_log.call_count, 2)  # turn_on and is_on
    
    @patch('wandb.log')
    def test_turn_off_instrument(self, mock_log):
        """Test turning off instruments"""
        self.agent.turn_on("UV-Vis")
        self.agent.turn_off("UV-Vis")
        self.assertFalse(self.agent.is_on("UV-Vis"))

class TestSafetyMonitoringAgent(unittest.TestCase):
    def setUp(self):
        self.agent = SafetyMonitoringAgent()
    
    def test_load_safety_config(self):
        """Test safety configuration loading"""
        # Should have loaded config or defaults
        self.assertIn('temperature', self.agent.safety_thresholds)
        self.assertIn('pressure', self.agent.safety_thresholds)
        self.assertIn('nitrogen', self.agent.safety_thresholds)
        self.assertIn('oxygen', self.agent.safety_thresholds)
    
    @patch('wandb.log')
    def test_monitor_parameters_simulated(self, mock_log):
        """Test parameter monitoring with simulated data"""
        self.agent.monitor_parameters(use_real_data=False)
        
        # Check that parameters were set
        self.assertTrue(hasattr(self.agent, 'current_temperature'))
        self.assertTrue(hasattr(self.agent, 'current_pressure'))
        self.assertTrue(hasattr(self.agent, 'current_nitrogen'))
        self.assertTrue(hasattr(self.agent, 'current_oxygen'))
        
        # Check logging
        mock_log.assert_called()
    
    @patch('wandb.log')
    def test_is_safe(self, mock_log):
        """Test safety checking"""
        # Set safe values
        self.agent.current_temperature = 25
        self.agent.current_pressure = 101
        self.agent.current_nitrogen = 80
        self.agent.current_oxygen = 21
        
        self.assertTrue(self.agent.is_safe())
        
        # Set unsafe temperature
        self.agent.current_temperature = 50
        self.assertFalse(self.agent.is_safe())
    
    def test_check_warning_levels(self):
        """Test warning level detection"""
        # Set values near warning threshold
        self.agent.current_temperature = 33  # Near max of 35
        self.agent.current_pressure = 108    # Near max of 110
        self.agent.current_nitrogen = 80     # Within safe range
        self.agent.current_oxygen = 21       # Within safe range
        
        warnings = self.agent.check_warning_levels()
        self.assertTrue(any('Temperature' in w for w in warnings))
        self.assertTrue(any('Pressure' in w for w in warnings))

class TestChemistryCalculations(unittest.TestCase):
    def test_calculate_sulfur_amount(self):
        """Test sulfur calculation"""
        gold_mass = 0.1576  # g
        result = calculate_sulfur_amount(gold_mass)
        
        self.assertAlmostEqual(result['mass_sulfur_g'], 0.1659, places=3)
        self.assertEqual(result['equivalents'], 3)
        self.assertGreater(result['moles_gold'], 0)
    
    def test_calculate_nabh4_amount(self):
        """Test NaBH4 calculation"""
        gold_mass = 0.1576  # g
        result = calculate_nabh4_amount(gold_mass)
        
        self.assertAlmostEqual(result['mass_nabh4_g'], 0.1514, places=3)
        self.assertEqual(result['equivalents'], 10)
        self.assertGreater(result['moles_nabh4'], 0)
    
    def test_calculate_percent_yield(self):
        """Test percent yield calculation"""
        gold_mass = 0.1576  # g
        actual_yield = 0.045  # g
        result = calculate_percent_yield(gold_mass, actual_yield)
        
        self.assertGreater(result['percent_yield'], 0)
        self.assertLess(result['percent_yield'], 100)
        self.assertAlmostEqual(result['gold_content_g'], 0.0788, places=3)
    
    def test_calculate_toab_ratio(self):
        """Test TOAB to gold ratio calculation"""
        gold_mass = 0.1576  # g
        toab_mass = 0.25    # g
        result = calculate_toab_ratio(gold_mass, toab_mass)
        
        self.assertGreater(result['toab_to_gold_ratio'], 0)
        self.assertAlmostEqual(result['toab_to_gold_ratio'], 1.14, places=2)

class TestExperimentFlow(unittest.TestCase):
    @patch('weavehacks_flow.main.DataCollectionAgent')
    @patch('weavehacks_flow.main.LabControlAgent')
    @patch('weavehacks_flow.main.SafetyMonitoringAgent')
    @patch('weave.init')
    def test_flow_initialization(self, mock_weave_init, mock_safety, mock_lab, mock_data):
        """Test experiment flow initialization"""
        from weavehacks_flow.main import ExperimentFlow
        
        flow = ExperimentFlow()
        
        # Check that agents were created
        self.assertIsNotNone(flow.data_agent)
        self.assertIsNotNone(flow.lab_agent)
        self.assertIsNotNone(flow.safety_agent)
        
        # Check initial state
        self.assertEqual(flow.state.step_num, 0)
        self.assertEqual(flow.state.exp_status, "not started")
        self.assertEqual(flow.state.safety_status, "safe")

if __name__ == '__main__':
    unittest.main()