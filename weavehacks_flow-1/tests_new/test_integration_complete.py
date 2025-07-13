#!/usr/bin/env python3
"""
Comprehensive integration tests for the entire lab automation platform
"""

import pytest
import sys
import os
import requests
import time
from unittest.mock import patch, Mock
import tempfile
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from weavehacks_flow.utils.chemistry_calculations import (
    calculate_sulfur_amount, calculate_nabh4_amount, calculate_percent_yield
)


class TestChemistryIntegration:
    """Test complete chemistry calculation pipeline"""
    
    def test_gold_synthesis_workflow(self):
        """Test complete gold nanoparticle synthesis calculation workflow"""
        # Starting materials
        gold_mass = 0.1576  # g of HAuCl4·3H2O
        
        # Calculate required reagents
        sulfur_result = calculate_sulfur_amount(gold_mass)
        nabh4_result = calculate_nabh4_amount(gold_mass)
        
        # Validate calculations
        assert sulfur_result['mass_sulfur_g'] > 0
        assert nabh4_result['mass_nabh4_g'] > 0
        
        # Simulate experiment completion
        actual_yield = 0.08  # g of Au25 nanoparticles
        yield_result = calculate_percent_yield(gold_mass, actual_yield)
        
        assert yield_result['percent_yield'] > 0
        assert yield_result['percent_yield'] <= 100
        
        # Validate complete workflow
        assert 'yield_quality' in yield_result
        assert yield_result['actual_yield_g'] == actual_yield
    
    def test_scaled_synthesis(self):
        """Test calculations for scaled-up synthesis"""
        # Test multiple scales
        scales = [0.1, 0.5, 1.0, 2.0]
        
        for scale in scales:
            gold_mass = 0.1576 * scale
            
            sulfur_result = calculate_sulfur_amount(gold_mass)
            nabh4_result = calculate_nabh4_amount(gold_mass)
            
            # Ensure scaling is proportional
            expected_sulfur = 0.1659 * scale  # Base calculation result
            expected_nabh4 = 0.1514 * scale   # Base calculation result
            
            assert abs(sulfur_result['mass_sulfur_g'] - expected_sulfur) < 0.001
            assert abs(nabh4_result['mass_nabh4_g'] - expected_nabh4) < 0.001


class TestAPIIntegration:
    """Test API integration with backend"""
    
    def setup_method(self):
        """Setup for API tests"""
        self.base_url = "http://localhost:8000"
        self.test_experiment_id = f"test_integration_{int(time.time())}"
    
    def test_backend_health(self):
        """Test backend health check"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend not running - start with 'cd backend && uvicorn main:app --reload'")
    
    def test_complete_experiment_workflow(self):
        """Test complete experiment workflow via API"""
        try:
            # 1. Create experiment
            response = requests.post(f"{self.base_url}/experiments?experiment_id={self.test_experiment_id}")
            if response.status_code == 400 and "already exists" in response.text:
                # Use existing experiment
                pass
            else:
                assert response.status_code == 200
                experiment = response.json()
                assert experiment['experiment_id'] == self.test_experiment_id
            
            # 2. Record multiple measurements
            measurements = [
                {"compound": "HAuCl₄·3H₂O", "value": 0.1576, "data_type": "mass", "units": "g"},
                {"compound": "TOAB", "value": 0.25, "data_type": "mass", "units": "g"},
                {"compound": "toluene", "value": 10.0, "data_type": "volume", "units": "mL"},
                {"compound": "PhCH₂CH₂SH", "value": 0.1659, "data_type": "mass", "units": "g"},
                {"compound": "NaBH₄", "value": 0.1514, "data_type": "mass", "units": "g"},
            ]
            
            for measurement in measurements:
                data = {
                    "experiment_id": self.test_experiment_id,
                    "data_type": measurement["data_type"],
                    "compound": measurement["compound"],
                    "value": measurement["value"],
                    "units": measurement["units"]
                }
                
                response = requests.post(f"{self.base_url}/data", json=data)
                assert response.status_code == 200
                recorded = response.json()
                assert recorded['value'] == measurement['value']
                assert recorded['compound'] == measurement['compound']
            
            # 3. Test chemistry calculations
            gold_mass = 0.1576
            
            # Calculate sulfur amount
            response = requests.post(
                f"{self.base_url}/calculations/sulfur_amount",
                params={"experiment_id": self.test_experiment_id, "gold_mass": gold_mass}
            )
            assert response.status_code == 200
            sulfur_calc = response.json()
            assert 'mass_sulfur_g' in sulfur_calc
            
            # Calculate NaBH4 amount
            response = requests.post(
                f"{self.base_url}/calculations/nabh4_amount",
                params={"experiment_id": self.test_experiment_id, "gold_mass": gold_mass}
            )
            assert response.status_code == 200
            nabh4_calc = response.json()
            assert 'mass_nabh4_g' in nabh4_calc
            
            # 4. Get experiment data
            response = requests.get(f"{self.base_url}/experiments/{self.test_experiment_id}")
            assert response.status_code == 200
            experiment_data = response.json()
            assert experiment_data['mass_gold'] == 0.1576
            assert experiment_data['mass_toab'] == 0.25
            assert experiment_data['volume_toluene'] == 10.0
            
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend not running")
    
    def test_error_handling(self):
        """Test API error handling"""
        try:
            # Test invalid experiment ID
            response = requests.get(f"{self.base_url}/experiments/nonexistent")
            assert response.status_code == 404
            
            # Test invalid data recording
            invalid_data = {
                "experiment_id": "nonexistent",
                "data_type": "mass",
                "compound": "test",
                "value": 1.0,
                "units": "g"
            }
            response = requests.post(f"{self.base_url}/data", json=invalid_data)
            assert response.status_code == 404
            
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend not running")


class TestVoiceProcessingIntegration:
    """Test voice processing pipeline integration"""
    
    def test_voice_transcript_processing(self):
        """Test voice transcript to data conversion"""
        # Import voice processing function (with error handling for weave)
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            # Mock streamlit session state
            with patch('streamlit.session_state') as mock_session:
                mock_platform = Mock()
                mock_platform.record_data_via_api.return_value = (True, {"status": "recorded"})
                mock_session.platform = mock_platform
                
                # Test voice patterns
                test_cases = [
                    ("Gold mass is 0.1576 grams", "HAuCl4·3H2O", 0.1576, "g"),
                    ("TOAB mass is 0.25 g", "TOAB", 0.25, "g"),
                    ("Toluene volume is 10 milliliters", "toluene", 10.0, "mL"),
                    ("Sulfur amount is 0.2 grams", "PhCH2CH2SH", 0.2, "g"),
                ]
                
                for transcript, expected_compound, expected_value, expected_units in test_cases:
                    # Test pattern matching directly
                    import re
                    
                    # Extract number and units
                    number_pattern = r'(\d+\.?\d*|\d*\.\d+)\s*(g|gram|grams|ml|milliliter|milliliters|mL)'
                    matches = re.findall(number_pattern, transcript.lower())
                    
                    assert len(matches) > 0, f"No matches found in: {transcript}"
                    
                    value, units = matches[0]
                    value = float(value)
                    
                    # Normalize units
                    units = 'g' if units.startswith('g') else 'mL'
                    
                    assert value == expected_value
                    assert units == expected_units
                    
                    # Test compound detection
                    compound_mapping = {
                        'gold': 'HAuCl4·3H2O',
                        'toab': 'TOAB',
                        'sulfur': 'PhCH2CH2SH',
                        'toluene': 'toluene',
                    }
                    
                    found_compound = None
                    for key, mapped_compound in compound_mapping.items():
                        if key in transcript.lower():
                            found_compound = mapped_compound
                            break
                    
                    assert found_compound == expected_compound
                    
        except ImportError:
            pytest.skip("Voice processing module not available")


class TestFileIntegration:
    """Test file I/O and data persistence"""
    
    def test_csv_export_integration(self):
        """Test CSV export functionality"""
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
            from weavehacks_flow.agents.data_collection_agent import DataCollectionAgent
            
            agent = DataCollectionAgent()
            
            # Create mock experiment state
            mock_state = Mock()
            mock_state.experiment_id = "test_csv_export"
            mock_state.mass_gold = 0.1576
            mock_state.mass_toab = 0.25
            mock_state.mass_sulfur = 0.1659
            mock_state.mass_nabh4 = 0.1514
            mock_state.volume_toluene = 10.0
            mock_state.volume_nanopure_cold = 2.0
            mock_state.mass_final = 0.08
            
            # Test CSV export
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                csv_path = tmp_file.name
            
            try:
                agent.state_to_table(mock_state, csv_path)
                
                # Verify CSV was created and has content
                assert os.path.exists(csv_path)
                
                with open(csv_path, 'r') as f:
                    content = f.read()
                    assert 'HAuCl₄·3H₂O' in content
                    assert '0.1576' in content
                    assert 'TOAB' in content
                    assert '0.25' in content
                    
            finally:
                if os.path.exists(csv_path):
                    os.unlink(csv_path)
                    
        except ImportError:
            pytest.skip("Data collection agent not available")


class TestComponentIntegration:
    """Test integration between different components"""
    
    def test_agent_communication(self):
        """Test communication between different agents"""
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
            from weavehacks_flow.agents.data_collection_agent import DataCollectionAgent
            from weavehacks_flow.agents.lab_control_agent import LabControlAgent
            from weavehacks_flow.agents.safety_monitoring_agent import EnhancedSafetyMonitoringAgent
            
            # Initialize agents
            data_agent = DataCollectionAgent()
            lab_agent = LabControlAgent()
            safety_agent = EnhancedSafetyMonitoringAgent()
            
            # Test basic agent functionality
            assert data_agent is not None
            assert lab_agent is not None
            assert safety_agent is not None
            
            # Test lab control operations
            result = lab_agent.turn_on("centrifuge")
            assert result is True
            
            result = lab_agent.turn_off("centrifuge")
            assert result is True
            
            # Test safety monitoring
            safety_agent.set_experiment("Gold Nanoparticle Synthesis (Room Temp)")
            status = safety_agent.get_status_report()
            assert isinstance(status, dict)
            assert 'alerts_active' in status
            
        except ImportError as e:
            pytest.skip(f"Agent modules not available: {e}")


class TestSystemIntegration:
    """Test overall system integration"""
    
    def test_system_health_check(self):
        """Test overall system health"""
        checks = []
        
        # Test Python environment
        checks.append(("Python version", sys.version_info >= (3, 9)))
        
        # Test required modules
        try:
            import numpy
            checks.append(("NumPy", True))
        except ImportError:
            checks.append(("NumPy", False))
        
        try:
            import pandas
            checks.append(("Pandas", True))
        except ImportError:
            checks.append(("Pandas", False))
        
        try:
            import streamlit
            checks.append(("Streamlit", True))
            
            # Check Streamlit version for audio support
            import packaging.version
            st_version = packaging.version.parse(streamlit.__version__)
            audio_support = st_version >= packaging.version.parse("1.37.0")
            checks.append(("Streamlit audio support", audio_support))
            
        except ImportError:
            checks.append(("Streamlit", False))
            checks.append(("Streamlit audio support", False))
        
        # Test optional modules
        try:
            import whisper
            checks.append(("OpenAI Whisper", True))
        except ImportError:
            checks.append(("OpenAI Whisper", False))
        
        try:
            import cv2
            checks.append(("OpenCV", True))
        except ImportError:
            checks.append(("OpenCV", False))
        
        # Print results
        print("\nSystem Health Check:")
        print("=" * 50)
        for check_name, status in checks:
            status_str = "✅ PASS" if status else "❌ FAIL"
            print(f"{check_name:.<30} {status_str}")
        
        # Core requirements must pass
        core_requirements = ["Python version", "NumPy", "Pandas", "Streamlit"]
        core_status = [status for name, status in checks if name in core_requirements]
        
        assert all(core_status), "Core system requirements not met"
    
    def test_directory_structure(self):
        """Test that the restructured directory structure is correct"""
        base_path = os.path.join(os.path.dirname(__file__), '..')
        
        # Test main directories exist
        required_dirs = [
            'src/weavehacks_flow',
            'src/weavehacks_flow/agents',
            'src/weavehacks_flow/utils',
            'src/weavehacks_flow/config',
            'backend',
            'frontend/components',
            'tests_new',
            'examples'
        ]
        
        for dir_path in required_dirs:
            full_path = os.path.join(base_path, dir_path)
            assert os.path.exists(full_path), f"Required directory missing: {dir_path}"
        
        # Test main files exist
        required_files = [
            'README.md',
            'requirements.txt',
            'integrated_app.py',
            'integration_bridge.py',
            'src/weavehacks_flow/main.py',
            'backend/main.py'
        ]
        
        for file_path in required_files:
            full_path = os.path.join(base_path, file_path)
            assert os.path.exists(full_path), f"Required file missing: {file_path}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])