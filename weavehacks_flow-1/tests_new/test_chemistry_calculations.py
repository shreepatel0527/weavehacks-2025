"""
Unit tests for chemistry calculations
"""
import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from weavehacks_flow.utils.chemistry_calculations import (
    calculate_moles, calculate_mass, calculate_sulfur_amount,
    calculate_nabh4_amount, calculate_percent_yield, calculate_toab_ratio,
    calculate_all_reagents
)
from weavehacks_flow.utils.error_handling import CalculationError
from weavehacks_flow.config.settings import get_chemistry_config

class TestChemistryCalculations(unittest.TestCase):
    """Test chemistry calculation functions"""
    
    def setUp(self):
        """Set up test data"""
        self.config = get_chemistry_config()
        self.gold_mass = 0.1576  # g of HAuCl4·3H2O
        self.tolerance = 0.0001  # 4 decimal places
    
    def test_calculate_moles(self):
        """Test mole calculation"""
        # Test valid calculation
        moles = calculate_moles(self.gold_mass, self.config.MW_HAuCl4_3H2O)
        expected = self.gold_mass / self.config.MW_HAuCl4_3H2O
        self.assertAlmostEqual(moles, expected, places=6)
        
        # Test zero mass
        moles = calculate_moles(0.0, self.config.MW_HAuCl4_3H2O)
        self.assertEqual(moles, 0.0)
        
        # Test invalid inputs should raise error
        with self.assertRaises(CalculationError):
            calculate_moles(-1.0, self.config.MW_HAuCl4_3H2O)
    
    def test_calculate_mass(self):
        """Test mass calculation"""
        # Test valid calculation
        moles = 0.0004
        mass = calculate_mass(moles, self.config.MW_HAuCl4_3H2O)
        expected = moles * self.config.MW_HAuCl4_3H2O
        self.assertAlmostEqual(mass, expected, places=4)
        
        # Test zero moles
        mass = calculate_mass(0.0, self.config.MW_HAuCl4_3H2O)
        self.assertEqual(mass, 0.0)
    
    def test_calculate_sulfur_amount(self):
        """Test sulfur amount calculation"""
        # Test with default equivalents (3)
        result = calculate_sulfur_amount(self.gold_mass)
        
        # Check all required fields are present
        self.assertIn('moles_gold', result)
        self.assertIn('moles_sulfur', result)
        self.assertIn('mass_sulfur_g', result)
        self.assertIn('volume_sulfur_ml', result)
        self.assertIn('equivalents', result)
        self.assertIn('units', result)
        
        # Check calculations
        moles_gold = self.gold_mass / self.config.MW_HAuCl4_3H2O
        expected_moles_sulfur = moles_gold * 3
        self.assertAlmostEqual(result['moles_sulfur'], expected_moles_sulfur, places=6)
        
        # Test with custom equivalents
        result = calculate_sulfur_amount(self.gold_mass, equivalents=5)
        self.assertEqual(result['equivalents'], 5)
        expected_moles_sulfur = moles_gold * 5
        self.assertAlmostEqual(result['moles_sulfur'], expected_moles_sulfur, places=6)
        
        # Test invalid input
        with self.assertRaises(CalculationError):
            calculate_sulfur_amount(-1.0)
    
    def test_calculate_nabh4_amount(self):
        """Test NaBH4 amount calculation"""
        # Test with default equivalents (10)
        result = calculate_nabh4_amount(self.gold_mass)
        
        # Check all required fields
        self.assertIn('moles_gold', result)
        self.assertIn('moles_nabh4', result)
        self.assertIn('mass_nabh4_g', result)
        self.assertIn('volume_solution_ml', result)
        self.assertIn('concentration_M', result)
        self.assertIn('equivalents', result)
        self.assertIn('units', result)
        
        # Check calculations
        moles_gold = self.gold_mass / self.config.MW_HAuCl4_3H2O
        expected_moles_nabh4 = moles_gold * 10
        self.assertAlmostEqual(result['moles_nabh4'], expected_moles_nabh4, places=6)
        
        # Check solution volume calculation
        expected_volume = (expected_moles_nabh4 / 0.5) * 1000  # 0.5 M solution
        self.assertAlmostEqual(result['volume_solution_ml'], expected_volume, places=1)
    
    def test_calculate_percent_yield(self):
        """Test percent yield calculation"""
        actual_yield = 0.045  # g
        
        result = calculate_percent_yield(self.gold_mass, actual_yield)
        
        # Check all required fields
        self.assertIn('starting_mass_g', result)
        self.assertIn('gold_content_g', result)
        self.assertIn('theoretical_yield_g', result)
        self.assertIn('actual_yield_g', result)
        self.assertIn('percent_yield', result)
        self.assertIn('yield_quality', result)
        self.assertIn('target_yield_percent', result)
        self.assertIn('units', result)
        
        # Check yield quality assessment
        self.assertIn(result['yield_quality'], ['Excellent', 'Good', 'Poor'])
        
        # Test zero actual yield
        result = calculate_percent_yield(self.gold_mass, 0.0)
        self.assertEqual(result['percent_yield'], 0.0)
        self.assertEqual(result['yield_quality'], 'Poor')
    
    def test_calculate_toab_ratio(self):
        """Test TOAB ratio calculation"""
        toab_mass = 0.25  # g
        
        result = calculate_toab_ratio(self.gold_mass, toab_mass)
        
        # Check all required fields
        self.assertIn('moles_gold', result)
        self.assertIn('moles_toab', result)
        self.assertIn('toab_to_gold_ratio', result)
        self.assertIn('ratio_assessment', result)
        self.assertIn('recommended_ratio', result)
        self.assertIn('units', result)
        
        # Check ratio assessment
        self.assertIn(result['ratio_assessment'], [
            'Optimal',
            'Too low - may affect phase transfer',
            'Too high - excess TOAB may interfere'
        ])
        
        # Test optimal ratio
        moles_gold = self.gold_mass / self.config.MW_HAuCl4_3H2O
        optimal_toab_mass = calculate_mass(moles_gold * 2, self.config.MW_TOAB)
        result = calculate_toab_ratio(self.gold_mass, optimal_toab_mass)
        self.assertAlmostEqual(result['toab_to_gold_ratio'], 2.0, places=1)
        self.assertEqual(result['ratio_assessment'], 'Optimal')
    
    def test_calculate_all_reagents(self):
        """Test complete reagent calculation"""
        result = calculate_all_reagents(self.gold_mass)
        
        # Check main sections
        self.assertIn('gold', result)
        self.assertIn('toab', result)
        self.assertIn('sulfur', result)
        self.assertIn('nabh4', result)
        self.assertIn('solvents', result)
        self.assertIn('total_volume_ml', result)
        
        # Check gold data
        self.assertAlmostEqual(result['gold']['mass_g'], self.gold_mass, places=4)
        
        # Check TOAB calculation (should be 2:1 ratio)
        expected_toab_moles = result['gold']['moles'] * 2
        self.assertAlmostEqual(result['toab']['moles'], expected_toab_moles, places=6)
        
        # Check total volume
        expected_total = result['solvents']['toluene_ml'] + result['solvents']['water_ml']
        self.assertEqual(result['total_volume_ml'], expected_total)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Very small mass
        small_mass = 0.0001
        result = calculate_sulfur_amount(small_mass)
        self.assertGreater(result['mass_sulfur_g'], 0)
        
        # Large mass
        large_mass = 10.0
        result = calculate_nabh4_amount(large_mass)
        self.assertGreater(result['mass_nabh4_g'], 0)
        self.assertGreater(result['volume_solution_ml'], 0)
        
        # High percent yield
        result = calculate_percent_yield(self.gold_mass, 0.1)
        self.assertGreater(result['percent_yield'], 50)
        
        # Low percent yield
        result = calculate_percent_yield(self.gold_mass, 0.01)
        self.assertLess(result['percent_yield'], 20)

class TestErrorHandling(unittest.TestCase):
    """Test error handling in calculations"""
    
    def test_negative_mass_errors(self):
        """Test that negative masses raise errors"""
        with self.assertRaises(CalculationError):
            calculate_sulfur_amount(-1.0)
        
        with self.assertRaises(CalculationError):
            calculate_nabh4_amount(-0.5)
        
        with self.assertRaises(CalculationError):
            calculate_percent_yield(-1.0, 0.1)
    
    def test_zero_denominators(self):
        """Test handling of zero denominators"""
        # This should be handled gracefully
        result = calculate_percent_yield(0.0001, 0.0)
        self.assertEqual(result['percent_yield'], 0.0)
    
    def test_invalid_equivalents(self):
        """Test invalid equivalents"""
        with self.assertRaises(CalculationError):
            calculate_sulfur_amount(0.1, equivalents=-1)
        
        with self.assertRaises(CalculationError):
            calculate_nabh4_amount(0.1, equivalents=0)

if __name__ == '__main__':
    unittest.main()