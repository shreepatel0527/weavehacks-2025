#!/usr/bin/env python
"""
Test script for the enhanced experiment flow with W&B Weave integration
"""
import sys
import os
import json
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_mock_input_file():
    """Create a mock input file for automated testing"""
    mock_inputs = [
        "0.1576",  # Gold mass
        "5.0",     # Room temp nanopure volume
        "0.25",    # TOAB mass
        "10.0",    # Toluene volume
        "0.052",   # Sulfur mass (calculated ~0.052g)
        "0.015",   # NaBH4 mass (calculated ~0.015g)
        "7.0",     # Cold nanopure volume
        "0.045"    # Final nanoparticle mass
    ]
    
    with open("mock_inputs.txt", "w") as f:
        f.write("\n".join(mock_inputs))
    
    return mock_inputs

def run_experiment_demo():
    """Run a demonstration of the experiment flow"""
    print("="*60)
    print("WeaveHacks Lab Assistant - Experiment Flow Demo")
    print("="*60)
    print("\nThis demo will showcase:")
    print("1. W&B Weave integration for agent monitoring")
    print("2. Automatic calculation of reagent amounts")
    print("3. Percent yield calculation")
    print("4. Enhanced safety monitoring with real sensor data")
    print("5. Improved code structure and error handling")
    print("\n" + "="*60)
    
    # Create mock inputs for automated testing
    print("\nCreating mock input data for demonstration...")
    mock_inputs = create_mock_input_file()
    
    print("\nMock input values:")
    print(f"  Gold mass: {mock_inputs[0]}g")
    print(f"  TOAB mass: {mock_inputs[2]}g")
    print(f"  Sulfur mass: {mock_inputs[4]}g")
    print(f"  NaBH4 mass: {mock_inputs[5]}g")
    print(f"  Final yield: {mock_inputs[7]}g")
    
    print("\n" + "="*60)
    print("Starting experiment flow...")
    print("="*60 + "\n")
    
    # Import and run the experiment
    try:
        # Redirect stdin to use mock inputs
        original_stdin = sys.stdin
        sys.stdin = open("mock_inputs.txt", "r")
        
        from weavehacks_flow.main import kickoff
        kickoff()
        
        # Restore original stdin
        sys.stdin = original_stdin
        
    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up mock input file
        if os.path.exists("mock_inputs.txt"):
            os.remove("mock_inputs.txt")
    
    print("\n" + "="*60)
    print("Experiment demo completed!")
    print("Check your W&B dashboard for detailed agent monitoring data")
    print("="*60)

def test_calculations():
    """Test the calculation functions independently"""
    print("\n" + "="*60)
    print("Testing Calculation Functions")
    print("="*60)
    
    # Test sulfur calculation
    gold_mass = 0.1576  # g
    MW_HAuCl4_3H2O = 393.83  # g/mol
    MW_PhCH2CH2SH = 138.23   # g/mol
    
    moles_gold = gold_mass / MW_HAuCl4_3H2O
    moles_sulfur = moles_gold * 3
    mass_sulfur_needed = moles_sulfur * MW_PhCH2CH2SH
    
    print(f"\nSulfur Calculation Test:")
    print(f"  Gold mass: {gold_mass}g")
    print(f"  Moles of gold: {moles_gold:.6f} mol")
    print(f"  Moles of sulfur (3 eq): {moles_sulfur:.6f} mol")
    print(f"  Mass of sulfur needed: {mass_sulfur_needed:.4f}g")
    
    # Test NaBH4 calculation
    MW_NaBH4 = 37.83  # g/mol
    moles_nabh4 = moles_gold * 10
    mass_nabh4_needed = moles_nabh4 * MW_NaBH4
    
    print(f"\nNaBH4 Calculation Test:")
    print(f"  Moles of NaBH4 (10 eq): {moles_nabh4:.6f} mol")
    print(f"  Mass of NaBH4 needed: {mass_nabh4_needed:.4f}g")
    
    # Test percent yield calculation
    MW_Au = 196.97  # g/mol
    mass_Au_in_HAuCl4 = (MW_Au / MW_HAuCl4_3H2O) * gold_mass
    actual_yield = 0.045  # g (example)
    percent_yield = (actual_yield / mass_Au_in_HAuCl4) * 100
    
    print(f"\nPercent Yield Calculation Test:")
    print(f"  Gold content in starting material: {mass_Au_in_HAuCl4:.4f}g")
    print(f"  Actual yield: {actual_yield}g")
    print(f"  Percent yield: {percent_yield:.2f}%")

if __name__ == "__main__":
    # Run calculation tests
    test_calculations()
    
    # Run experiment demo
    print("\n\nProceed with experiment demo? (y/n): ", end="")
    response = input().strip().lower()
    
    if response == 'y':
        run_experiment_demo()
    else:
        print("Demo cancelled.")