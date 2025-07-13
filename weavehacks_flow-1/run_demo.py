#!/usr/bin/env python3
"""
Demo script to run the experiment flow
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("="*60)
print("WEAVEHACKS LAB ASSISTANT - DEMO")
print("="*60)
print("\nThis will run the main experiment flow.")
print("Since we don't have actual lab equipment, the data collection")
print("will be simulated with mock inputs.\n")

# Create mock input file
mock_inputs = """0.1576
10.0
0.25
10.0
0.1659
0.1514
7.0
0.05"""

with open("demo_inputs.txt", "w") as f:
    f.write(mock_inputs)

print("Mock inputs prepared:")
print("- Gold (HAuCl₄·3H₂O): 0.1576g")
print("- Water (room temp): 10.0mL") 
print("- TOAB: 0.25g")
print("- Toluene: 10.0mL")
print("- Sulfur compound: 0.1659g")
print("- NaBH4: 0.1514g")
print("- Water (ice-cold): 7.0mL")
print("- Final yield: 0.05g")
print("\n" + "-"*60 + "\n")

# Redirect stdin to use mock inputs
original_stdin = sys.stdin
sys.stdin = open("demo_inputs.txt", "r")

try:
    from weavehacks_flow.main import kickoff
    kickoff()
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Restore stdin and cleanup
    sys.stdin = original_stdin
    if os.path.exists("demo_inputs.txt"):
        os.remove("demo_inputs.txt")

print("\n" + "="*60)
print("DEMO COMPLETE")
print("="*60)