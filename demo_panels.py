#!/usr/bin/env python3
"""
WeaveHacks 2025 - Data and Step Panels Demo
Test script for the new data panel and step panel functionality
"""

import requests
import json
import time
from datetime import datetime

API_BASE_URL = "http://localhost:8000"

def test_panels_functionality():
    """Test the data and step panels functionality"""
    print("📋 WeaveHacks Panels Demo")
    print("=" * 50)
    
    # Check backend health
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Backend API is healthy")
        else:
            print("❌ Backend API not responding correctly")
            return
    except:
        print("❌ Cannot connect to backend API")
        print("   Start backend with: cd backend && uvicorn main:app --reload")
        return
    
    # Create test experiment
    experiment_id = f"panels_demo_{int(time.time())}"
    print(f"\n🧪 Creating test experiment: {experiment_id}")
    
    try:
        response = requests.post(f"{API_BASE_URL}/experiments", 
                               params={"experiment_id": experiment_id})
        if response.status_code == 200:
            print("✅ Experiment created successfully")
        else:
            print(f"❌ Failed to create experiment: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error creating experiment: {e}")
        return
    
    # Test protocol steps API
    print("\n📋 Testing Protocol Steps API...")
    try:
        response = requests.get(f"{API_BASE_URL}/protocol/steps")
        if response.status_code == 200:
            data = response.json()
            steps = data["steps"]
            print(f"✅ Loaded {len(steps)} protocol steps")
            
            # Display first few steps
            for i, step in enumerate(steps[:3]):
                print(f"   Step {i+1}: {step['title']} ({step['estimated_time']})")
                print(f"           {step['description']}")
        else:
            print(f"❌ Failed to load protocol steps: {response.status_code}")
    except Exception as e:
        print(f"❌ Error loading protocol steps: {e}")
    
    # Test step progression
    print("\n⏭️ Testing Step Progression...")
    for step_num in range(3):
        try:
            response = requests.put(f"{API_BASE_URL}/experiments/{experiment_id}/step",
                                   json={"step_num": step_num})
            if response.status_code == 200:
                print(f"✅ Updated to step {step_num + 1}")
            else:
                print(f"❌ Failed to update step: {response.status_code}")
        except Exception as e:
            print(f"❌ Error updating step: {e}")
        
        time.sleep(0.5)  # Brief pause
    
    # Test data recording
    print("\n📊 Testing Data Recording...")
    test_data = [
        ("HAuCl₄·3H₂O", 0.1576, "g", "mass"),
        ("nanopure water", 5.0, "mL", "volume"),
        ("TOAB", 0.2543, "g", "mass"),
        ("toluene", 10.0, "mL", "volume")
    ]
    
    for compound, value, units, data_type in test_data:
        try:
            data_payload = {
                "experiment_id": experiment_id,
                "data_type": data_type,
                "compound": compound,
                "value": value,
                "units": units
            }
            
            response = requests.post(f"{API_BASE_URL}/data", json=data_payload)
            if response.status_code == 200:
                print(f"✅ Recorded {compound}: {value} {units}")
            else:
                print(f"❌ Failed to record {compound}: {response.status_code}")
        except Exception as e:
            print(f"❌ Error recording {compound}: {e}")
    
    # Test step completion
    print("\n✅ Testing Step Completion...")
    try:
        completion_data = {
            "step_id": 1,
            "step_title": "Weigh HAuCl₄·3H₂O"
        }
        
        response = requests.post(f"{API_BASE_URL}/experiments/{experiment_id}/steps/complete",
                               json=completion_data)
        if response.status_code == 200:
            print("✅ Step 1 marked as complete")
        else:
            print(f"❌ Failed to mark step complete: {response.status_code}")
    except Exception as e:
        print(f"❌ Error marking step complete: {e}")
    
    # Test step notes
    print("\n📝 Testing Step Notes...")
    try:
        note_data = {
            "step_id": 1,
            "note": "Gold compound weighed accurately on analytical balance. Sample appeared pure and dry."
        }
        
        response = requests.post(f"{API_BASE_URL}/experiments/{experiment_id}/steps/note",
                               json=note_data)
        if response.status_code == 200:
            print("✅ Step note added successfully")
        else:
            print(f"❌ Failed to add step note: {response.status_code}")
    except Exception as e:
        print(f"❌ Error adding step note: {e}")
    
    # Test qualitative observations
    print("\n🔍 Testing Qualitative Observations...")
    try:
        obs_data = {
            "observations": """Step 3 - Gold solution: Solution is clear yellow/orange as expected
Step 7 - Two-phase system: Good separation observed, organic phase appears colorless
Step 9 - Vigorous stirring: Excellent emulsion formation, gold transfer visible"""
        }
        
        response = requests.put(f"{API_BASE_URL}/experiments/{experiment_id}/observations",
                               json=obs_data)
        if response.status_code == 200:
            print("✅ Qualitative observations saved")
        else:
            print(f"❌ Failed to save observations: {response.status_code}")
    except Exception as e:
        print(f"❌ Error saving observations: {e}")
    
    # Test data export
    print("\n📤 Testing Data Export...")
    try:
        # Test CSV export
        response = requests.get(f"{API_BASE_URL}/experiments/{experiment_id}/export/csv")
        if response.status_code == 200:
            csv_data = response.json()
            print(f"✅ CSV export generated: {csv_data['filename']}")
            print("   Preview:")
            print("   " + "\n   ".join(csv_data['csv_content'].split('\n')[:8]))
        else:
            print(f"❌ Failed to export CSV: {response.status_code}")
    except Exception as e:
        print(f"❌ Error exporting CSV: {e}")
    
    try:
        # Test report export
        response = requests.get(f"{API_BASE_URL}/experiments/{experiment_id}/export/report")
        if response.status_code == 200:
            report_data = response.json()
            print(f"✅ Report export generated: {report_data['filename']}")
        else:
            print(f"❌ Failed to export report: {response.status_code}")
    except Exception as e:
        print(f"❌ Error exporting report: {e}")
    
    # Final experiment status
    print(f"\n📊 Final Experiment Status:")
    try:
        response = requests.get(f"{API_BASE_URL}/experiments/{experiment_id}")
        if response.status_code == 200:
            exp_data = response.json()
            print(f"   Experiment ID: {exp_data['experiment_id']}")
            print(f"   Current Step: {exp_data['step_num']}/12")
            print(f"   Status: {exp_data['status']}")
            print(f"   Masses recorded: Gold={exp_data['mass_gold']}g, TOAB={exp_data['mass_toab']}g")
            print(f"   Volumes recorded: Water={exp_data['volume_nanopure_rt']}mL, Toluene={exp_data['volume_toluene']}mL")
            print(f"   Safety status: {exp_data['safety_status']}")
        else:
            print(f"❌ Failed to get experiment status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting experiment status: {e}")
    
    print("\n🎉 Panels functionality demo completed!")
    print("\n💡 To see the panels in action:")
    print("   1. Start the frontend: streamlit run integrated_app.py")
    print("   2. Go to the 'Dashboard' tab and create an experiment")
    print("   3. Explore the 'Protocol Steps' and 'Data Panel' tabs")
    print("   4. Try the enhanced data entry and step management features")

def test_chemistry_calculations():
    """Test chemistry calculation features"""
    print("\n🧮 Testing Chemistry Calculations...")
    
    # Use a known experiment with gold mass
    experiment_id = "calc_test_001"
    
    # Create experiment
    try:
        response = requests.post(f"{API_BASE_URL}/experiments", 
                               params={"experiment_id": experiment_id})
        
        # Add gold mass
        gold_data = {
            "experiment_id": experiment_id,
            "data_type": "mass",
            "compound": "HAuCl₄·3H₂O",
            "value": 0.1576,
            "units": "g"
        }
        requests.post(f"{API_BASE_URL}/data", json=gold_data)
        
        # Test sulfur calculation
        response = requests.post(f"{API_BASE_URL}/calculations/sulfur-amount",
                               json={"experiment_id": experiment_id, "gold_mass": 0.1576})
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Sulfur calculation: {result['mass_sulfur_needed_g']:.4f}g needed")
        else:
            print(f"❌ Sulfur calculation failed: {response.status_code}")
        
        # Test NaBH4 calculation
        response = requests.post(f"{API_BASE_URL}/calculations/nabh4-amount",
                               json={"experiment_id": experiment_id, "gold_mass": 0.1576})
        if response.status_code == 200:
            result = response.json()
            print(f"✅ NaBH4 calculation: {result['mass_nabh4_needed_g']:.4f}g needed")
        else:
            print(f"❌ NaBH4 calculation failed: {response.status_code}")
        
        # Test percent yield (with mock final mass)
        final_data = {
            "experiment_id": experiment_id,
            "data_type": "mass",
            "compound": "Au₂₅ nanoparticles",
            "value": 0.0654,
            "units": "g"
        }
        requests.post(f"{API_BASE_URL}/data", json=final_data)
        
        response = requests.post(f"{API_BASE_URL}/calculations/percent-yield",
                               json={"experiment_id": experiment_id, "gold_mass": 0.1576, "final_mass": 0.0654})
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Percent yield: {result['percent_yield']:.2f}%")
        else:
            print(f"❌ Percent yield calculation failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error in chemistry calculations: {e}")

if __name__ == "__main__":
    test_panels_functionality()
    test_chemistry_calculations()