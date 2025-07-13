#!/usr/bin/env python3
"""
Test script for WeaveHacks Lab Automation API
Tests the API endpoints to diagnose 400 and 500 errors
"""

import requests
import json
from datetime import datetime

API_BASE_URL = "http://localhost:8000"

def test_api():
    print("WeaveHacks API Test Suite")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Connection error: {e}")
        print("\n❌ Backend API is not running!")
        print("   Start it with: cd backend && uvicorn main:app --reload")
        return
    
    # Test 2: Create experiment
    print("\n2. Testing experiment creation...")
    test_exp_id = f"test_exp_{int(datetime.now().timestamp())}"
    
    # Try different ways to send the experiment_id
    print(f"   Experiment ID: {test_exp_id}")
    
    # Method 1: As query parameter (what the backend expects)
    print("\n   Method 1: Query parameter")
    try:
        response = requests.post(f"{API_BASE_URL}/experiments?experiment_id={test_exp_id}")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ Success: {response.json()}")
            experiment_data = response.json()
        else:
            print(f"   ❌ Error: {response.text}")
            return
    except Exception as e:
        print(f"   Connection error: {e}")
        return
    
    # Test 3: Get experiment
    print("\n3. Testing get experiment...")
    try:
        response = requests.get(f"{API_BASE_URL}/experiments/{test_exp_id}")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ Success")
            exp_data = response.json()
            print(f"   Step: {exp_data['step_num']}, Status: {exp_data['status']}")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   Connection error: {e}")
    
    # Test 4: Record data
    print("\n4. Testing data recording...")
    data_entry = {
        "experiment_id": test_exp_id,
        "data_type": "mass",
        "compound": "HAuCl₄·3H₂O",
        "value": 0.1576,
        "units": "g",
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/data", json=data_entry)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ Success: Recorded {data_entry['compound']}: {data_entry['value']} {data_entry['units']}")
        else:
            print(f"   ❌ Error: {response.text}")
            print(f"   Request data: {json.dumps(data_entry, indent=2)}")
    except Exception as e:
        print(f"   Connection error: {e}")
    
    # Test 5: Chemistry calculations
    print("\n5. Testing chemistry calculations...")
    
    # Test sulfur calculation
    print("\n   5a. Sulfur amount calculation")
    try:
        response = requests.post(
            f"{API_BASE_URL}/calculations/sulfur_amount?experiment_id={test_exp_id}&gold_mass=0.1576"
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success: Sulfur needed: {result.get('mass_sulfur_g', 0):.4f}g")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   Connection error: {e}")
    
    # Test 6: Voice transcription
    print("\n6. Testing voice transcription...")
    try:
        # Method 1: As query parameters
        print("\n   Method 1: Query parameters")
        response = requests.post(
            f"{API_BASE_URL}/voice/transcribe?experiment_id={test_exp_id}&audio_text=gold mass is 0.1576 grams"
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success: {result}")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   Connection error: {e}")
    
    # Test 7: Protocol steps
    print("\n7. Testing protocol steps...")
    try:
        response = requests.get(f"{API_BASE_URL}/protocol/steps")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success: Found {result['total_steps']} protocol steps")
        else:
            print(f"   ❌ Error: {response.text}")
    except Exception as e:
        print(f"   Connection error: {e}")
    
    print("\n" + "=" * 50)
    print("API Testing Complete")
    print("=" * 50)

if __name__ == "__main__":
    test_api()