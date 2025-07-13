#!/usr/bin/env python3
"""
Start the backend API server with proper error handling
"""

import subprocess
import sys
import os
import time

def check_dependencies():
    """Check if required dependencies are installed"""
    required = ['fastapi', 'uvicorn', 'pydantic']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print(f"   Install with: pip3 install --user {' '.join(missing)}")
        return False
    
    return True

def start_backend():
    """Start the backend server"""
    print("Starting WeaveHacks Backend API...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Change to backend directory
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    if not os.path.exists(backend_dir):
        print(f"❌ Backend directory not found: {backend_dir}")
        return
    
    os.chdir(backend_dir)
    
    # Start uvicorn
    print("✅ Starting API server on http://localhost:8000")
    print("   Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        # Run uvicorn directly with Python
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--reload",
            "--host", "0.0.0.0",
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\n✅ Backend stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTry running manually:")
        print("  cd backend")
        print("  python3 -m uvicorn main:app --reload")

if __name__ == "__main__":
    start_backend()