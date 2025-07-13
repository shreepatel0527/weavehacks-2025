#!/usr/bin/env python
"""
Main entry point for the WeaveHacks Lab Assistant
"""
import sys
import os
import argparse
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'streamlit',
        'crewai',
        'weave',
        'wandb',
        'pydantic',
        'plotly',
        'pandas',
        'whisper'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Missing dependencies detected:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True

def run_streamlit_ui():
    """Run the Streamlit unified UI"""
    print("Starting WeaveHacks Lab Assistant UI...")
    print("=" * 60)
    print("Features:")
    print("  ✓ AI-powered lab assistant (Claude & Gemini)")
    print("  ✓ Real-time safety monitoring")
    print("  ✓ Automated chemical calculations")
    print("  ✓ Protocol step tracking")
    print("  ✓ Voice input support")
    print("  ✓ W&B Weave integration for monitoring")
    print("=" * 60)
    
    # Run Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "unified_lab_assistant.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ])

def run_crewai_flow():
    """Run the CrewAI experiment flow"""
    print("Starting CrewAI Experiment Flow...")
    print("=" * 60)
    
    # Change to the correct directory
    os.chdir("weavehacks_flow-1")
    
    # Run the test script
    subprocess.run([sys.executable, "test_experiment_flow.py"])

def run_tests():
    """Run the test suite"""
    print("Running test suite...")
    print("=" * 60)
    
    # Change to the correct directory
    os.chdir("weavehacks_flow-1")
    
    # Run tests
    subprocess.run([
        sys.executable, "-m", "pytest", "tests/", "-v"
    ])

def main():
    parser = argparse.ArgumentParser(
        description="WeaveHacks Lab Assistant - AI-powered wet lab assistant"
    )
    
    parser.add_argument(
        'mode',
        choices=['ui', 'flow', 'test', 'all'],
        help='Run mode: ui (Streamlit UI), flow (CrewAI flow), test (run tests), all (everything)'
    )
    
    parser.add_argument(
        '--skip-deps',
        action='store_true',
        help='Skip dependency check'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not args.skip_deps:
        if not check_dependencies():
            print("\nPlease install missing dependencies before running.")
            return 1
    
    # Run based on mode
    if args.mode == 'ui':
        run_streamlit_ui()
    elif args.mode == 'flow':
        run_crewai_flow()
    elif args.mode == 'test':
        run_tests()
    elif args.mode == 'all':
        # Run tests first
        print("1. Running tests...")
        run_tests()
        
        print("\n2. Running CrewAI flow demo...")
        run_crewai_flow()
        
        print("\n3. Starting UI...")
        run_streamlit_ui()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())