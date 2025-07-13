#!/usr/bin/env python3
"""
Dependency installation helper for WeaveHacks Flow
Installs dependencies incrementally to avoid resolution conflicts
"""

import subprocess
import sys
import os

def run_pip_install(package, user=True):
    """Install a package using pip"""
    cmd = [sys.executable, "-m", "pip", "install"]
    if user:
        cmd.append("--user")
    cmd.append(package)
    
    print(f"Installing: {package}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Successfully installed: {package}")
            return True
        else:
            print(f"❌ Failed to install: {package}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception installing {package}: {e}")
        return False

def main():
    print("WeaveHacks Flow Dependency Installer")
    print("=" * 50)
    
    # Essential dependencies in order of importance
    essential_packages = [
        # Core Python packages
        ("pydantic>=1.10.0,<2.0.0", "Data validation"),
        ("numpy==1.24.4", "Numerical computing"),
        ("requests==2.31.0", "HTTP requests"),
        ("python-dotenv==1.0.1", "Environment variables"),
        
        # Scientific computing
        ("scipy==1.10.1", "Scientific computing"),
        ("matplotlib==3.7.5", "Plotting"),
        ("pandas==1.5.3", "Data analysis"),
        
        # Testing
        ("pytest==8.0.2", "Testing framework"),
    ]
    
    optional_packages = [
        # UI
        ("streamlit==1.32.2", "Web UI framework"),
        ("plotly==5.20.0", "Interactive plots"),
        
        # Audio
        ("sounddevice==0.4.6", "Audio device access"),
        ("soundfile==0.12.1", "Audio file I/O"),
        
        # Backend API
        ("fastapi==0.111.0", "API framework"),
        ("uvicorn==0.29.0", "ASGI server"),
        
        # Computer vision (often problematic)
        ("opencv-python==4.8.1.78", "Computer vision"),
    ]
    
    problematic_packages = [
        # These often cause dependency conflicts
        ("crewai==0.1.40", "Agent framework - COMPLEX DEPENDENCIES"),
        ("wandb==0.16.6", "Experiment tracking"),
        ("weave==0.50.0", "W&B Weave"),
        ("openai-whisper", "Speech recognition - VERY LARGE"),
    ]
    
    # Install essential packages
    print("\n1. Installing essential packages...")
    print("-" * 30)
    essential_failed = []
    for package, desc in essential_packages:
        print(f"\n📦 {desc}")
        if not run_pip_install(package):
            essential_failed.append(package)
    
    if essential_failed:
        print(f"\n⚠️  Failed to install essential packages: {', '.join(essential_failed)}")
        print("The system may not work properly without these.")
    
    # Ask about optional packages
    print("\n2. Optional packages")
    print("-" * 30)
    install_optional = input("Install optional packages? (y/n): ").lower() == 'y'
    
    if install_optional:
        optional_failed = []
        for package, desc in optional_packages:
            print(f"\n📦 {desc}")
            if not run_pip_install(package):
                optional_failed.append(package)
        
        if optional_failed:
            print(f"\n⚠️  Failed to install optional packages: {', '.join(optional_failed)}")
    
    # Ask about problematic packages
    print("\n3. Problematic packages (often cause conflicts)")
    print("-" * 30)
    print("These packages have complex dependencies and may cause issues:")
    for package, desc in problematic_packages:
        print(f"  - {package}: {desc}")
    
    install_problematic = input("\nInstall problematic packages? (y/n): ").lower() == 'y'
    
    if install_problematic:
        for package, desc in problematic_packages:
            install_this = input(f"\nInstall {package} ({desc})? (y/n): ").lower() == 'y'
            if install_this:
                print(f"\n📦 {desc}")
                run_pip_install(package)
    
    # Summary
    print("\n" + "=" * 50)
    print("Installation Summary")
    print("=" * 50)
    
    # Test imports
    print("\nTesting core imports...")
    test_imports = [
        "numpy",
        "pandas",
        "requests",
        "pydantic",
        "pytest",
    ]
    
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
    
    print("\n✅ Installation complete!")
    print("\nTo run the WeaveHacks Flow:")
    print("1. Basic flow: python3 run_demo.py")
    print("2. With UI: streamlit run integrated_app.py")
    print("3. Backend API: cd backend && uvicorn main:app --reload")

if __name__ == "__main__":
    main()