#!/usr/bin/env python3
"""
Check which dependencies are actually imported by the codebase
"""

import os
import re
from pathlib import Path

def find_imports(file_path):
    """Extract import statements from a Python file"""
    imports = set()
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Find import statements
        import_pattern = r'^\s*(?:from\s+(\S+)|import\s+(\S+))'
        for match in re.finditer(import_pattern, content, re.MULTILINE):
            module = match.group(1) or match.group(2)
            if module:
                # Get the base module name
                base_module = module.split('.')[0]
                imports.add(base_module)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return imports

def scan_directory(directory):
    """Scan all Python files in directory for imports"""
    all_imports = set()
    
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ directories
        if '__pycache__' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                imports = find_imports(file_path)
                all_imports.update(imports)
    
    return all_imports

def main():
    print("Scanning codebase for actual dependencies...")
    print("=" * 50)
    
    # Directories to scan
    directories = [
        'src/weavehacks_flow',
        'backend',
        'frontend',
        'tests'
    ]
    
    all_imports = set()
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"\nScanning {directory}...")
            imports = scan_directory(directory)
            all_imports.update(imports)
            print(f"Found {len(imports)} unique imports")
    
    # Categorize imports
    stdlib_modules = {
        'os', 'sys', 'time', 'datetime', 'json', 'random', 'math',
        'threading', 'queue', 'logging', 'pathlib', 'enum', 'dataclasses',
        'typing', 'collections', 'tempfile', 'warnings', 're', 'subprocess',
        'asyncio', 'functools', 'itertools', 'abc', 'copy', 'pickle',
        'urllib', 'http', 'socket', 'ssl', 'base64', 'hashlib', 'uuid'
    }
    
    external_imports = all_imports - stdlib_modules
    
    # Map import names to package names
    package_mapping = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'np': 'numpy',
        'pandas': 'pandas',
        'pd': 'pandas',
        'matplotlib': 'matplotlib',
        'plt': 'matplotlib',
        'scipy': 'scipy',
        'requests': 'requests',
        'streamlit': 'streamlit',
        'st': 'streamlit',
        'pytest': 'pytest',
        'pydantic': 'pydantic',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'plotly': 'plotly',
        'sounddevice': 'sounddevice',
        'sd': 'sounddevice',
        'soundfile': 'soundfile',
        'sf': 'soundfile',
        'whisper': 'openai-whisper',
        'weave': 'weave',
        'wandb': 'wandb',
        'crewai': 'crewai',
        'langchain': 'langchain',
        'dotenv': 'python-dotenv',
    }
    
    # Identify required packages
    required_packages = set()
    for imp in external_imports:
        if imp in package_mapping:
            required_packages.add(package_mapping[imp])
    
    print("\n" + "=" * 50)
    print("ANALYSIS RESULTS")
    print("=" * 50)
    
    print(f"\nTotal unique imports found: {len(all_imports)}")
    print(f"Standard library imports: {len(all_imports & stdlib_modules)}")
    print(f"External imports: {len(external_imports)}")
    
    print("\nRequired external packages:")
    for pkg in sorted(required_packages):
        print(f"  - {pkg}")
    
    print("\nImports not mapped to packages:")
    unmapped = external_imports - set(package_mapping.keys())
    for imp in sorted(unmapped):
        if not imp.startswith('_'):  # Skip private modules
            print(f"  - {imp}")
    
    # Check current requirements.txt
    if os.path.exists('requirements.txt'):
        print("\n" + "=" * 50)
        print("REQUIREMENTS.TXT ANALYSIS")
        print("=" * 50)
        
        with open('requirements.txt', 'r') as f:
            req_content = f.read()
        
        req_packages = set()
        for line in req_content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract package name
                pkg_name = re.split(r'[><=!]', line)[0].strip()
                req_packages.add(pkg_name)
        
        print(f"\nPackages in requirements.txt: {len(req_packages)}")
        
        print("\nRequired but not in requirements.txt:")
        missing = required_packages - req_packages
        for pkg in sorted(missing):
            print(f"  - {pkg}")
        
        print("\nIn requirements.txt but not imported:")
        unused = req_packages - required_packages
        for pkg in sorted(unused):
            print(f"  - {pkg}")

if __name__ == "__main__":
    main()