#!/usr/bin/env python3
"""
Make the codebase work without problematic dependencies
"""

import os
import re
import shutil
from datetime import datetime

def create_backup(file_path):
    """Create a backup of the file"""
    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(file_path, backup_path)
    return backup_path

def patch_file(file_path, patches):
    """Apply patches to a file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original = content
    for pattern, replacement in patches:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
    
    if content != original:
        create_backup(file_path)
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    print("Making codebase standalone (work without problematic dependencies)")
    print("=" * 60)
    
    # Patches to apply
    patches = {
        'src/weavehacks_flow/main.py': [
            # Already has CrewAI compatibility layer, just ensure it works
            (r'^from crewai import .*$', '# from crewai import ... # Using compatibility layer'),
            
            # Make weave optional
            (r'^import weave\n', '''try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
    print("Weave not available - logging disabled")
'''),
            (r'@weave\.op\(\)', '''def weave_op_decorator(func):
    if WEAVE_AVAILABLE:
        return weave.op()(func)
    return func

@weave_op_decorator'''),
            
            # Make wandb optional (already has safe_wandb_log)
        ],
        
        'src/weavehacks_flow/agents/voice_recognition_agent.py': [
            # Make whisper optional
            (r'^import whisper\n', '''try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Whisper not available - voice recognition disabled")
'''),
            
            # Update load_model to check availability
            (r'def load_model\(self\):', '''def load_model(self):
        """Load the Whisper model."""
        if not WHISPER_AVAILABLE:
            self.logger.warning("Whisper not available")
            return'''),
        ],
        
        'src/weavehacks_flow/agents/video_monitoring_agent.py': [
            # Already handles cv2 import gracefully
        ],
        
        'src/weavehacks_flow/utils/chemistry_calculations.py': [
            # Make weave optional  
            (r'^import weave\n', '''try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
'''),
            (r'@weave\.op\(\)', '''def weave_op_decorator(func):
    if WEAVE_AVAILABLE:
        return weave.op()(func)
    return func

@weave_op_decorator'''),
        ]
    }
    
    # Apply patches
    patched_files = []
    for file_path, file_patches in patches.items():
        if os.path.exists(file_path):
            print(f"\nPatching {file_path}...")
            if patch_file(file_path, file_patches):
                patched_files.append(file_path)
                print(f"✅ Patched successfully")
            else:
                print(f"⏭️  No changes needed")
    
    # Create a simple requirements file
    simple_requirements = """# Minimal requirements for standalone operation
pydantic==1.10.14
numpy==1.24.4
requests==2.31.0
python-dotenv==1.0.1
pandas==1.5.3
matplotlib==3.7.5
scipy==1.10.1
plotly==5.20.0
streamlit==1.32.2
sounddevice==0.4.6
soundfile==0.12.1
pytest==8.0.2

# Optional (comment out if issues):
# opencv-python==4.8.1.78
# fastapi==0.111.0
# uvicorn==0.29.0
"""
    
    with open('requirements_standalone.txt', 'w') as f:
        f.write(simple_requirements)
    
    print("\n" + "=" * 60)
    print("STANDALONE CONVERSION COMPLETE")
    print("=" * 60)
    
    if patched_files:
        print(f"\n✅ Patched {len(patched_files)} files")
        print("Backups created for all modified files")
    
    print("\n📋 Next steps:")
    print("1. Install minimal dependencies:")
    print("   pip3 install --user -r requirements_standalone.txt")
    print("\n2. Run the demo:")
    print("   python3 run_demo.py")
    print("\n3. If you need specific features:")
    print("   - Voice: pip3 install --user openai-whisper")
    print("   - Video: pip3 install --user opencv-python==4.8.1.78")
    print("   - Logging: pip3 install --user wandb==0.16.6 weave==0.50.0")

if __name__ == "__main__":
    main()