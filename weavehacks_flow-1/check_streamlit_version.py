#!/usr/bin/env python3
"""
Check Streamlit version and available features
"""

import sys

def check_streamlit():
    """Check Streamlit installation and version"""
    print("Streamlit Version Check")
    print("=" * 50)
    
    try:
        import streamlit as st
        print(f"✅ Streamlit installed")
        print(f"   Version: {st.__version__}")
        
        # Parse version
        version_parts = st.__version__.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1])
        
        print(f"\n📋 Feature Availability:")
        
        # Check for audio_input (added in 1.37.0)
        if hasattr(st, 'audio_input'):
            print(f"✅ st.audio_input - Available")
        else:
            print(f"❌ st.audio_input - Not available (requires 1.37.0+)")
            print(f"   Current version: {st.__version__}")
            print(f"   Upgrade with: pip3 install --user --upgrade streamlit")
        
        # Check for other common features
        features = [
            ('file_uploader', 'File upload'),
            ('audio', 'Audio playback'),
            ('chat_input', 'Chat input (1.24.0+)'),
            ('chat_message', 'Chat message (1.24.0+)'),
            ('tabs', 'Tab layout (1.10.0+)'),
            ('metric', 'Metrics display'),
            ('expander', 'Expandable sections'),
            ('form', 'Forms'),
            ('plotly_chart', 'Plotly charts'),
            ('dataframe', 'Dataframe display'),
        ]
        
        print(f"\n📊 Other Features:")
        for attr, name in features:
            if hasattr(st, attr):
                print(f"✅ {name} (st.{attr})")
            else:
                print(f"❌ {name} (st.{attr})")
        
        # Recommend version
        print(f"\n💡 Recommendations:")
        if major < 1 or (major == 1 and minor < 37):
            print(f"   - Consider upgrading to Streamlit 1.37.0+ for audio_input")
            print(f"   - Command: pip3 install --user streamlit>=1.37.0")
        else:
            print(f"   - Your Streamlit version is up to date!")
            
    except ImportError:
        print("❌ Streamlit not installed")
        print("   Install with: pip3 install --user streamlit")
        return False
    
    return True

def check_audio_dependencies():
    """Check audio processing dependencies"""
    print(f"\n🎤 Audio Processing Dependencies")
    print("=" * 50)
    
    deps = {
        'whisper': 'OpenAI Whisper (speech recognition)',
        'sounddevice': 'Audio device access',
        'soundfile': 'Audio file I/O',
        'numpy': 'Numerical operations',
        'scipy': 'Signal processing',
    }
    
    for module, description in deps.items():
        try:
            __import__(module)
            print(f"✅ {module} - {description}")
        except ImportError:
            print(f"❌ {module} - {description}")
            if module == 'whisper':
                print(f"   Install with: pip3 install --user openai-whisper")
            else:
                print(f"   Install with: pip3 install --user {module}")

def main():
    """Run all checks"""
    check_streamlit()
    check_audio_dependencies()
    
    print(f"\n📝 Summary")
    print("=" * 50)
    print("The integrated app has fallbacks for missing features:")
    print("- If audio_input is not available, file upload is used instead")
    print("- If whisper is not installed, manual text entry is available")
    print("- The app will work with older Streamlit versions")

if __name__ == "__main__":
    main()