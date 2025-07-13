#!/usr/bin/env python3
"""
Audio Diagnostics Script for Voice Recognition Issues

Run this script to diagnose "Error querying device -1" and other audio problems.

Usage: python diagnose_audio.py
"""

import sys
import os

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("🔊 VOICE RECOGNITION AUDIO DIAGNOSTICS")
    print("=" * 50)
    
    try:
        from weavehacks_flow.utils.audio_diagnostics import AudioDiagnostics
        from weavehacks_flow.agents.voice_recognition_agent import SpeechRecognizerAgent
        
        # Run full diagnosis
        print(AudioDiagnostics.run_full_diagnosis())
        
        # Test voice recognition agent
        print("\n🤖 Testing Voice Recognition Agent...")
        print("-" * 40)
        
        try:
            agent = SpeechRecognizerAgent(model_size="tiny")
            print(f"✅ Agent created successfully")
            print(f"📱 Audio initialized: {agent.audio_initialized}")
            print(f"🎤 Current device: {agent.device_id}")
            
            # Test microphone access
            mic_test = agent.test_microphone()
            print(f"🧪 Microphone test: {'✅ PASSED' if mic_test else '❌ FAILED'}")
            
            if not mic_test:
                print("\n💡 SUGGESTED SOLUTIONS:")
                print("1. Check microphone permissions")
                print("2. Ensure microphone is connected and enabled")
                print("3. Try a different audio device (see device list above)")
                print("4. Restart your audio services")
                
                # Show device info
                device_info = agent.get_device_info()
                if "suggested_device" in device_info and device_info["suggested_device"] is not None:
                    print(f"5. Try setting device manually: agent.set_audio_device({device_info['suggested_device']})")
        
        except Exception as e:
            print(f"❌ Agent creation failed: {e}")
            print("\n💡 This indicates a serious audio system issue.")
            print("Please follow the recommendations above.")
    
    except ImportError as e:
        print(f"❌ Module import failed: {e}")
        print("\n💡 Make sure you're running this from the project root directory.")
        print("Try: cd /path/to/weavehacks_flow-1 && python diagnose_audio.py")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("\n💡 Please report this issue with the full error message.")
    
    print("\n" + "=" * 50)
    print("📞 Need more help? Check the project documentation or file an issue.")

if __name__ == "__main__":
    main()