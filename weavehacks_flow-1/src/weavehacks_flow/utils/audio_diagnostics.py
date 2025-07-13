"""
Audio Device Diagnostics and Troubleshooting Utilities
"""
import sounddevice as sd
import numpy as np
import platform
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


class AudioDiagnostics:
    """Comprehensive audio device diagnostics and troubleshooting."""
    
    @staticmethod
    def get_system_info() -> Dict:
        """Get system information for audio diagnostics."""
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": sys.version,
            "sounddevice_version": getattr(sd, '__version__', 'unknown')
        }
    
    @staticmethod
    def diagnose_audio_devices() -> Dict:
        """Comprehensive audio device diagnosis."""
        diagnosis = {
            "system_info": AudioDiagnostics.get_system_info(),
            "devices": [],
            "default_input": None,
            "default_output": None,
            "errors": [],
            "recommendations": []
        }
        
        try:
            # Query all devices
            devices = sd.query_devices()
            diagnosis["devices"] = devices
            
            # Get default devices
            try:
                default_input = sd.default.device[0] if sd.default.device else None
                default_output = sd.default.device[1] if len(sd.default.device) > 1 else None
                diagnosis["default_input"] = default_input
                diagnosis["default_output"] = default_output
            except Exception as e:
                diagnosis["errors"].append(f"Error getting default devices: {e}")
        
        except Exception as e:
            diagnosis["errors"].append(f"Error querying devices: {e}")
            
        # Add recommendations based on findings
        diagnosis["recommendations"] = AudioDiagnostics._generate_recommendations(diagnosis)
        
        return diagnosis
    
    @staticmethod
    def _generate_recommendations(diagnosis: Dict) -> List[str]:
        """Generate troubleshooting recommendations based on diagnosis."""
        recommendations = []
        
        # Check if no devices found
        if not diagnosis.get("devices") or len(diagnosis["devices"]) == 0:
            recommendations.extend([
                "No audio devices detected. Check if:",
                "1. Microphone/headset is properly connected",
                "2. Audio drivers are installed and updated",
                "3. Audio services are running",
                "4. Application has microphone permissions"
            ])
        
        # Check default input device
        if diagnosis.get("default_input") is None or diagnosis.get("default_input") == -1:
            recommendations.extend([
                "No default input device set. Try:",
                "1. Set a default microphone in system settings",
                "2. Restart audio services",
                "3. Reinstall audio drivers"
            ])
        
        # Platform-specific recommendations
        system = diagnosis.get("system_info", {}).get("platform", "").lower()
        
        if "darwin" in system:  # macOS
            recommendations.extend([
                "macOS specific fixes:",
                "1. Grant microphone permissions in System Preferences > Security & Privacy > Privacy > Microphone",
                "2. Restart Core Audio: sudo killall coreaudiod",
                "3. Reset NVRAM/PRAM if issues persist"
            ])
        elif "windows" in system:  # Windows
            recommendations.extend([
                "Windows specific fixes:",
                "1. Check Windows Privacy Settings > Microphone > Allow apps to access microphone",
                "2. Update audio drivers from Device Manager",
                "3. Run Windows Audio Troubleshooter",
                "4. Restart Windows Audio service"
            ])
        elif "linux" in system:  # Linux
            recommendations.extend([
                "Linux specific fixes:",
                "1. Install/update ALSA: sudo apt-get install libasound2-dev",
                "2. Check PulseAudio: pulseaudio --check",
                "3. Add user to audio group: sudo usermod -a -G audio $USER",
                "4. Install PortAudio: sudo apt-get install portaudio19-dev"
            ])
        
        return recommendations
    
    @staticmethod
    def test_audio_recording(duration: float = 1.0, sample_rate: int = 16000) -> Tuple[bool, str]:
        """Test basic audio recording functionality."""
        try:
            # Test with default device
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            
            # Check if audio was actually recorded
            if audio_data is None or len(audio_data) == 0:
                return False, "No audio data recorded"
            
            # Check for silence (might indicate no microphone input)
            max_amplitude = np.max(np.abs(audio_data))
            if max_amplitude < 0.001:  # Very quiet threshold
                return False, f"Audio recorded but very quiet (max amplitude: {max_amplitude:.6f}). Check microphone levels."
            
            return True, f"Audio recording successful (max amplitude: {max_amplitude:.3f})"
            
        except sd.PortAudioError as e:
            return False, f"PortAudio error: {e}"
        except Exception as e:
            return False, f"Recording test failed: {e}"
    
    @staticmethod
    def suggest_device_selection(devices: List) -> Optional[int]:
        """Suggest the best input device based on available devices."""
        if not devices:
            return None
        
        # Look for devices with input channels
        input_devices = []
        for i, device in enumerate(devices):
            if device.get('max_input_channels', 0) > 0:
                input_devices.append((i, device))
        
        if not input_devices:
            return None
        
        # Prefer devices with "microphone" or "input" in name
        for device_id, device in input_devices:
            name = device.get('name', '').lower()
            if any(keyword in name for keyword in ['mic', 'microphone', 'input', 'headset']):
                return device_id
        
        # Return first available input device
        return input_devices[0][0]
    
    @staticmethod
    def run_full_diagnosis() -> str:
        """Run complete audio diagnosis and return formatted report."""
        diagnosis = AudioDiagnostics.diagnose_audio_devices()
        
        report = []
        report.append("🔍 AUDIO DEVICE DIAGNOSIS REPORT")
        report.append("=" * 50)
        
        # System Info
        system_info = diagnosis.get("system_info", {})
        report.append(f"\n📱 System: {system_info.get('platform', 'Unknown')} {system_info.get('platform_version', '')}")
        report.append(f"🐍 Python: {system_info.get('python_version', 'Unknown')}")
        report.append(f"🔊 SoundDevice: {system_info.get('sounddevice_version', 'Unknown')}")
        
        # Device List
        devices = diagnosis.get("devices", [])
        report.append(f"\n🎧 Found {len(devices)} audio device(s):")
        
        if devices:
            for i, device in enumerate(devices):
                name = device.get('name', 'Unknown')
                max_inputs = device.get('max_input_channels', 0)
                max_outputs = device.get('max_output_channels', 0)
                sample_rate = device.get('default_samplerate', 'Unknown')
                
                status = "📥 INPUT" if max_inputs > 0 else ""
                if max_outputs > 0:
                    status += " 📤 OUTPUT" if status else "📤 OUTPUT"
                
                report.append(f"  [{i}] {name} - {status} - {sample_rate}Hz")
        else:
            report.append("  ❌ No audio devices found!")
        
        # Default Devices
        default_input = diagnosis.get("default_input")
        default_output = diagnosis.get("default_output")
        
        report.append(f"\n🎤 Default Input Device: {default_input if default_input is not None else '❌ None'}")
        report.append(f"🔊 Default Output Device: {default_output if default_output is not None else '❌ None'}")
        
        # Test Recording
        report.append("\n🧪 Testing Audio Recording...")
        test_success, test_message = AudioDiagnostics.test_audio_recording(0.5)
        report.append(f"   {'✅' if test_success else '❌'} {test_message}")
        
        # Suggested Device
        if devices:
            suggested_device = AudioDiagnostics.suggest_device_selection(devices)
            if suggested_device is not None:
                device_name = devices[suggested_device].get('name', 'Unknown')
                report.append(f"\n💡 Suggested Input Device: [{suggested_device}] {device_name}")
        
        # Errors
        errors = diagnosis.get("errors", [])
        if errors:
            report.append("\n❌ ERRORS FOUND:")
            for error in errors:
                report.append(f"   • {error}")
        
        # Recommendations
        recommendations = diagnosis.get("recommendations", [])
        if recommendations:
            report.append("\n💡 TROUBLESHOOTING RECOMMENDATIONS:")
            for rec in recommendations:
                report.append(f"   {rec}")
        
        report.append("\n" + "=" * 50)
        report.append("For more help, visit: https://github.com/spatialaudio/python-sounddevice/blob/master/TROUBLESHOOTING.rst")
        
        return "\n".join(report)


# Standalone diagnostic script
if __name__ == "__main__":
    print(AudioDiagnostics.run_full_diagnosis())