#!/usr/bin/env python3
"""
Voice input widget for older Streamlit versions
Uses JavaScript audio recording with file upload fallback
"""

import streamlit as st
import streamlit.components.v1 as components
import base64
import tempfile
import os

def create_audio_recorder():
    """Create a custom audio recorder component using JavaScript"""
    
    # JavaScript code for audio recording
    recorder_html = """
    <div id="audio-recorder" style="padding: 20px; border: 1px solid #ddd; border-radius: 10px; text-align: center;">
        <h3>🎤 Voice Recorder</h3>
        <button id="recordButton" onclick="toggleRecording()" style="padding: 10px 20px; font-size: 16px; cursor: pointer;">
            Start Recording
        </button>
        <p id="status" style="margin-top: 10px;">Ready to record</p>
        <audio id="audioPlayback" controls style="display: none; margin-top: 10px;"></audio>
        <input type="hidden" id="audioData" name="audioData" />
    </div>
    
    <script>
    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;
    
    async function toggleRecording() {
        if (!isRecording) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                
                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    audioChunks = [];
                    
                    // Convert to base64
                    const reader = new FileReader();
                    reader.readAsDataURL(audioBlob);
                    reader.onloadend = () => {
                        const base64Audio = reader.result.split(',')[1];
                        
                        // Store in hidden input
                        document.getElementById('audioData').value = base64Audio;
                        
                        // Show playback
                        const audioUrl = URL.createObjectURL(audioBlob);
                        const audio = document.getElementById('audioPlayback');
                        audio.src = audioUrl;
                        audio.style.display = 'block';
                        
                        // Send to Streamlit
                        window.parent.postMessage({
                            type: 'streamlit:setComponentValue',
                            value: base64Audio
                        }, '*');
                    };
                };
                
                mediaRecorder.start();
                isRecording = true;
                document.getElementById('recordButton').textContent = 'Stop Recording';
                document.getElementById('status').textContent = '🔴 Recording...';
                
            } catch (err) {
                document.getElementById('status').textContent = '❌ Error: ' + err.message;
                alert('Microphone access denied. Please allow microphone access and try again.');
            }
        } else {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
            isRecording = false;
            document.getElementById('recordButton').textContent = 'Start Recording';
            document.getElementById('status').textContent = '✅ Recording complete';
        }
    }
    </script>
    """
    
    return recorder_html

def voice_input_widget(key="voice_input"):
    """
    Voice input widget that works with older Streamlit versions
    Returns base64 encoded audio data or None
    """
    
    # Container for the widget
    container = st.container()
    
    with container:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### 🎤 Voice Input")
            
            # Method 1: Try to use native audio_input if available
            if hasattr(st, 'audio_input'):
                audio_data = st.audio_input("Record your measurement", key=f"{key}_native")
                if audio_data:
                    return audio_data.read()
            
            # Method 2: Custom JavaScript recorder
            else:
                # Try custom component
                try:
                    components.html(create_audio_recorder(), height=250, key=f"{key}_js")
                    
                    # Also provide file upload as fallback
                    with st.expander("Can't use microphone? Upload audio file instead"):
                        uploaded_file = st.file_uploader(
                            "Choose an audio file",
                            type=['wav', 'mp3', 'ogg', 'm4a', 'webm'],
                            key=f"{key}_upload"
                        )
                        if uploaded_file:
                            return uploaded_file.read()
                
                except Exception as e:
                    st.error(f"Audio recording not supported: {e}")
                    
                    # Fallback to file upload only
                    uploaded_file = st.file_uploader(
                        "Upload an audio file",
                        type=['wav', 'mp3', 'ogg', 'm4a', 'webm'],
                        key=f"{key}_upload_fallback"
                    )
                    if uploaded_file:
                        return uploaded_file.read()
        
        with col2:
            # Instructions
            st.markdown("""
            **Tips:**
            - Allow microphone access
            - Speak clearly
            - Keep it short (< 10s)
            """)
    
    return None

def demo_voice_input():
    """Demo of the voice input widget"""
    st.title("Voice Input Widget Demo")
    
    # Check Streamlit version
    st.write(f"Streamlit version: {st.__version__}")
    
    # Use the voice input widget
    audio_data = voice_input_widget()
    
    if audio_data:
        st.success("Audio recorded successfully!")
        
        # Save to temp file for processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name
        
        # Display audio
        st.audio(audio_data, format='audio/wav')
        
        # Here you would process with Whisper
        st.info(f"Audio saved to: {tmp_path}")
        st.write("Process this with Whisper for transcription")
        
        # Clean up
        os.unlink(tmp_path)

if __name__ == "__main__":
    demo_voice_input()