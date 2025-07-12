import streamlit as st
from claude_interface import ClaudeInterface
from gemini_interface import GeminiInterface
from speech_recognition_module import SpeechRecognizer
from datetime import datetime
import tempfile
import os

# Try to import whisper for browser audio support
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'claude' not in st.session_state:
        st.session_state.claude = ClaudeInterface()
    if 'gemini' not in st.session_state:
        st.session_state.gemini = GeminiInterface()
    if 'speech_recognizer' not in st.session_state:
        st.session_state.speech_recognizer = SpeechRecognizer(model_size="base")
    if 'whisper_model' not in st.session_state and WHISPER_AVAILABLE:
        with st.spinner("Loading Whisper model..."):
            st.session_state.whisper_model = whisper.load_model("base")
    if 'audio_input_active' not in st.session_state:
        st.session_state.audio_input_active = False
    if 'ai_model' not in st.session_state:
        st.session_state.ai_model = "Claude"
    if 'audio_method' not in st.session_state:
        st.session_state.audio_method = "System Microphone"


def display_chat_history():
    """Display the chat history."""
    # Create a container for chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "timestamp" in message:
                    caption_text = message["timestamp"]
                    if "model" in message:
                        caption_text += f" • {message['model']}"
                    st.caption(caption_text)


def process_user_input(prompt: str):
    """Process user input and get AI response."""
    # Add user message to chat history
    timestamp = datetime.now().strftime("%I:%M %p")
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })
    
    # Get AI response based on selected model
    model_name = st.session_state.ai_model
    
    if model_name == "Claude":
        ai_interface = st.session_state.claude
    else:  # Google Gemini 2.5 Pro
        ai_interface = st.session_state.gemini
    
    # Get the response
    success, response = ai_interface.send_message(prompt)
    
    if success:
        response_timestamp = datetime.now().strftime("%I:%M %p")
        # Add assistant message to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": response_timestamp,
            "model": model_name
        })
    else:
        st.error(response)
    
    # Force a rerun to display the new messages
    st.rerun()


def transcribe_browser_audio(audio_bytes):
    """Transcribe audio bytes using Whisper (for browser audio)."""
    if not WHISPER_AVAILABLE:
        return False, "Whisper not available. Install with: pip install openai-whisper"
    
    try:
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        # Transcribe
        result = st.session_state.whisper_model.transcribe(
            tmp_path,
            language="en",
            fp16=False
        )
        
        # Clean up
        os.unlink(tmp_path)
        
        text = result["text"].strip()
        return bool(text), text if text else "No speech detected"
        
    except Exception as e:
        return False, f"Transcription error: {str(e)}"


def main():
    st.set_page_config(
        page_title="AI Chat Interface",
        page_icon="💬",
        layout="centered"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header section
    st.title("AI Chat Interface")
    st.markdown("Chat with Claude or Google Gemini using local credentials")
    
    # Configuration row
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # AI Model Selection
        selected_model = st.radio(
            "Select AI Model:",
            ["Claude", "Google Gemini 2.5 Pro"],
            horizontal=True,
            index=0 if st.session_state.ai_model == "Claude" else 1,
            key="ai_model_radio"
        )
        
        # Update session state only if changed
        if selected_model != st.session_state.ai_model:
            st.session_state.ai_model = selected_model
    
    with col2:
        # Audio input method selection
        st.session_state.audio_method = st.radio(
            "Audio Input:",
            ["System Microphone", "Browser Recording"],
            horizontal=True,
            help="System Microphone: Direct recording\nBrowser Recording: Uses browser's audio input"
        )
    
    # Test selected AI connection
    if st.session_state.ai_model == "Claude":
        if not st.session_state.claude.test_connection():
            st.error("⚠️ Claude CLI not found. Please ensure 'claude -p' is available on your system.")
            st.stop()
    elif st.session_state.ai_model == "Google Gemini 2.5 Pro":
        if not st.session_state.gemini.test_connection():
            st.error("⚠️ Gemini API not configured. Please ensure API key exists at $HOME/Documents/Ephemeral/gapi")
            st.stop()
    
    # Create main chat area with proper spacing
    st.markdown("---")
    
    # Chat messages area (this will expand as needed)
    display_chat_history()
    
    # Add some space before the input area
    st.markdown("<br>" * 2, unsafe_allow_html=True)
    
    # Bottom input area - fixed at bottom
    input_container = st.container()
    
    with input_container:
        # Handle system microphone popup
        if st.session_state.audio_input_active and st.session_state.audio_method == "System Microphone":
            st.session_state.audio_input_active = False
            
            with st.container():
                st.info("🎤 Click the button below to start recording (5 seconds max)")
                
                if st.button("Start Recording", type="primary", use_container_width=True):
                    # Record and transcribe
                    success, text = st.session_state.speech_recognizer.record_and_transcribe(duration=5.0)
                    
                    if success:
                        st.success(f"Transcribed: {text}")
                        # Process the transcribed text
                        process_user_input(text)
                    else:
                        st.error(text)
        
        # Input area based on selected audio method
        if st.session_state.audio_method == "Browser Recording":
            # Browser audio input layout
            col1, col2 = st.columns([10, 1])
            
            with col1:
                prompt = st.chat_input("Type your message here...")
            
            with col2:
                # Audio recorder using Streamlit's native audio_input
                audio_bytes = st.audio_input("🎤", label_visibility="collapsed")
            
            # Handle text input
            if prompt:
                process_user_input(prompt)
            
            # Handle browser audio input
            if audio_bytes is not None:
                with st.spinner("Processing audio..."):
                    success, text = transcribe_browser_audio(audio_bytes.read())
                    
                    if success:
                        st.success(f"Transcribed: {text}")
                        process_user_input(text)
                    else:
                        st.error(text)
        else:
            # System microphone layout
            col1, col2 = st.columns([10, 1])
            
            with col2:
                if st.button("🎤", help="Click to record speech", use_container_width=True):
                    st.session_state.audio_input_active = True
                    st.rerun()
            
            with col1:
                prompt = st.chat_input("Type your message here...")
            
            # Handle text input
            if prompt:
                process_user_input(prompt)
    
    # Sidebar with options
    with st.sidebar:
        st.header("Options")
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        st.subheader("Speech Settings")
        
        if st.session_state.audio_method == "System Microphone":
            model_size = st.selectbox(
                "Whisper Model Size",
                ["tiny", "base", "small", "medium"],
                index=1,
                help="Larger models are more accurate but slower"
            )
            
            if model_size != st.session_state.speech_recognizer.model_size:
                st.session_state.speech_recognizer = SpeechRecognizer(model_size=model_size)
                st.info(f"Switched to {model_size} model")
        else:
            if WHISPER_AVAILABLE:
                st.info("Browser audio uses Whisper base model")
            else:
                st.warning("Whisper not installed for browser audio")
        
        st.divider()
        
        st.subheader("Connection Status")
        
        # Claude status
        claude_status = "✅ Connected" if st.session_state.claude.test_connection() else "❌ Not Available"
        st.caption(f"Claude CLI: {claude_status}")
        
        # Gemini status
        gemini_status = "✅ Connected" if st.session_state.gemini.test_connection() else "❌ Not Available"
        st.caption(f"Gemini API: {gemini_status}")
        
        st.divider()
        
        st.caption("Chat History")
        st.caption(f"Messages: {len(st.session_state.messages)}")
        
        st.divider()
        st.caption("💡 Tip: Check stderr/console for verbose AI logs")


if __name__ == "__main__":
    main()