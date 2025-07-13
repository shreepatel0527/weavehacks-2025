import streamlit as st
from claude_interface import ClaudeInterface
from gemini_interface import GeminiInterface
from speech_recognition_module import SpeechRecognizer
from sensor_data_module import SensorDataCollector, SensorReading, SensorType
from safety_integration import IntegratedLabSafetySystem
from experiment_config import ExperimentManager
from datetime import datetime
import tempfile
import os
import threading
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
    
    # Lab Safety System Integration
    if 'lab_safety_system' not in st.session_state:
        st.session_state.lab_safety_system = IntegratedLabSafetySystem()
    if 'safety_monitoring_active' not in st.session_state:
        st.session_state.safety_monitoring_active = False
    if 'sensor_data_display' not in st.session_state:
        st.session_state.sensor_data_display = True
    if 'current_experiment' not in st.session_state:
        st.session_state.current_experiment = None
    
    # Test connections only once at startup
    if 'claude_status' not in st.session_state:
        st.session_state.claude_status = st.session_state.claude.test_connection()
    if 'gemini_status' not in st.session_state:
        st.session_state.gemini_status = st.session_state.gemini.test_connection()


def display_lab_context_only():
    """Display only the Lab Assistant Context messages."""
    # Filter messages to show only Lab Assistant Context
    context_messages = [
        msg for msg in st.session_state.messages 
        if msg.get("model") == "Lab Assistant Context" or msg.get("role") == "system"
    ]
    
    if context_messages:
        st.subheader("🧪 Current Lab Context")
        for message in context_messages[-1:]:  # Show only the latest context
            with st.expander("Lab Assistant Context", expanded=True):
                st.markdown(message["content"])
    else:
        st.info("No lab context available. Select an experiment to generate context.")


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


def display_sensor_dashboard():
    """Display real-time sensor data dashboard."""
    st.subheader("🔬 Lab Sensor Dashboard")
    
    lab_system = st.session_state.lab_safety_system
    
    # Experiment selection
    st.subheader("🧪 Select Experiment")
    available_experiments = lab_system.get_available_experiments()
    experiment_names = {}
    
    for exp_id in available_experiments:
        exp_config = lab_system.experiment_manager.get_experiment_config(exp_id)
        experiment_names[exp_id] = exp_config.name
    
    selected_exp = st.selectbox(
        "Choose experiment type:",
        options=list(experiment_names.keys()),
        format_func=lambda x: experiment_names[x],
        index=0 if not st.session_state.current_experiment else list(experiment_names.keys()).index(st.session_state.current_experiment)
    )
    
    # Update experiment if changed
    if selected_exp != st.session_state.current_experiment:
        if lab_system.set_experiment(selected_exp):
            st.session_state.current_experiment = selected_exp
            st.success(f"Experiment set to: {experiment_names[selected_exp]}")
            if st.session_state.safety_monitoring_active:
                st.info("Safety thresholds updated for current experiment")
            st.rerun()
    
    # Display current experiment info
    if st.session_state.current_experiment:
        exp_info = lab_system.get_current_experiment_info()
        if exp_info:
            with st.expander("📋 Current Experiment Details", expanded=False):
                st.write(f"**Name:** {exp_info['name']}")
                st.write(f"**Type:** {exp_info['type']}")
                st.write(f"**Description:** {exp_info['description']}")
                st.write(f"**Temperature Range:** {exp_info['temperature_range']}")
                st.write(f"**Pressure Range:** {exp_info['pressure_range']}")
                st.write(f"**Duration:** {exp_info['duration_hours']} hours")
                if exp_info['special_notes']:
                    st.write(f"**Special Notes:** {exp_info['special_notes']}")
    
    st.divider()
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🟢 Start Monitoring", disabled=st.session_state.safety_monitoring_active):
            if not st.session_state.current_experiment:
                st.error("Please select an experiment first!")
            else:
                lab_system.start_system()
                st.session_state.safety_monitoring_active = True
                st.success("Lab safety monitoring started!")
                st.rerun()
    
    with col2:
        if st.button("🔴 Stop Monitoring", disabled=not st.session_state.safety_monitoring_active):
            lab_system.stop_system()
            st.session_state.safety_monitoring_active = False
            st.info("Lab safety monitoring stopped.")
            st.rerun()
    
    with col3:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # Display system status
    if st.session_state.safety_monitoring_active:
        status = lab_system.get_system_status()
        
        # Status metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Sensors", status['sensor_system']['active_sensors'])
        
        with col2:
            st.metric("Queue Size", status['sensor_system']['queue_size'])
        
        with col3:
            st.metric("Active Alerts", len(status['safety_system']['active_alerts']))
        
        with col4:
            st.metric("Total Readings", status['sensor_system']['history_size'])
        
        # Recent sensor readings
        if status['sensor_system']['recent_readings']:
            st.subheader("📊 Real-time Sensor Data")
            
            # Convert to DataFrame for visualization
            readings_data = []
            for reading in status['sensor_system']['recent_readings']:
                # Handle both string and enum types for sensor_type
                sensor_type = reading['sensor_type']
                if hasattr(sensor_type, 'value'):
                    # It's an enum, get the value
                    type_display = sensor_type.value.title()
                elif isinstance(sensor_type, str):
                    # It's already a string
                    type_display = sensor_type.title()
                else:
                    # Fallback
                    type_display = str(sensor_type).title()
                
                readings_data.append({
                    'Timestamp': pd.to_datetime(reading['timestamp']),
                    'Sensor': reading['sensor_id'],
                    'Type': type_display,
                    'Value': reading['value'],
                    'Units': reading['units'],
                    'Location': reading.get('location', 'lab')
                })
            
            df = pd.DataFrame(readings_data)
            
            if not df.empty:
                # Sort by timestamp
                df = df.sort_values('Timestamp')
                
                # Create separate plots for temperature and pressure
                temp_data = df[df['Type'] == 'Temperature']
                pressure_data = df[df['Type'] == 'Pressure']
                
                # Get current experiment info for safety ranges
                current_exp_info = lab_system.get_current_experiment_info()
                
                # Temperature plot with safety ranges
                if not temp_data.empty and current_exp_info:
                    st.subheader("🌡️ Temperature Monitoring")
                    
                    fig_temp = go.Figure()
                    
                    # Add temperature readings
                    fig_temp.add_trace(go.Scatter(
                        x=temp_data['Timestamp'],
                        y=temp_data['Value'],
                        mode='lines+markers',
                        name='Temperature',
                        line=dict(color='red', width=2),
                        marker=dict(size=4)
                    ))
                    
                    # Add safety range bands
                    temp_range = current_exp_info['temperature_range'].split('-')
                    if len(temp_range) == 2:
                        temp_min = float(temp_range[0])
                        temp_max = float(temp_range[1].split('°')[0])
                        
                        # Safe range
                        fig_temp.add_hline(y=temp_min, line_dash="dash", line_color="green", 
                                         annotation_text=f"Min Safe ({temp_min}°C)")
                        fig_temp.add_hline(y=temp_max, line_dash="dash", line_color="green",
                                         annotation_text=f"Max Safe ({temp_max}°C)")
                        
                        # Warning zones
                        fig_temp.add_hrect(y0=temp_min-2, y1=temp_min, fillcolor="yellow", opacity=0.2, 
                                         annotation_text="Warning Zone", annotation_position="top left")
                        fig_temp.add_hrect(y0=temp_max, y1=temp_max+2, fillcolor="yellow", opacity=0.2)
                        
                        # Critical zones
                        fig_temp.add_hrect(y0=temp_min-5, y1=temp_min-2, fillcolor="red", opacity=0.2,
                                         annotation_text="Critical Zone", annotation_position="top left")
                        fig_temp.add_hrect(y0=temp_max+2, y1=temp_max+5, fillcolor="red", opacity=0.2)
                    
                    fig_temp.update_layout(
                        title=f"Temperature - {current_exp_info['name']}",
                        xaxis_title="Time",
                        yaxis_title="Temperature (°C)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_temp, use_container_width=True)
                
                # Pressure plot with safety ranges  
                if not pressure_data.empty and current_exp_info:
                    st.subheader("📊 Pressure Monitoring")
                    
                    fig_pressure = go.Figure()
                    
                    # Add pressure readings
                    fig_pressure.add_trace(go.Scatter(
                        x=pressure_data['Timestamp'],
                        y=pressure_data['Value'],
                        mode='lines+markers',
                        name='Pressure',
                        line=dict(color='blue', width=2),
                        marker=dict(size=4)
                    ))
                    
                    # Add safety range bands
                    pressure_range = current_exp_info['pressure_range'].split('-')
                    if len(pressure_range) == 2:
                        pressure_min = float(pressure_range[0])
                        pressure_max = float(pressure_range[1].split('k')[0])
                        
                        # Safe range
                        fig_pressure.add_hline(y=pressure_min, line_dash="dash", line_color="green",
                                             annotation_text=f"Min Safe ({pressure_min} kPa)")
                        fig_pressure.add_hline(y=pressure_max, line_dash="dash", line_color="green",
                                             annotation_text=f"Max Safe ({pressure_max} kPa)")
                        
                        # Warning zones
                        fig_pressure.add_hrect(y0=pressure_min-1.5, y1=pressure_min, fillcolor="yellow", opacity=0.2)
                        fig_pressure.add_hrect(y0=pressure_max, y1=pressure_max+1.5, fillcolor="yellow", opacity=0.2)
                        
                        # Critical zones
                        fig_pressure.add_hrect(y0=pressure_min-4, y1=pressure_min-1.5, fillcolor="red", opacity=0.2)
                        fig_pressure.add_hrect(y0=pressure_max+1.5, y1=pressure_max+4, fillcolor="red", opacity=0.2)
                    
                    fig_pressure.update_layout(
                        title=f"Pressure - {current_exp_info['name']}",
                        xaxis_title="Time",
                        yaxis_title="Pressure (kPa)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_pressure, use_container_width=True)
                
                # Summary statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    if not temp_data.empty:
                        st.metric(
                            "Current Temperature",
                            f"{temp_data.iloc[-1]['Value']:.1f}°C",
                            f"{temp_data.iloc[-1]['Value'] - temp_data.iloc[-2]['Value']:.1f}" if len(temp_data) > 1 else None
                        )
                
                with col2:
                    if not pressure_data.empty:
                        st.metric(
                            "Current Pressure", 
                            f"{pressure_data.iloc[-1]['Value']:.1f} kPa",
                            f"{pressure_data.iloc[-1]['Value'] - pressure_data.iloc[-2]['Value']:.1f}" if len(pressure_data) > 1 else None
                        )
                
                # Data table
                with st.expander("📋 Detailed Readings", expanded=False):
                    # Format timestamp for better display
                    df_display = df.copy()
                    df_display['Time'] = df_display['Timestamp'].dt.strftime('%H:%M:%S')
                    df_display = df_display[['Time', 'Type', 'Value', 'Units', 'Sensor']].sort_values('Time', ascending=False)
                    st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Safety alerts
        recent_alerts = lab_system.get_recent_alerts(count=5)
        if recent_alerts:
            st.subheader("🚨 Recent Safety Alerts")
            
            for alert in recent_alerts[-3:]:  # Show last 3 alerts
                alert_color = {
                    'warning': '🟡',
                    'critical': '🟠', 
                    'emergency': '🔴'
                }.get(alert['level'], '⚪')
                
                st.warning(f"{alert_color} **{alert['level'].upper()}** - "
                          f"{alert['parameter']}: {alert['value']} - "
                          f"{alert['message']}")
    
    else:
        if not st.session_state.current_experiment:
            st.warning("⚠️ Please select an experiment to configure safety monitoring.")
        else:
            st.info("🔬 Lab safety monitoring is not active. Click 'Start Monitoring' to begin.")


def display_ai_lab_assistant():
    """Display combined AI chat and lab protocol assistance interface."""
    st.subheader("💬 AI Lab Assistant")
    
    # Get current experiment info for context
    lab_system = st.session_state.lab_safety_system
    current_exp_info = lab_system.get_current_experiment_info()
    
    # Display current experiment context
    if current_exp_info:
        with st.expander("🧪 Current Experiment Context", expanded=False):
            st.write(f"**{current_exp_info['name']}**")
            st.write(f"Type: {current_exp_info['type']}")
            st.write(f"Temperature: {current_exp_info['temperature_range']}")
            st.write(f"Pressure: {current_exp_info['pressure_range']}")
            if current_exp_info['special_notes']:
                st.write(f"Notes: {current_exp_info['special_notes']}")
        
        experiment_context = f"""
Current Experiment: {current_exp_info['name']}
- Type: {current_exp_info['type']}
- Description: {current_exp_info['description']}
- Temperature Range: {current_exp_info['temperature_range']}
- Pressure Range: {current_exp_info['pressure_range']}
- Duration: {current_exp_info['duration_hours']} hours
- Special Notes: {current_exp_info['special_notes']}
"""
    else:
        st.warning("⚠️ No experiment selected. Choose an experiment in the Safety Monitoring tab for context-aware assistance.")
        experiment_context = "No experiment currently selected. Please select an experiment in the Safety Monitoring tab."
    
    # Protocol context for AI
    lab_context = f"""
    You are an AI lab assistant specializing in nanoparticle synthesis protocols and lab automation:
    
    1. Gold nanoparticle synthesis using HAuCl4·3H2O and TOAB
    2. Real-time monitoring with temperature and pressure sensors
    3. Safety protocols for chemical handling with experiment-specific ranges
    4. Data recording for mass, volume, and qualitative observations
    5. Safety monitoring and alert interpretation
    
    {experiment_context}
    
    Provide specific guidance for:
    - Lab procedures and protocols
    - Safety procedures and risk assessment
    - Data interpretation and analysis
    - Equipment operation and troubleshooting
    - Experimental optimization
    
    Be concise, practical, and safety-focused in your responses.
    """
    
    # Add lab context to AI messages (only once)
    if not st.session_state.messages or not any("AI lab assistant" in msg.get("content", "").lower() for msg in st.session_state.messages):
        st.session_state.messages.insert(0, {
            "role": "system", 
            "content": lab_context,
            "timestamp": datetime.now().strftime("%I:%M %p"),
            "model": "Lab Assistant Context"
        })
    
    # AI Model Selection and Configuration
    st.subheader("🤖 AI Configuration")
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
    
    # Check selected AI connection status
    if st.session_state.ai_model == "Claude":
        if not st.session_state.claude_status:
            st.error("⚠️ Claude CLI not found. Please ensure 'claude -p' is available on your system.")
            st.stop()
    elif st.session_state.ai_model == "Google Gemini 2.5 Pro":
        if not st.session_state.gemini_status:
            st.error("⚠️ Gemini API not configured. Please ensure API key exists at $HOME/Documents/Ephemeral/gapi")
            st.stop()
    
    st.divider()
    
    # Quick action buttons
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Protocol Help", use_container_width=True):
            if current_exp_info:
                prompt = f"Explain the protocol steps for {current_exp_info['name']} including safety considerations"
            else:
                prompt = "Explain the general gold nanoparticle synthesis protocol steps"
            process_user_input(prompt)
    
    with col2:
        if st.button("⚠️ Safety Check", use_container_width=True):
            if current_exp_info:
                prompt = f"What safety precautions should I take for {current_exp_info['name']}? Check current temperature and pressure ranges."
            else:
                prompt = "What are the key safety considerations for nanoparticle synthesis?"
            process_user_input(prompt)
    
    with col3:
        if st.button("📊 Data Analysis", use_container_width=True):
            prompt = "Help me interpret the current sensor data and any recent alerts"
            process_user_input(prompt)
    
    st.divider()
    
    # Chat interface - showing only Lab Assistant Context
    st.subheader("💬 Chat Interface")
    
    # Display only lab context
    display_lab_context_only()
    
    # Input area
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
                prompt = st.chat_input("Ask about protocols, safety, data analysis, or general lab questions...")
            
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
            
            with col1:
                prompt = st.chat_input("Ask about protocols, safety, data analysis, or general lab questions...")
            
            # Handle text input
            if prompt:
                process_user_input(prompt)


def main():
    st.set_page_config(
        page_title="Lab AI Assistant",
        page_icon="🔬",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header section
    st.title("🔬 Lab AI Assistant with Safety Monitoring")
    st.markdown("Advanced lab automation with AI chat, sensor monitoring, and safety protocols")
    
    # Main navigation tabs
    tab1, tab2 = st.tabs(["🛡️ Safety Monitoring", "💬 AI Lab Assistant"])
    
    with tab1:
        display_sensor_dashboard()
    
    with tab2:
        display_ai_lab_assistant()
    
    # Sidebar with options
    with st.sidebar:
        st.header("Options")
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
        
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
        
        # Claude status (from cached value)
        claude_status = "✅ Connected" if st.session_state.claude_status else "❌ Not Available"
        st.caption(f"Claude CLI: {claude_status}")
        
        # Gemini status (from cached value)
        gemini_status = "✅ Connected" if st.session_state.gemini_status else "❌ Not Available"
        st.caption(f"Gemini API: {gemini_status}")
        
        st.divider()
        
        st.caption("Chat History")
        st.caption(f"Messages: {len(st.session_state.messages)}")
        
        st.divider()
        
        # Lab Safety System Status
        st.subheader("Lab Safety Status")
        
        if st.session_state.safety_monitoring_active:
            st.success("🟢 Safety monitoring active")
            lab_status = st.session_state.lab_safety_system.get_system_status()
            st.caption(f"Sensors: {lab_status['sensor_system']['active_sensors']}")
            st.caption(f"Alerts: {len(lab_status['safety_system']['active_alerts'])}")
        else:
            st.warning("🟡 Safety monitoring inactive")
        
        st.divider()
        
        st.caption("💡 Tip: Use the Sensor Dashboard for real-time monitoring")
        st.caption("🔬 Lab Protocol tab provides synthesis guidance")


if __name__ == "__main__":
    main()