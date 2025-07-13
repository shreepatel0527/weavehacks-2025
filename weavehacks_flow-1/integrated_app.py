#!/usr/bin/env python3
"""
WeaveHacks 2025 - Integrated Lab Automation Platform
Connects Prototype-1 Streamlit frontend with weavehacks_flow-1 backend agents
"""

import streamlit as st
import requests
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import threading
import time
import weave

# Import components via integration bridge
from integration_bridge import (
    ClaudeInterface,
    GeminiInterface,
    SpeechRecognizer,
    DataCollectionAgent,
    LabControlAgent,
    EnhancedSafetyMonitoringAgent
)

# Backend API Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize Weave for observability
try:
    weave.init('weavehacks-integrated-app')
    print("Weave initialized for integrated app")
except Exception as e:
    print(f"Weave init failed (optional): {e}")

class IntegratedLabPlatform:
    """Main integration class connecting frontend and backend"""
    
    def __init__(self):
        self.api_base = API_BASE_URL
        self.data_agent = DataCollectionAgent()
        self.lab_agent = LabControlAgent()
        self.safety_agent = EnhancedSafetyMonitoringAgent()
        
    @weave.op()
    def create_experiment(self, experiment_id: str):
        """Create a new experiment via API"""
        try:
            # The API expects experiment_id as a query parameter based on the backend code
            response = requests.post(f"{self.api_base}/experiments?experiment_id={experiment_id}")
            if response.status_code == 200:
                return True, response.json()
            else:
                # Get more details about the error
                error_detail = ""
                try:
                    error_detail = response.json().get('detail', '')
                except:
                    error_detail = response.text
                return False, f"API Error {response.status_code}: {error_detail}"
        except Exception as e:
            return False, f"Connection Error: {str(e)}"
    
    @weave.op()
    def get_experiment(self, experiment_id: str):
        """Get experiment details via API"""
        try:
            response = requests.get(f"{self.api_base}/experiments/{experiment_id}")
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, f"API Error: {response.status_code}"
        except Exception as e:
            return False, f"Connection Error: {str(e)}"
    
    @weave.op()
    def record_data_via_api(self, experiment_id: str, data_type: str, compound: str, value: float, units: str):
        """Record experimental data via API"""
        data = {
            "experiment_id": experiment_id,
            "data_type": data_type,
            "compound": compound,
            "value": value,
            "units": units
        }
        try:
            response = requests.post(f"{self.api_base}/data", json=data)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, f"API Error: {response.status_code}"
        except Exception as e:
            return False, f"Connection Error: {str(e)}"
    
    @weave.op()
    def create_safety_alert(self, experiment_id: str, parameter: str, value: float, threshold: float, severity: str):
        """Create safety alert via API"""
        alert = {
            "experiment_id": experiment_id,
            "parameter": parameter,
            "value": value,
            "threshold": threshold,
            "severity": severity
        }
        try:
            response = requests.post(f"{self.api_base}/safety/alert", json=alert)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, f"API Error: {response.status_code}"
        except Exception as e:
            return False, f"Connection Error: {str(e)}"
    
    @weave.op()
    def calculate_chemistry(self, calculation_type: str, experiment_id: str, **kwargs):
        """Perform chemistry calculations via API"""
        try:
            endpoint = f"{self.api_base}/calculations/{calculation_type}"
            data = {"experiment_id": experiment_id, **kwargs}
            response = requests.post(endpoint, json=data)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, f"API Error: {response.status_code}"
        except Exception as e:
            return False, f"Connection Error: {str(e)}"


def initialize_session_state():
    """Initialize session state with integrated platform"""
    if 'platform' not in st.session_state:
        st.session_state.platform = IntegratedLabPlatform()
    
    if 'current_experiment_id' not in st.session_state:
        st.session_state.current_experiment_id = None
    
    if 'current_experiment' not in st.session_state:
        st.session_state.current_experiment = None
    
    # AI interfaces
    if 'claude' not in st.session_state:
        st.session_state.claude = ClaudeInterface()
    if 'gemini' not in st.session_state:
        st.session_state.gemini = GeminiInterface()
    if 'speech_recognizer' not in st.session_state:
        st.session_state.speech_recognizer = SpeechRecognizer(model_size="base")
    
    # UI state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'ai_model' not in st.session_state:
        st.session_state.ai_model = "Claude"


@weave.op()
def process_voice_data_entry(transcript: str, experiment_id: str):
    """Process voice input for data entry using integrated agents"""
    platform = st.session_state.platform
    
    # Use data collection agent to interpret the transcript
    # Parse common patterns like "gold mass is 0.1576 grams"
    transcript_lower = transcript.lower()
    
    # Compound mapping
    compound_mapping = {
        'gold': 'HAuCl4·3H2O',
        'haucl4': 'HAuCl4·3H2O',
        'toab': 'TOAB',
        'sulfur': 'PhCH2CH2SH',
        'nabh4': 'NaBH4',
        'toluene': 'toluene',
        'nanopure': 'nanopure water',
        'water': 'nanopure water',
        'final': 'Au25 nanoparticles'
    }
    
    # Extract value and units
    import re
    
    # Pattern for numbers followed by units
    number_pattern = r'(\d+\.?\d*)\s*(g|gram|grams|ml|milliliter|milliliters|mL)'
    matches = re.findall(number_pattern, transcript_lower)
    
    if matches:
        value, units = matches[0]
        value = float(value)
        
        # Normalize units
        if units.startswith('g'):
            units = 'g'
        elif units.startswith('ml') or units == 'mL':
            units = 'mL'
        
        # Find compound
        compound = None
        for key, mapped_compound in compound_mapping.items():
            if key in transcript_lower:
                compound = mapped_compound
                break
        
        if compound:
            data_type = "mass" if units == 'g' else "volume"
            success, result = platform.record_data_via_api(
                experiment_id, data_type, compound, value, units
            )
            
            if success:
                return True, f"Recorded {compound}: {value} {units}"
            else:
                return False, f"Failed to record data: {result}"
        else:
            return False, "Could not identify compound from voice input"
    else:
        return False, "Could not extract numerical value from voice input"


def display_experiment_dashboard():
    """Enhanced experiment dashboard with API integration"""
    st.subheader("🧪 Experiment Management")
    
    platform = st.session_state.platform
    
    # Experiment creation/selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        new_exp_id = st.text_input("New Experiment ID", placeholder="exp_nanoparticle_001")
    
    with col2:
        if st.button("Create Experiment"):
            if new_exp_id:
                success, result = platform.create_experiment(new_exp_id)
                if success:
                    st.session_state.current_experiment_id = new_exp_id
                    st.session_state.current_experiment = result
                    st.success(f"Created experiment: {new_exp_id}")
                    st.rerun()
                else:
                    st.error(f"Failed to create experiment: {result}")
    
    # Current experiment display
    if st.session_state.current_experiment_id:
        exp = st.session_state.current_experiment
        
        st.info(f"**Current Experiment:** {exp['experiment_id']}")
        
        # Progress tracking
        progress = exp['step_num'] / 12 * 100
        st.progress(progress / 100, f"Step {exp['step_num']}/12 ({progress:.0f}%)")
        
        # Data display
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Gold Mass (g)", f"{exp['mass_gold']:.4f}")
            st.metric("TOAB Mass (g)", f"{exp['mass_toab']:.4f}")
            st.metric("Sulfur Mass (g)", f"{exp['mass_sulfur']:.4f}")
        
        with col2:
            st.metric("NaBH4 Mass (g)", f"{exp['mass_nabh4']:.4f}")
            st.metric("Toluene Volume (mL)", f"{exp['volume_toluene']:.2f}")
            st.metric("Final Mass (g)", f"{exp['mass_final']:.4f}")
        
        # Chemistry calculations
        st.subheader("🧮 Chemistry Calculations")
        
        calc_col1, calc_col2, calc_col3 = st.columns(3)
        
        with calc_col1:
            if st.button("Calculate Sulfur Amount"):
                if exp['mass_gold'] > 0:
                    success, result = platform.calculate_chemistry(
                        "sulfur_amount", exp['experiment_id'], gold_mass=exp['mass_gold']
                    )
                    if success:
                        st.success(f"Sulfur needed: {result.get('mass_sulfur_g', 0):.4f}g")
                    else:
                        st.error(f"Calculation failed: {result}")
                else:
                    st.warning("Please record gold mass first")
        
        with calc_col2:
            if st.button("Calculate NaBH4 Amount"):
                if exp['mass_gold'] > 0:
                    success, result = platform.calculate_chemistry(
                        "nabh4_amount", exp['experiment_id'], gold_mass=exp['mass_gold']
                    )
                    if success:
                        st.success(f"NaBH4 needed: {result.get('mass_nabh4_g', 0):.4f}g")
                    else:
                        st.error(f"Calculation failed: {result}")
                else:
                    st.warning("Please record gold mass first")
        
        with calc_col3:
            if st.button("Calculate Percent Yield"):
                if exp['mass_gold'] > 0 and exp['mass_final'] > 0:
                    success, result = platform.calculate_chemistry(
                        "percent_yield", exp['experiment_id'], 
                        gold_mass=exp['mass_gold'], actual_yield=exp['mass_final']
                    )
                    if success:
                        st.success(f"Percent yield: {result['percent_yield']:.2f}%")
                    else:
                        st.error(f"Calculation failed: {result}")
                else:
                    st.warning("Please record gold and final masses first")
    
    else:
        st.warning("No experiment selected. Create a new experiment to begin.")


def display_voice_data_entry():
    """Voice data entry interface with API integration"""
    st.subheader("🎤 Voice Data Entry")
    
    if not st.session_state.current_experiment_id:
        st.warning("Please create an experiment first in the Dashboard tab.")
        return
    
    # Voice recording interface
    st.write("**Instructions:** Say something like:")
    st.code("""
    "Gold mass is 0.1576 grams"
    "TOAB mass is 0.25 grams"  
    "Toluene volume is 10 mL"
    "Final nanoparticle mass is 0.08 grams"
    """)
    
    # Audio input - check for compatibility
    audio_bytes = None
    
    # Method 1: Try to use native audio_input if available (Streamlit >= 1.37.0)
    if hasattr(st, 'audio_input'):
        audio_bytes = st.audio_input("🎤 Record your measurement")
    else:
        # Method 2: Fallback for older Streamlit versions
        st.info("💡 Using alternative voice input method")
        
        # Option A: File upload
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Option 1: Upload Audio File**")
            audio_file = st.file_uploader(
                "Choose audio file", 
                type=['wav', 'mp3', 'ogg', 'm4a', 'webm'],
                help="Record with your phone or computer, then upload"
            )
            if audio_file is not None:
                audio_bytes = audio_file
        
        with col2:
            st.markdown("**Option 2: Use Microphone**")
            if st.button("🎤 Open Recorder", help="Opens a popup for recording"):
                st.info("Microphone recording requires Streamlit 1.37.0+")
                st.code("pip install --upgrade streamlit>=1.37.0")
    
    if audio_bytes is not None:
        with st.spinner("Processing audio..."):
            # Transcribe audio using Whisper
            try:
                import whisper
                import tempfile
                import os
                
                # Save audio to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_file.write(audio_bytes.read())
                    tmp_path = tmp_file.name
                
                # Load Whisper model if not already loaded
                if 'whisper_model' not in st.session_state:
                    st.session_state.whisper_model = whisper.load_model("base")
                
                # Transcribe
                result = st.session_state.whisper_model.transcribe(tmp_path, language="en")
                transcript = result["text"].strip()
                
                # Clean up
                os.unlink(tmp_path)
                
                st.success(f"Transcribed: {transcript}")
                
                # Process the transcript
                success, message = process_voice_data_entry(transcript, st.session_state.current_experiment_id)
                
                if success:
                    st.success(message)
                    # Refresh experiment data
                    platform = st.session_state.platform
                    success, updated_exp = platform.get_experiment(st.session_state.current_experiment_id)
                    if success:
                        st.session_state.current_experiment = updated_exp
                        st.rerun()
                else:
                    st.error(message)
                    
            except ImportError:
                st.error("Whisper not installed. Install with: pip install openai-whisper")
            except Exception as e:
                st.error(f"Audio processing error: {str(e)}")
    
    # Manual data entry fallback
    st.subheader("📝 Manual Data Entry")
    
    with st.form("manual_data_entry"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            compound = st.selectbox("Compound", [
                "HAuCl4·3H2O", "TOAB", "PhCH2CH2SH", "NaBH4", 
                "toluene", "nanopure water", "Au25 nanoparticles"
            ])
        
        with col2:
            value = st.number_input("Value", min_value=0.0, step=0.0001, format="%.4f")
        
        with col3:
            units = st.selectbox("Units", ["g", "mL"])
        
        if st.form_submit_button("Record Data"):
            platform = st.session_state.platform
            data_type = "mass" if units == "g" else "volume"
            
            success, result = platform.record_data_via_api(
                st.session_state.current_experiment_id, data_type, compound, value, units
            )
            
            if success:
                st.success(f"Recorded {compound}: {value} {units}")
                # Refresh experiment data
                success, updated_exp = platform.get_experiment(st.session_state.current_experiment_id)
                if success:
                    st.session_state.current_experiment = updated_exp
                    st.rerun()
            else:
                st.error(f"Failed to record data: {result}")


def display_safety_monitoring():
    """Enhanced safety monitoring interface with real-time sensor data"""
    st.subheader("🛡️ Real-Time Safety Monitoring")
    
    if not st.session_state.current_experiment_id:
        st.warning("Please create an experiment first in the Dashboard tab.")
        return
    
    platform = st.session_state.platform
    
    # Get sensor status
    try:
        response = requests.get(f"{API_BASE_URL}/sensors/status")
        if response.status_code == 200:
            sensor_data = response.json()
        else:
            sensor_data = {"experiment": {"active": False}, "recent_readings": [], "safety_alerts": []}
    except:
        sensor_data = {"experiment": {"active": False}, "recent_readings": [], "safety_alerts": []}
    
    # Experiment controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Get available experiment types
        try:
            exp_types_response = requests.get(f"{API_BASE_URL}/sensors/experiment-types")
            if exp_types_response.status_code == 200:
                exp_types_data = exp_types_response.json()
                experiment_types = list(exp_types_data["experiment_types"])
            else:
                experiment_types = ["nanoparticle_synthesis"]
        except:
            experiment_types = ["nanoparticle_synthesis"]
        
        selected_exp_type = st.selectbox("Experiment Type", experiment_types)
    
    with col2:
        if st.button("🟢 Start Sensor Monitoring"):
            try:
                response = requests.post(f"{API_BASE_URL}/sensors/start-experiment", 
                                       params={"experiment_type": selected_exp_type, 
                                              "experiment_id": st.session_state.current_experiment_id})
                if response.status_code == 200:
                    st.success("Sensor monitoring started!")
                    st.rerun()
                else:
                    st.error("Failed to start monitoring")
            except Exception as e:
                st.error(f"Connection error: {e}")
    
    with col3:
        if st.button("🔴 Stop Monitoring"):
            try:
                response = requests.post(f"{API_BASE_URL}/sensors/stop")
                if response.status_code == 200:
                    st.info("Sensor monitoring stopped")
                    st.rerun()
                else:
                    st.error("Failed to stop monitoring")
            except Exception as e:
                st.error(f"Connection error: {e}")
    
    # Display experiment status
    experiment_status = sensor_data.get("experiment", {})
    if experiment_status.get("active", False):
        st.success(f"🟢 Monitoring Active: {experiment_status.get('experiment_type', 'Unknown')}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            elapsed_hours = experiment_status.get("elapsed_hours", 0)
            st.metric("Elapsed Time", f"{elapsed_hours:.1f}h")
        
        with col2:
            progress = experiment_status.get("progress_percent", 0)
            st.metric("Progress", f"{progress:.1f}%")
        
        with col3:
            expected_hours = experiment_status.get("expected_duration_hours", 0)
            st.metric("Expected Duration", f"{expected_hours}h")
        
        # Progress bar
        st.progress(progress / 100, f"Experiment Progress: {progress:.1f}%")
    
    else:
        st.warning("🟡 No active sensor monitoring")
    
    # Safety alerts
    safety_alerts = sensor_data.get("safety_alerts", [])
    if safety_alerts:
        st.subheader("🚨 Active Safety Alerts")
        for alert in safety_alerts[-3:]:  # Show last 3 alerts
            severity_icon = {"warning": "🟡", "critical": "🔴"}.get(alert["severity"], "⚪")
            st.error(f"{severity_icon} **{alert['severity'].upper()}**: {alert['message']}")
    
    # Real-time sensor data
    recent_readings = sensor_data.get("recent_readings", [])
    if recent_readings:
        st.subheader("📊 Live Sensor Data")
        
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame(recent_readings)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Separate by sensor type
        temp_data = df[df['sensor_type'] == 'temperature']
        pressure_data = df[df['sensor_type'] == 'pressure']
        gas_data = df[df['sensor_type'] == 'gas_level']
        
        # Temperature chart
        if not temp_data.empty:
            st.subheader("🌡️ Temperature Monitoring")
            
            fig_temp = go.Figure()
            
            for sensor_id in temp_data['sensor_id'].unique():
                sensor_data_subset = temp_data[temp_data['sensor_id'] == sensor_id]
                fig_temp.add_trace(go.Scatter(
                    x=sensor_data_subset['timestamp'],
                    y=sensor_data_subset['value'],
                    mode='lines+markers',
                    name=f"{sensor_id} ({sensor_data_subset['location'].iloc[0]})",
                    line=dict(width=2),
                    marker=dict(size=4)
                ))
            
            # Add safety thresholds
            fig_temp.add_hline(y=90, line_dash="dash", line_color="orange", 
                             annotation_text="Warning (90°C)")
            fig_temp.add_hline(y=100, line_dash="dash", line_color="red",
                             annotation_text="Critical (100°C)")
            
            fig_temp.update_layout(
                title="Real-Time Temperature Monitoring",
                xaxis_title="Time",
                yaxis_title="Temperature (°C)",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_temp, use_container_width=True)
            
            # Current temperature metrics
            current_temps = temp_data.groupby('sensor_id')['value'].last()
            cols = st.columns(len(current_temps))
            for i, (sensor_id, temp) in enumerate(current_temps.items()):
                with cols[i]:
                    delta_color = "normal"
                    if temp > 100:
                        delta_color = "inverse"
                    elif temp > 90:
                        delta_color = "off"
                    
                    st.metric(
                        f"{sensor_id}",
                        f"{temp:.1f}°C",
                        delta=None,
                        delta_color=delta_color
                    )
        
        # Pressure chart
        if not pressure_data.empty:
            st.subheader("📊 Pressure Monitoring")
            
            fig_pressure = go.Figure()
            
            for sensor_id in pressure_data['sensor_id'].unique():
                sensor_data_subset = pressure_data[pressure_data['sensor_id'] == sensor_id]
                fig_pressure.add_trace(go.Scatter(
                    x=sensor_data_subset['timestamp'],
                    y=sensor_data_subset['value'],
                    mode='lines+markers',
                    name=f"{sensor_id} ({sensor_data_subset['location'].iloc[0]})",
                    line=dict(width=2),
                    marker=dict(size=4)
                ))
            
            # Add safety thresholds
            fig_pressure.add_hline(y=115, line_dash="dash", line_color="orange",
                                 annotation_text="Warning (115 kPa)")
            fig_pressure.add_hline(y=125, line_dash="dash", line_color="red",
                                 annotation_text="Critical (125 kPa)")
            
            fig_pressure.update_layout(
                title="Real-Time Pressure Monitoring",
                xaxis_title="Time",
                yaxis_title="Pressure (kPa)",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_pressure, use_container_width=True)
            
            # Current pressure metrics
            current_pressures = pressure_data.groupby('sensor_id')['value'].last()
            cols = st.columns(len(current_pressures))
            for i, (sensor_id, pressure) in enumerate(current_pressures.items()):
                with cols[i]:
                    delta_color = "normal"
                    if pressure > 125:
                        delta_color = "inverse"
                    elif pressure > 115:
                        delta_color = "off"
                    
                    st.metric(
                        f"{sensor_id}",
                        f"{pressure:.2f} kPa",
                        delta=None,
                        delta_color=delta_color
                    )
        
        # Gas level monitoring
        if not gas_data.empty:
            st.subheader("💨 Gas Level Monitoring")
            
            current_gas_levels = gas_data.groupby('sensor_id')['value'].last()
            cols = st.columns(len(current_gas_levels))
            for i, (sensor_id, gas_level) in enumerate(current_gas_levels.items()):
                with cols[i]:
                    delta_color = "normal"
                    if gas_level > 5000:
                        delta_color = "inverse"
                    elif gas_level > 1000:
                        delta_color = "off"
                    
                    st.metric(
                        f"{sensor_id}",
                        f"{gas_level:.0f} ppm",
                        delta=None,
                        delta_color=delta_color
                    )
        
        # Data table
        with st.expander("📋 Recent Sensor Readings", expanded=False):
            display_df = df.copy()
            display_df['time'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
            display_df = display_df[['time', 'sensor_id', 'sensor_type', 'value', 'units', 'location']]
            display_df = display_df.sort_values('time', ascending=False)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    else:
        st.info("No sensor data available. Start monitoring to see live data.")
    
    # Auto-refresh
    if st.checkbox("Auto-refresh (every 10 seconds)", value=False):
        time.sleep(10)
        st.rerun()


def display_ai_assistant():
    """AI assistant interface"""
    st.subheader("🤖 AI Lab Assistant")
    
    # Model selection
    model = st.radio("AI Model", ["Claude", "Google Gemini 2.5 Pro"], horizontal=True)
    st.session_state.ai_model = model
    
    # Context about current experiment
    if st.session_state.current_experiment:
        exp = st.session_state.current_experiment
        context = f"""
        Current Experiment: {exp['experiment_id']}
        Step: {exp['step_num']}/12
        Status: {exp['status']}
        Gold mass: {exp['mass_gold']}g
        Safety status: {exp['safety_status']}
        """
        
        with st.expander("🧪 Current Experiment Context"):
            st.code(context)
    
    # Chat interface - check compatibility
    if hasattr(st, 'chat_message') and hasattr(st, 'chat_input'):
        # Modern chat interface (Streamlit >= 1.24.0)
        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your experiment, safety, or protocols..."):
            process_chat = True
        else:
            process_chat = False
            prompt = None
    else:
        # Fallback for older Streamlit versions
        st.info("💬 Chat interface (using compatibility mode)")
        
        # Display messages
        for message in st.session_state.messages:
            if message["role"] != "system":
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**Assistant:** {message['content']}")
        
        # Text input fallback
        prompt = st.text_input("Ask a question:", key="chat_input_fallback")
        process_chat = st.button("Send", key="chat_send_button")
    
    if process_chat and prompt:
        # Add context about current experiment
        if st.session_state.current_experiment:
            exp = st.session_state.current_experiment
            enhanced_prompt = f"""
            Current lab context:
            - Experiment: {exp['experiment_id']} (Step {exp['step_num']}/12)
            - Gold mass: {exp['mass_gold']}g, TOAB: {exp['mass_toab']}g
            - Safety: {exp['safety_status']}
            
            User question: {prompt}
            """
        else:
            enhanced_prompt = prompt
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get AI response
        if model == "Claude":
            ai_interface = st.session_state.claude
        else:
            ai_interface = st.session_state.gemini
        
        success, response = ai_interface.send_message(enhanced_prompt)
        
        if success:
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error(f"AI Error: {response}")
        
        st.rerun()


def display_protocol_steps():
    """Protocol step management interface"""
    if not st.session_state.current_experiment_id:
        st.warning("Please create an experiment first in the Dashboard tab.")
        return
    
    # Import and use step panel
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'frontend', 'components'))
        from step_panel import render_step_panel
        render_step_panel(st.session_state.current_experiment_id)
    except (ImportError, Exception) as e:
        # Fallback simple step panel
        render_simple_step_panel()

def display_data_panel():
    """Data panel interface"""
    if not st.session_state.current_experiment_id:
        st.warning("Please create an experiment first in the Dashboard tab.")
        return
    
    # Import and use data panel
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'frontend', 'components'))
        from data_panel import render_experiment_data_panel
        render_experiment_data_panel(st.session_state.current_experiment_id)
    except (ImportError, Exception) as e:
        # Fallback simple data panel
        render_simple_data_panel()

def render_simple_step_panel():
    """Simple fallback step panel"""
    st.subheader("📋 Protocol Steps")
    
    # Get protocol steps from API
    try:
        response = requests.get(f"{API_BASE_URL}/protocol/steps")
        if response.status_code == 200:
            data = response.json()
            steps = data["steps"]
        else:
            steps = []
    except:
        steps = []
    
    if not steps:
        st.error("Could not load protocol steps")
        return
    
    # Get current experiment step
    platform = st.session_state.platform
    success, exp_data = platform.get_experiment(st.session_state.current_experiment_id)
    current_step = exp_data.get('step_num', 0) if success else 0
    
    # Display current step
    if current_step < len(steps):
        step_info = steps[current_step]
        
        st.markdown(f"### Step {current_step + 1}: {step_info['title']}")
        st.markdown(f"**Description:** {step_info['description']}")
        st.markdown(f"**Details:** {step_info['details']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"⏱️ **Time:** {step_info['estimated_time']}")
        with col2:
            st.warning(f"⚠️ **Safety:** {step_info['safety_notes']}")
        
        # Navigation
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Previous", disabled=current_step <= 0):
                new_step = max(0, current_step - 1)
                update_experiment_step_api(st.session_state.current_experiment_id, new_step)
                st.rerun()
        
        with col2:
            st.metric("Progress", f"{current_step + 1}/{len(steps)}")
        
        with col3:
            if st.button("Next ➡️", disabled=current_step >= len(steps) - 1):
                new_step = min(len(steps) - 1, current_step + 1)
                update_experiment_step_api(st.session_state.current_experiment_id, new_step)
                st.rerun()
    
    # Step overview
    with st.expander("📋 All Steps Overview", expanded=False):
        for i, step in enumerate(steps):
            status = "✅" if i < current_step else "🟡" if i == current_step else "⚪"
            st.write(f"{status} **Step {i+1}:** {step['title']} ({step['estimated_time']})")

def render_simple_data_panel():
    """Simple fallback data panel"""
    st.subheader("📊 Experimental Data")
    
    # Get current experiment data
    platform = st.session_state.platform
    success, exp_data = platform.get_experiment(st.session_state.current_experiment_id)
    
    if not success:
        st.error("Could not load experiment data")
        return
    
    # Create simple data table
    data_items = [
        ("HAuCl₄·3H₂O", exp_data.get('mass_gold', 0), "g"),
        ("TOAB", exp_data.get('mass_toab', 0), "g"),
        ("PhCH₂CH₂SH", exp_data.get('mass_sulfur', 0), "g"),
        ("NaBH₄", exp_data.get('mass_nabh4', 0), "g"),
        ("Final Au₂₅", exp_data.get('mass_final', 0), "g"),
        ("Nanopure (RT)", exp_data.get('volume_nanopure_rt', 0), "mL"),
        ("Toluene", exp_data.get('volume_toluene', 0), "mL"),
        ("Nanopure (Cold)", exp_data.get('volume_nanopure_cold', 0), "mL")
    ]
    
    st.write("**Current Measurements:**")
    
    for substance, value, units in data_items:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(substance)
        with col2:
            if value > 0:
                st.success(f"{value:.4f}" if units == "g" else f"{value:.2f}")
            else:
                st.info("Not recorded")
        with col3:
            st.write(units)
    
    # Quick data entry
    st.subheader("🚀 Quick Entry")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.form("quick_mass_entry"):
            compound = st.selectbox("Compound", [
                "HAuCl₄·3H₂O", "TOAB", "PhCH₂CH₂SH", "NaBH₄", "Au₂₅ nanoparticles"
            ])
            mass_val = st.number_input("Mass (g)", min_value=0.0, step=0.0001)
            
            if st.form_submit_button("Record Mass"):
                data_type = "mass"
                success = record_measurement_simple(compound, mass_val, "g", data_type)
                if success:
                    st.success("Recorded!")
                    st.rerun()
    
    with col2:
        with st.form("quick_volume_entry"):
            liquid = st.selectbox("Liquid", [
                "nanopure water", "toluene", "ice-cold nanopure water"
            ])
            vol_val = st.number_input("Volume (mL)", min_value=0.0, step=0.01)
            
            if st.form_submit_button("Record Volume"):
                data_type = "volume"
                success = record_measurement_simple(liquid, vol_val, "mL", data_type)
                if success:
                    st.success("Recorded!")
                    st.rerun()

def update_experiment_step_api(experiment_id: str, step_num: int):
    """Update experiment step via API"""
    try:
        response = requests.put(f"{API_BASE_URL}/experiments/{experiment_id}/step",
                               json={"step_num": step_num})
        return response.status_code == 200
    except:
        return False

def record_measurement_simple(compound: str, value: float, units: str, data_type: str):
    """Simple measurement recording"""
    platform = st.session_state.platform
    return platform.record_data_via_api(
        st.session_state.current_experiment_id, data_type, compound, value, units
    )[0]


def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="WeaveHacks Lab Platform",
        page_icon="🔬",
        layout="wide"
    )
    
    # Initialize
    initialize_session_state()
    
    # Header
    st.title("🔬 WeaveHacks Lab Automation Platform")
    st.markdown("**Integrated AI-Powered Lab Assistant with Real-time Monitoring**")
    
    # Check backend connection
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            st.success("🟢 Backend API Connected")
        else:
            st.error("🔴 Backend API Error")
    except:
        st.error("🔴 Backend API Not Available - Start with: `uvicorn backend.main:app --reload`")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard", 
        "🎤 Voice Entry", 
        "🛡️ Safety Monitor", 
        "🤖 AI Assistant",
        "📋 Protocol Steps",
        "📈 Data Panel"
    ])
    
    with tab1:
        display_experiment_dashboard()
    
    with tab2:
        display_voice_data_entry()
    
    with tab3:
        display_safety_monitoring()
    
    with tab4:
        display_ai_assistant()
    
    with tab5:
        display_protocol_steps()
    
    with tab6:
        display_data_panel()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        if st.session_state.current_experiment_id:
            st.success(f"Experiment: {st.session_state.current_experiment_id}")
        else:
            st.warning("No active experiment")
        
        st.divider()
        
        # Agent status (simulated)
        st.subheader("🤖 Agent Status")
        st.caption("🟢 Data Collection Agent: Active")
        st.caption("🟢 Lab Control Agent: Active")
        st.caption("🟢 Safety Monitor Agent: Active")
        
        st.divider()
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()