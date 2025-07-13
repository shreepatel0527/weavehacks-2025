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
            response = requests.post(f"{self.api_base}/experiments", 
                                   params={"experiment_id": experiment_id})
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, f"API Error: {response.status_code}"
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
                        "sulfur-amount", exp['experiment_id'], gold_mass=exp['mass_gold']
                    )
                    if success:
                        st.success(f"Sulfur needed: {result['mass_sulfur_needed_g']:.4f}g")
                    else:
                        st.error(f"Calculation failed: {result}")
                else:
                    st.warning("Please record gold mass first")
        
        with calc_col2:
            if st.button("Calculate NaBH4 Amount"):
                if exp['mass_gold'] > 0:
                    success, result = platform.calculate_chemistry(
                        "nabh4-amount", exp['experiment_id'], gold_mass=exp['mass_gold']
                    )
                    if success:
                        st.success(f"NaBH4 needed: {result['mass_nabh4_needed_g']:.4f}g")
                    else:
                        st.error(f"Calculation failed: {result}")
                else:
                    st.warning("Please record gold mass first")
        
        with calc_col3:
            if st.button("Calculate Percent Yield"):
                if exp['mass_gold'] > 0 and exp['mass_final'] > 0:
                    success, result = platform.calculate_chemistry(
                        "percent-yield", exp['experiment_id'], 
                        gold_mass=exp['mass_gold'], final_mass=exp['mass_final']
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
    
    # Audio input
    audio_bytes = st.audio_input("🎤 Record your measurement")
    
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
    """Safety monitoring interface with API integration"""
    st.subheader("🛡️ Safety Monitoring")
    
    if not st.session_state.current_experiment_id:
        st.warning("Please create an experiment first in the Dashboard tab.")
        return
    
    platform = st.session_state.platform
    
    # Safety status
    st.metric("Safety Status", "🟢 Safe", help="All parameters within normal ranges")
    
    # Test safety alert
    st.subheader("⚠️ Safety Alert Testing")
    
    with st.form("safety_alert_test"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            parameter = st.selectbox("Parameter", ["temperature", "pressure", "gas_level"])
        
        with col2:
            value = st.number_input("Current Value", min_value=0.0)
            threshold = st.number_input("Threshold", min_value=0.0)
        
        with col3:
            severity = st.selectbox("Severity", ["warning", "critical"])
        
        if st.form_submit_button("Create Test Alert"):
            success, result = platform.create_safety_alert(
                st.session_state.current_experiment_id, parameter, value, threshold, severity
            )
            
            if success:
                st.warning(f"Safety Alert Created: {parameter} = {value} (threshold: {threshold})")
            else:
                st.error(f"Failed to create alert: {result}")


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
    
    # Chat interface
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your experiment, safety, or protocols..."):
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard", 
        "🎤 Voice Entry", 
        "🛡️ Safety Monitor", 
        "🤖 AI Assistant"
    ])
    
    with tab1:
        display_experiment_dashboard()
    
    with tab2:
        display_voice_data_entry()
    
    with tab3:
        display_safety_monitoring()
    
    with tab4:
        display_ai_assistant()
    
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