"""
Unified Lab Assistant UI - Integrating all components
"""
import streamlit as st
import sys
import os
from datetime import datetime
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Add directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rohit_prototype'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'weavehacks_flow-1/src'))

# Import components
from claude_interface import ClaudeInterface
from gemini_interface import GeminiInterface
from speech_recognition_module import SpeechRecognizer
from weavehacks_flow.agents.safety_monitoring_agent import SafetyMonitoringAgent
from weavehacks_flow.utils.chemistry_calculations import (
    calculate_sulfur_amount,
    calculate_nabh4_amount,
    calculate_percent_yield
)

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
    if 'ai_model' not in st.session_state:
        st.session_state.ai_model = "Claude"
    if 'audio_method' not in st.session_state:
        st.session_state.audio_method = "System Microphone"
    if 'experiment_data' not in st.session_state:
        st.session_state.experiment_data = {
            'mass_gold': 0.0,
            'mass_toab': 0.0,
            'mass_sulfur': 0.0,
            'mass_nabh4': 0.0,
            'volume_toluene': 0.0,
            'volume_nanopure_rt': 0.0,
            'volume_nanopure_cold': 0.0,
            'mass_final': 0.0,
            'observations': []
        }
    if 'safety_agent' not in st.session_state:
        st.session_state.safety_agent = SafetyMonitoringAgent()
    if 'safety_monitoring_active' not in st.session_state:
        st.session_state.safety_monitoring_active = False
    
    # Test connections only once at startup
    if 'claude_status' not in st.session_state:
        st.session_state.claude_status = st.session_state.claude.test_connection()
    if 'gemini_status' not in st.session_state:
        st.session_state.gemini_status = st.session_state.gemini.test_connection()

def render_safety_panel():
    """Render the safety monitoring panel"""
    st.subheader("🚨 Safety Monitoring")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        monitoring_active = st.toggle(
            "Active Monitoring",
            value=st.session_state.safety_monitoring_active,
            help="Enable real-time safety parameter monitoring"
        )
        st.session_state.safety_monitoring_active = monitoring_active
    
    with col2:
        if st.button("Check Status", type="primary"):
            st.session_state.safety_agent.monitor_parameters(use_real_data=True)
    
    # Display current parameters
    if hasattr(st.session_state.safety_agent, 'current_temperature'):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            temp_color = "🟢" if 15 <= st.session_state.safety_agent.current_temperature <= 35 else "🔴"
            st.metric(
                f"{temp_color} Temperature",
                f"{st.session_state.safety_agent.current_temperature:.1f}°C",
                help="Safe range: 15-35°C"
            )
        
        with col2:
            press_color = "🟢" if 95 <= st.session_state.safety_agent.current_pressure <= 110 else "🔴"
            st.metric(
                f"{press_color} Pressure",
                f"{st.session_state.safety_agent.current_pressure:.1f} kPa",
                help="Safe range: 95-110 kPa"
            )
        
        with col3:
            n2_color = "🟢" if 75 <= st.session_state.safety_agent.current_nitrogen <= 85 else "🔴"
            st.metric(
                f"{n2_color} Nitrogen",
                f"{st.session_state.safety_agent.current_nitrogen:.1f}%",
                help="Safe range: 75-85%"
            )
        
        with col4:
            o2_color = "🟢" if 19 <= st.session_state.safety_agent.current_oxygen <= 23 else "🔴"
            st.metric(
                f"{o2_color} Oxygen",
                f"{st.session_state.safety_agent.current_oxygen:.1f}%",
                help="Safe range: 19-23%"
            )
        
        # Show safety status
        if st.session_state.safety_agent.is_safe():
            st.success("✅ All parameters within safe range")
        else:
            st.error("⚠️ Safety parameters exceeded! Check equipment immediately.")
            warnings = st.session_state.safety_agent.check_warning_levels()
            for warning in warnings:
                st.warning(warning)

def render_data_collection_panel():
    """Render the data collection panel"""
    st.subheader("📊 Data Collection")
    
    tab1, tab2, tab3 = st.tabs(["Reagent Masses", "Volumes", "Calculations"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.experiment_data['mass_gold'] = st.number_input(
                "HAuCl₄·3H₂O (g)",
                min_value=0.0,
                max_value=10.0,
                value=st.session_state.experiment_data['mass_gold'],
                step=0.0001,
                format="%.4f"
            )
            
            st.session_state.experiment_data['mass_toab'] = st.number_input(
                "TOAB (g)",
                min_value=0.0,
                max_value=10.0,
                value=st.session_state.experiment_data['mass_toab'],
                step=0.0001,
                format="%.4f"
            )
        
        with col2:
            st.session_state.experiment_data['mass_sulfur'] = st.number_input(
                "PhCH₂CH₂SH (g)",
                min_value=0.0,
                max_value=10.0,
                value=st.session_state.experiment_data['mass_sulfur'],
                step=0.0001,
                format="%.4f"
            )
            
            st.session_state.experiment_data['mass_nabh4'] = st.number_input(
                "NaBH₄ (g)",
                min_value=0.0,
                max_value=10.0,
                value=st.session_state.experiment_data['mass_nabh4'],
                step=0.0001,
                format="%.4f"
            )
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.experiment_data['volume_toluene'] = st.number_input(
                "Toluene (mL)",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.experiment_data['volume_toluene'],
                step=0.1,
                format="%.1f"
            )
            
            st.session_state.experiment_data['volume_nanopure_rt'] = st.number_input(
                "Nanopure Water RT (mL)",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.experiment_data['volume_nanopure_rt'],
                step=0.1,
                format="%.1f"
            )
        
        with col2:
            st.session_state.experiment_data['volume_nanopure_cold'] = st.number_input(
                "Nanopure Water Cold (mL)",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.experiment_data['volume_nanopure_cold'],
                step=0.1,
                format="%.1f"
            )
    
    with tab3:
        if st.session_state.experiment_data['mass_gold'] > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Sulfur Calculation (3 eq)**")
                sulfur_calc = calculate_sulfur_amount(st.session_state.experiment_data['mass_gold'])
                st.info(f"Required: {sulfur_calc['mass_sulfur_g']:.4f} g")
                st.caption(f"Moles Au: {sulfur_calc['moles_gold']:.6f}")
                st.caption(f"Moles S: {sulfur_calc['moles_sulfur']:.6f}")
            
            with col2:
                st.write("**NaBH₄ Calculation (10 eq)**")
                nabh4_calc = calculate_nabh4_amount(st.session_state.experiment_data['mass_gold'])
                st.info(f"Required: {nabh4_calc['mass_nabh4_g']:.4f} g")
                st.caption(f"Moles Au: {nabh4_calc['moles_gold']:.6f}")
                st.caption(f"Moles NaBH₄: {nabh4_calc['moles_nabh4']:.6f}")
            
            # Final yield section
            st.write("**Final Product**")
            st.session_state.experiment_data['mass_final'] = st.number_input(
                "Au₂₅ Nanoparticles (g)",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.experiment_data['mass_final'],
                step=0.0001,
                format="%.4f"
            )
            
            if st.session_state.experiment_data['mass_final'] > 0:
                yield_calc = calculate_percent_yield(
                    st.session_state.experiment_data['mass_gold'],
                    st.session_state.experiment_data['mass_final']
                )
                st.success(f"Percent Yield: {yield_calc['percent_yield']:.2f}%")
                st.caption(f"Theoretical Au: {yield_calc['gold_content_g']:.4f} g")

def render_protocol_panel():
    """Render the protocol steps panel"""
    st.subheader("📋 Protocol Steps")
    
    protocol_steps = [
        "1. Weigh HAuCl₄·3H₂O (0.1576g)",
        "2. Measure Nanopure water (5 mL)",
        "3. Dissolve gold compound in water",
        "4. Weigh TOAB (~0.25g)",
        "5. Measure toluene (10 mL)",
        "6. Dissolve TOAB in toluene",
        "7. Combine both in round-bottom flask",
        "8. Move to fume hood",
        "9. Stir vigorously (~1100 rpm) for ~15 min",
        "10. Remove aqueous layer",
        "11. Purge with N₂",
        "12. Cool to 0°C in ice bath",
        "13. Calculate and add PhCH₂CH₂SH (3 eq)",
        "14. Observe color change",
        "15. Calculate and add NaBH₄ (10 eq) in cold water",
        "16. Stir overnight under N₂",
        "17. Remove aqueous layer",
        "18. Add ethanol to precipitate",
        "19. Collect Au₂₅ clusters"
    ]
    
    # Track current step
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    
    # Display current step
    if st.session_state.current_step < len(protocol_steps):
        st.info(f"Current: {protocol_steps[st.session_state.current_step]}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Previous", disabled=st.session_state.current_step == 0):
                st.session_state.current_step -= 1
        with col2:
            st.progress((st.session_state.current_step + 1) / len(protocol_steps))
        with col3:
            if st.button("Next ➡️", disabled=st.session_state.current_step >= len(protocol_steps) - 1):
                st.session_state.current_step += 1
    
    # Show all steps
    with st.expander("View All Steps"):
        for i, step in enumerate(protocol_steps):
            if i == st.session_state.current_step:
                st.markdown(f"**→ {step}**")
            elif i < st.session_state.current_step:
                st.markdown(f"✓ ~~{step}~~")
            else:
                st.markdown(f"○ {step}")

def render_observations_panel():
    """Render the qualitative observations panel"""
    st.subheader("📝 Qualitative Observations")
    
    new_observation = st.text_area(
        "Add observation",
        placeholder="Describe what you observe (color changes, precipitates, etc.)",
        height=100
    )
    
    if st.button("Add Observation", type="primary"):
        if new_observation:
            observation = {
                'timestamp': datetime.now().isoformat(),
                'step': st.session_state.current_step if 'current_step' in st.session_state else 0,
                'text': new_observation
            }
            st.session_state.experiment_data['observations'].append(observation)
            st.success("Observation recorded!")
            st.rerun()
    
    # Display observations
    if st.session_state.experiment_data['observations']:
        st.write("**Recorded Observations:**")
        for obs in reversed(st.session_state.experiment_data['observations']):
            timestamp = datetime.fromisoformat(obs['timestamp'])
            st.write(f"**Step {obs['step'] + 1}** - {timestamp.strftime('%H:%M:%S')}")
            st.write(obs['text'])
            st.divider()

def main():
    st.set_page_config(
        page_title="WeaveHacks Lab Assistant",
        page_icon="🧪",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Header
    st.title("🧪 WeaveHacks Lab Assistant")
    st.caption("AI-powered assistance for wet lab nanoparticle synthesis")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎙️ AI Assistant",
        "🚨 Safety",
        "📊 Data Collection",
        "📋 Protocol",
        "📝 Observations"
    ])
    
    with tab1:
        # AI Assistant functionality
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("AI Laboratory Assistant")
        
        with col2:
            st.session_state.ai_model = st.selectbox(
                "AI Model",
                ["Claude", "Gemini"],
                index=0 if st.session_state.ai_model == "Claude" else 1
            )
        
        # Chat interface
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if "timestamp" in message:
                        st.caption(f"{message['timestamp']} via {message.get('model', 'Unknown')}")
        
        # Input methods
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.chat_input("Ask about your experiment...")
        
        with col2:
            audio_method = st.radio(
                "Audio",
                ["🎤 Mic", "🌐 Browser"],
                horizontal=True
            )
        
        if user_input:
            # Process text input
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            # Get AI response
            with st.spinner("Thinking..."):
                if st.session_state.ai_model == "Claude":
                    response = st.session_state.claude.query(user_input)
                else:
                    response = st.session_state.gemini.query(user_input)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "model": st.session_state.ai_model
                })
            
            st.rerun()
    
    with tab2:
        render_safety_panel()
    
    with tab3:
        render_data_collection_panel()
    
    with tab4:
        render_protocol_panel()
    
    with tab5:
        render_observations_panel()
    
    # Sidebar with experiment summary
    with st.sidebar:
        st.header("Experiment Summary")
        
        if st.session_state.experiment_data['mass_gold'] > 0:
            st.metric("Gold compound", f"{st.session_state.experiment_data['mass_gold']:.4f} g")
        if st.session_state.experiment_data['mass_final'] > 0:
            yield_calc = calculate_percent_yield(
                st.session_state.experiment_data['mass_gold'],
                st.session_state.experiment_data['mass_final']
            )
            st.metric("Yield", f"{yield_calc['percent_yield']:.1f}%")
        
        st.divider()
        
        # Export functionality
        if st.button("Export Data", type="primary"):
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'experiment_data': st.session_state.experiment_data,
                'observations': st.session_state.experiment_data['observations']
            }
            
            st.download_button(
                label="Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()