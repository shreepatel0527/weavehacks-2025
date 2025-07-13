"""
Real-time Lab Assistant UI with WebSocket support and live visualizations
"""
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import threading
import queue
import json
from pathlib import Path
import sys

# Add to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "weavehacks_flow-1/src"))

# Import our modules
from realtime_voice_processor import RealtimeVoiceProcessor, VoiceCommandProcessor
from realtime_safety_monitor import RealtimeSafetyMonitor, SafetyLevel
from weavehacks_flow.utils.chemistry_calculations import (
    calculate_sulfur_amount,
    calculate_nabh4_amount,
    calculate_percent_yield
)

# Configure Streamlit
st.set_page_config(
    page_title="WeaveHacks Real-time Lab Assistant",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better real-time display
st.markdown("""
<style>
    .stAlert {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-critical {
        background-color: #ff4b4b;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        animation: blink 0.5s infinite;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

class RealtimeLabUI:
    """Real-time UI for lab assistant"""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_components()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.voice_processor = None
            st.session_state.safety_monitor = None
            st.session_state.voice_enabled = False
            st.session_state.monitoring_enabled = False
            st.session_state.messages = []
            st.session_state.alerts = []
            st.session_state.experiment_data = {
                'mass_gold': 0.0,
                'mass_toab': 0.0,
                'mass_sulfur': 0.0,
                'mass_nabh4': 0.0,
                'volume_toluene': 0.0,
                'volume_nanopure_rt': 0.0,
                'volume_nanopure_cold': 0.0,
                'mass_final': 0.0,
                'observations': [],
                'current_step': 0
            }
            st.session_state.parameter_history = {
                'temperature': [],
                'pressure': [],
                'nitrogen': [],
                'oxygen': []
            }
    
    def setup_components(self):
        """Setup real-time components"""
        # Voice processor
        if st.session_state.voice_processor is None:
            command_processor = VoiceCommandProcessor()
            
            def voice_callback(text):
                # Add to messages
                st.session_state.messages.append({
                    'role': 'user',
                    'content': text,
                    'timestamp': datetime.now(),
                    'source': 'voice'
                })
                
                # Process command
                response = command_processor.process_command(text)
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now()
                })
            
            st.session_state.voice_processor = RealtimeVoiceProcessor(
                callback=voice_callback
            )
        
        # Safety monitor
        if st.session_state.safety_monitor is None:
            monitor = RealtimeSafetyMonitor()
            
            def alert_callback(event):
                st.session_state.alerts.append({
                    'timestamp': event.timestamp,
                    'level': event.level.value,
                    'parameter': event.parameter,
                    'value': event.value,
                    'message': event.message
                })
            
            monitor.register_alert_callback(alert_callback)
            st.session_state.safety_monitor = monitor
    
    def render_header(self):
        """Render header with real-time status"""
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.title("🧪 Real-time Lab Assistant")
        
        with col2:
            voice_status = "🟢 Active" if st.session_state.voice_enabled else "🔴 Inactive"
            st.metric("Voice", voice_status)
        
        with col3:
            monitor_status = "🟢 Active" if st.session_state.monitoring_enabled else "🔴 Inactive"
            st.metric("Monitoring", monitor_status)
        
        with col4:
            # Auto-refresh for real-time updates
            count = st_autorefresh(interval=1000, limit=None, key="refresh")
            st.metric("Updates", f"{count}")
    
    def render_safety_dashboard(self):
        """Render real-time safety dashboard"""
        st.header("🚨 Safety Monitoring Dashboard")
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Monitoring" if not st.session_state.monitoring_enabled else "Stop Monitoring"):
                if not st.session_state.monitoring_enabled:
                    st.session_state.safety_monitor.start()
                    st.session_state.monitoring_enabled = True
                else:
                    st.session_state.safety_monitor.stop()
                    st.session_state.monitoring_enabled = False
        
        # Get current status
        if st.session_state.monitoring_enabled:
            status = st.session_state.safety_monitor.get_current_status()
            
            # Display current values
            col1, col2, col3, col4 = st.columns(4)
            
            parameters = ['temperature', 'pressure', 'nitrogen', 'oxygen']
            units = {'temperature': '°C', 'pressure': 'kPa', 'nitrogen': '%', 'oxygen': '%'}
            icons = {'temperature': '🌡️', 'pressure': '🔵', 'nitrogen': '💨', 'oxygen': '⭕'}
            
            for i, param in enumerate(parameters):
                with [col1, col2, col3, col4][i]:
                    if param in status['current_values']:
                        value = status['current_values'][param]['value']
                        unit = units[param]
                        icon = icons[param]
                        
                        # Check if there's an active alert
                        if param in status['active_alerts']:
                            alert = status['active_alerts'][param]
                            if alert['level'] == 'critical':
                                st.markdown(f"""
                                <div class="alert-critical">
                                    <h3>{icon} {param.title()}</h3>
                                    <h1>{value:.1f} {unit}</h1>
                                    <p>⚠️ {alert['message']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.warning(f"{icon} {param.title()}: {value:.1f} {unit}")
                        else:
                            st.metric(f"{icon} {param.title()}", f"{value:.1f} {unit}")
            
            # Display live charts
            st.subheader("📊 Live Parameter Trends")
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Temperature', 'Pressure', 'Nitrogen', 'Oxygen'),
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # Add traces for each parameter
            for i, param in enumerate(parameters):
                row = i // 2 + 1
                col = i % 2 + 1
                
                # Get history
                history = st.session_state.safety_monitor.get_parameter_history(param, 5)
                
                if history:
                    df = pd.DataFrame(history)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=df['value'],
                            mode='lines+markers',
                            name=param.title(),
                            line=dict(width=2)
                        ),
                        row=row, col=col
                    )
                    
                    # Add safe range
                    config = st.session_state.safety_monitor.config[param]
                    fig.add_hline(
                        y=config['min_safe'], 
                        line_dash="dash", 
                        line_color="orange",
                        row=row, col=col
                    )
                    fig.add_hline(
                        y=config['max_safe'], 
                        line_dash="dash", 
                        line_color="orange",
                        row=row, col=col
                    )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Alert history
            if st.session_state.alerts:
                st.subheader("🚨 Recent Alerts")
                alert_df = pd.DataFrame(st.session_state.alerts[-10:])
                alert_df['timestamp'] = pd.to_datetime(alert_df['timestamp'])
                alert_df = alert_df.sort_values('timestamp', ascending=False)
                
                for _, alert in alert_df.iterrows():
                    if alert['level'] == 'critical':
                        st.error(f"{alert['timestamp'].strftime('%H:%M:%S')} - {alert['message']}")
                    else:
                        st.warning(f"{alert['timestamp'].strftime('%H:%M:%S')} - {alert['message']}")
    
    def render_voice_interface(self):
        """Render voice control interface"""
        st.header("🎤 Voice Control")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("Enable Voice" if not st.session_state.voice_enabled else "Disable Voice"):
                if not st.session_state.voice_enabled:
                    st.session_state.voice_processor.start()
                    st.session_state.voice_enabled = True
                    st.success("Voice control activated!")
                else:
                    st.session_state.voice_processor.stop()
                    st.session_state.voice_enabled = False
                    st.info("Voice control deactivated")
        
        with col2:
            if st.session_state.voice_enabled:
                st.info("🎤 Listening... Speak naturally and pause to trigger commands")
                
                # Show recent transcriptions
                recent = st.session_state.voice_processor.get_recent_transcriptions(3)
                if recent:
                    st.write("Recent commands:")
                    for trans in recent:
                        st.caption(f"• {trans['text']}")
    
    def render_data_panel(self):
        """Render data collection panel"""
        st.header("📊 Experiment Data")
        
        tab1, tab2, tab3 = st.tabs(["Measurements", "Calculations", "Observations"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Masses (g)")
                st.session_state.experiment_data['mass_gold'] = st.number_input(
                    "HAuCl₄·3H₂O", 
                    value=st.session_state.experiment_data['mass_gold'],
                    format="%.4f", step=0.0001
                )
                st.session_state.experiment_data['mass_toab'] = st.number_input(
                    "TOAB",
                    value=st.session_state.experiment_data['mass_toab'],
                    format="%.4f", step=0.0001
                )
                st.session_state.experiment_data['mass_sulfur'] = st.number_input(
                    "PhCH₂CH₂SH",
                    value=st.session_state.experiment_data['mass_sulfur'],
                    format="%.4f", step=0.0001
                )
                st.session_state.experiment_data['mass_nabh4'] = st.number_input(
                    "NaBH₄",
                    value=st.session_state.experiment_data['mass_nabh4'],
                    format="%.4f", step=0.0001
                )
            
            with col2:
                st.subheader("Volumes (mL)")
                st.session_state.experiment_data['volume_toluene'] = st.number_input(
                    "Toluene",
                    value=st.session_state.experiment_data['volume_toluene'],
                    format="%.1f", step=0.1
                )
                st.session_state.experiment_data['volume_nanopure_rt'] = st.number_input(
                    "Nanopure (RT)",
                    value=st.session_state.experiment_data['volume_nanopure_rt'],
                    format="%.1f", step=0.1
                )
                st.session_state.experiment_data['volume_nanopure_cold'] = st.number_input(
                    "Nanopure (Cold)",
                    value=st.session_state.experiment_data['volume_nanopure_cold'],
                    format="%.1f", step=0.1
                )
        
        with tab2:
            if st.session_state.experiment_data['mass_gold'] > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Reagent Calculations")
                    
                    # Sulfur calculation
                    sulfur_calc = calculate_sulfur_amount(st.session_state.experiment_data['mass_gold'])
                    st.info(f"Sulfur needed: {sulfur_calc['mass_sulfur_g']:.4f} g")
                    
                    # NaBH4 calculation
                    nabh4_calc = calculate_nabh4_amount(st.session_state.experiment_data['mass_gold'])
                    st.info(f"NaBH₄ needed: {nabh4_calc['mass_nabh4_g']:.4f} g")
                
                with col2:
                    st.subheader("Yield Calculation")
                    
                    st.session_state.experiment_data['mass_final'] = st.number_input(
                        "Final Au₂₅ mass (g)",
                        value=st.session_state.experiment_data['mass_final'],
                        format="%.4f", step=0.0001
                    )
                    
                    if st.session_state.experiment_data['mass_final'] > 0:
                        yield_calc = calculate_percent_yield(
                            st.session_state.experiment_data['mass_gold'],
                            st.session_state.experiment_data['mass_final']
                        )
                        st.success(f"Percent Yield: {yield_calc['percent_yield']:.2f}%")
        
        with tab3:
            st.subheader("Qualitative Observations")
            
            new_obs = st.text_area("Add observation", height=100)
            if st.button("Record Observation"):
                if new_obs:
                    st.session_state.experiment_data['observations'].append({
                        'timestamp': datetime.now(),
                        'text': new_obs,
                        'step': st.session_state.experiment_data['current_step']
                    })
                    st.success("Observation recorded!")
                    st.rerun()
            
            # Display observations
            for obs in reversed(st.session_state.experiment_data['observations'][-5:]):
                st.write(f"**Step {obs['step']}** - {obs['timestamp'].strftime('%H:%M:%S')}")
                st.write(obs['text'])
                st.divider()
    
    def render_chat_interface(self):
        """Render chat interface"""
        st.header("💬 Lab Assistant Chat")
        
        # Display messages
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages[-10:]:
                if msg['role'] == 'user':
                    if msg.get('source') == 'voice':
                        st.chat_message("user").write(f"🎤 {msg['content']}")
                    else:
                        st.chat_message("user").write(msg['content'])
                else:
                    st.chat_message("assistant").write(msg['content'])
        
        # Input
        user_input = st.chat_input("Ask about your experiment...")
        if user_input:
            st.session_state.messages.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now(),
                'source': 'text'
            })
            
            # Simple response for now
            response = f"I understand you're asking about: {user_input}. How can I help?"
            st.session_state.messages.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now()
            })
            st.rerun()
    
    def render_sidebar(self):
        """Render sidebar with summary"""
        with st.sidebar:
            st.header("📊 Experiment Summary")
            
            # Progress
            progress = st.session_state.experiment_data['current_step'] / 19
            st.progress(progress)
            st.caption(f"Step {st.session_state.experiment_data['current_step'] + 1} of 19")
            
            # Key metrics
            if st.session_state.experiment_data['mass_gold'] > 0:
                st.metric("Gold mass", f"{st.session_state.experiment_data['mass_gold']:.4f} g")
            
            if st.session_state.experiment_data['mass_final'] > 0:
                yield_calc = calculate_percent_yield(
                    st.session_state.experiment_data['mass_gold'],
                    st.session_state.experiment_data['mass_final']
                )
                st.metric("Yield", f"{yield_calc['percent_yield']:.1f}%")
            
            # Active alerts
            active_alerts = 0
            if st.session_state.monitoring_enabled:
                status = st.session_state.safety_monitor.get_current_status()
                active_alerts = len(status.get('active_alerts', {}))
            
            if active_alerts > 0:
                st.error(f"⚠️ {active_alerts} active safety alerts!")
            else:
                st.success("✅ All systems normal")
            
            st.divider()
            
            # Export
            if st.button("Export Data", type="primary"):
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'experiment_data': st.session_state.experiment_data,
                    'alerts': st.session_state.alerts
                }
                
                st.download_button(
                    "Download JSON",
                    json.dumps(export_data, indent=2, default=str),
                    f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
    
    def run(self):
        """Run the UI"""
        self.render_header()
        self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🚨 Safety", "🎤 Voice", "📊 Data", "💬 Chat"
        ])
        
        with tab1:
            self.render_safety_dashboard()
        
        with tab2:
            self.render_voice_interface()
        
        with tab3:
            self.render_data_panel()
        
        with tab4:
            self.render_chat_interface()

if __name__ == "__main__":
    ui = RealtimeLabUI()
    ui.run()