"""
Fully integrated lab assistant system combining all real-time features
"""
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import asyncio
import threading
import time
from datetime import datetime
import json
from pathlib import Path
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "weavehacks_flow-1/src"))

# Import all our modules
from realtime_voice_processor import RealtimeVoiceProcessor, VoiceCommandProcessor
from realtime_safety_monitor import RealtimeSafetyMonitor, SafetyLevel
from enhanced_agent_system import (
    MessageBus, EnhancedAgent, DataCollectionAgent, 
    SafetyMonitoringAgent, CoordinatorAgent, Priority, Task
)
from protocol_automation import ProtocolAutomation, StepStatus
from weavehacks_flow.utils.chemistry_calculations import (
    calculate_sulfur_amount, calculate_nabh4_amount, calculate_percent_yield
)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Configure page
st.set_page_config(
    page_title="Integrated Lab Assistant",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .protocol-step {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .protocol-step-active {
        background-color: #e8f4f9;
        border-left-color: #ff7f0e;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
    .agent-card {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.25rem;
        border: 1px solid #dee2e6;
    }
    .agent-busy {
        border-color: #ffc107;
        background-color: #fff3cd;
    }
    .safety-critical {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class IntegratedLabSystem:
    """Fully integrated lab assistant system"""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_systems()
        
    def initialize_session_state(self):
        """Initialize all session state variables"""
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = True
            st.session_state.systems_active = {
                'voice': False,
                'safety': False,
                'agents': False,
                'protocol': False
            }
            
            # System components
            st.session_state.voice_processor = None
            st.session_state.safety_monitor = None
            st.session_state.message_bus = None
            st.session_state.agents = {}
            st.session_state.protocol = None
            
            # Data storage
            st.session_state.experiment_data = {
                'measurements': {},
                'observations': [],
                'calculations': {},
                'alerts': []
            }
            
            # UI state
            st.session_state.selected_tab = 0
            st.session_state.messages = []
            st.session_state.agent_statuses = {}
    
    def setup_systems(self):
        """Setup all system components"""
        # Voice processor
        if st.session_state.voice_processor is None:
            command_processor = VoiceCommandProcessor()
            
            def voice_callback(text):
                st.session_state.messages.append({
                    'type': 'voice',
                    'content': text,
                    'timestamp': datetime.now()
                })
                
                # Process through agent system if active
                if st.session_state.systems_active['agents']:
                    self.process_voice_command(text)
            
            st.session_state.voice_processor = RealtimeVoiceProcessor(callback=voice_callback)
        
        # Safety monitor
        if st.session_state.safety_monitor is None:
            monitor = RealtimeSafetyMonitor()
            
            def safety_callback(event):
                st.session_state.experiment_data['alerts'].append({
                    'timestamp': event.timestamp,
                    'level': event.level.value,
                    'message': event.message,
                    'parameter': event.parameter,
                    'value': event.value
                })
                
                # Send to agent system if active
                if st.session_state.systems_active['agents'] and 'safety_agent' in st.session_state.agents:
                    self.handle_safety_alert(event)
            
            monitor.register_alert_callback(safety_callback)
            st.session_state.safety_monitor = monitor
        
        # Agent system
        if st.session_state.message_bus is None:
            self.setup_agent_system()
        
        # Protocol automation
        if st.session_state.protocol is None:
            protocol = ProtocolAutomation()
            
            # Register callbacks
            protocol.register_callback('step_started', self.on_protocol_step_started)
            protocol.register_callback('step_ready', self.on_protocol_step_ready)
            protocol.register_callback('step_completed', self.on_protocol_step_completed)
            
            st.session_state.protocol = protocol
    
    def setup_agent_system(self):
        """Setup the multi-agent system"""
        # Create message bus
        bus = MessageBus()
        st.session_state.message_bus = bus
        
        # Create agents
        data_agent = DataCollectionAgent("data_agent_1", bus)
        safety_agent = SafetyMonitoringAgent("safety_agent_1", bus)
        coordinator = CoordinatorAgent("coordinator", bus)
        
        # Store agents
        st.session_state.agents = {
            'data_agent': data_agent,
            'safety_agent': safety_agent,
            'coordinator': coordinator
        }
        
        # Register agents with coordinator
        coordinator.register_agent(data_agent)
        coordinator.register_agent(safety_agent)
    
    def process_voice_command(self, text):
        """Process voice command through agent system"""
        # Simple command parsing
        text_lower = text.lower()
        
        if "record" in text_lower and "mass" in text_lower:
            # Extract substance and value
            task = Task(
                id=f"voice_task_{int(time.time())}",
                name="record_mass",
                agent_id="data_agent_1",
                priority=Priority.HIGH,
                payload={'text': text}
            )
            st.session_state.agents['data_agent'].add_task(task)
        
        elif "safety" in text_lower or "check" in text_lower:
            task = Task(
                id=f"voice_safety_{int(time.time())}",
                name="check_safety",
                agent_id="safety_agent_1",
                priority=Priority.HIGH,
                payload={}
            )
            st.session_state.agents['safety_agent'].add_task(task)
    
    def handle_safety_alert(self, event):
        """Handle safety alert through agent system"""
        task = Task(
            id=f"safety_alert_{int(time.time())}",
            name="handle_alert",
            agent_id="safety_agent_1",
            priority=Priority.CRITICAL,
            payload={
                'parameter': event.parameter,
                'value': event.value,
                'level': event.level.value
            }
        )
        st.session_state.agents['safety_agent'].add_task(task)
    
    def on_protocol_step_started(self, step):
        """Handle protocol step started"""
        st.session_state.messages.append({
            'type': 'protocol',
            'content': f"Started: {step.name}",
            'timestamp': datetime.now()
        })
    
    def on_protocol_step_ready(self, step):
        """Handle protocol step ready for input"""
        st.session_state.messages.append({
            'type': 'protocol',
            'content': f"Ready for input: {step.name}",
            'timestamp': datetime.now()
        })
        
        # Create data collection task if needed
        if step.data_to_record and st.session_state.systems_active['agents']:
            for data_type in step.data_to_record:
                task = Task(
                    id=f"protocol_data_{step.id}_{data_type}",
                    name="record_data",
                    agent_id="data_agent_1",
                    priority=Priority.HIGH,
                    payload={
                        'step_id': step.id,
                        'data_type': data_type
                    }
                )
                st.session_state.agents['data_agent'].add_task(task)
    
    def on_protocol_step_completed(self, step):
        """Handle protocol step completed"""
        st.session_state.messages.append({
            'type': 'protocol',
            'content': f"Completed: {step.name}",
            'timestamp': datetime.now()
        })
    
    def render_header(self):
        """Render system header"""
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown('<p class="big-font">🧪 Integrated Lab Assistant</p>', unsafe_allow_html=True)
        
        with col2:
            active_systems = sum(st.session_state.systems_active.values())
            st.metric("Active Systems", f"{active_systems}/4")
        
        with col3:
            # Auto-refresh
            count = st_autorefresh(interval=1000, limit=None, key="main_refresh")
    
    def render_system_controls(self):
        """Render system control panel"""
        st.subheader("⚙️ System Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🎤 Voice" if not st.session_state.systems_active['voice'] else "🔴 Voice"):
                if not st.session_state.systems_active['voice']:
                    st.session_state.voice_processor.start()
                    st.session_state.systems_active['voice'] = True
                else:
                    st.session_state.voice_processor.stop()
                    st.session_state.systems_active['voice'] = False
        
        with col2:
            if st.button("🚨 Safety" if not st.session_state.systems_active['safety'] else "🔴 Safety"):
                if not st.session_state.systems_active['safety']:
                    st.session_state.safety_monitor.start()
                    st.session_state.systems_active['safety'] = True
                else:
                    st.session_state.safety_monitor.stop()
                    st.session_state.systems_active['safety'] = False
        
        with col3:
            if st.button("🤖 Agents" if not st.session_state.systems_active['agents'] else "🔴 Agents"):
                if not st.session_state.systems_active['agents']:
                    for agent in st.session_state.agents.values():
                        agent.start()
                    st.session_state.systems_active['agents'] = True
                else:
                    for agent in st.session_state.agents.values():
                        agent.stop()
                    st.session_state.systems_active['agents'] = False
        
        with col4:
            if st.button("📋 Protocol" if not st.session_state.systems_active['protocol'] else "⏸️ Protocol"):
                if not st.session_state.systems_active['protocol']:
                    st.session_state.protocol.start_protocol()
                    st.session_state.systems_active['protocol'] = True
                else:
                    st.session_state.protocol.pause_protocol()
                    st.session_state.systems_active['protocol'] = False
    
    def render_main_dashboard(self):
        """Render main dashboard with all systems"""
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "🚨 Safety", "🤖 Agents", "📋 Protocol", "💬 Messages"
        ])
        
        with tab1:
            self.render_overview()
        
        with tab2:
            self.render_safety_dashboard()
        
        with tab3:
            self.render_agent_dashboard()
        
        with tab4:
            self.render_protocol_dashboard()
        
        with tab5:
            self.render_message_log()
    
    def render_overview(self):
        """Render system overview"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔬 Experiment Progress")
            
            if st.session_state.systems_active['protocol']:
                status = st.session_state.protocol.get_protocol_status()
                
                # Progress bar
                st.progress(status['progress_percentage'] / 100)
                
                # Current step
                if status['current_step']:
                    st.markdown(f"""
                    <div class="protocol-step-active">
                        <h4>{status['current_step']['name']}</h4>
                        <p>Type: {status['current_step']['type']}</p>
                        <p>Status: {status['current_step']['status']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if 'time_remaining' in status['current_step']:
                        st.metric("Time Remaining", f"{status['current_step']['time_remaining']:.0f}s")
            else:
                st.info("Protocol not started. Click 'Protocol' button to begin.")
        
        with col2:
            st.subheader("📈 Key Metrics")
            
            # Display key measurements
            if st.session_state.experiment_data['measurements']:
                for key, value in list(st.session_state.experiment_data['measurements'].items())[:5]:
                    st.metric(key.replace('_', ' ').title(), f"{value['value']:.4f}")
    
    def render_safety_dashboard(self):
        """Render safety monitoring dashboard"""
        if not st.session_state.systems_active['safety']:
            st.warning("Safety monitoring is not active. Click 'Safety' button to start.")
            return
        
        # Get current status
        status = st.session_state.safety_monitor.get_current_status()
        
        # Safety alerts
        if status['active_alerts']:
            for param, alert in status['active_alerts'].items():
                if alert['level'] == 'critical':
                    st.markdown(f"""
                    <div class="safety-critical">
                        ⚠️ CRITICAL: {alert['message']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(alert['message'])
        
        # Parameter display
        col1, col2, col3, col4 = st.columns(4)
        params = ['temperature', 'pressure', 'nitrogen', 'oxygen']
        
        for i, param in enumerate(params):
            with [col1, col2, col3, col4][i]:
                if param in status['current_values']:
                    val = status['current_values'][param]
                    st.metric(
                        param.title(),
                        f"{val['value']:.1f} {val['units']}",
                        delta=None  # Could add trend
                    )
        
        # Live charts
        st.subheader("📊 Parameter Trends")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=params,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for i, param in enumerate(params):
            row = i // 2 + 1
            col = i % 2 + 1
            
            # Get history
            history = st.session_state.safety_monitor.get_parameter_history(param, 10)
            
            if history:
                df = pd.DataFrame(history)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['value'],
                        mode='lines+markers',
                        name=param,
                        line=dict(width=2),
                        marker=dict(size=6)
                    ),
                    row=row, col=col
                )
                
                # Add safe range
                config = st.session_state.safety_monitor.config[param]
                latest_time = df['timestamp'].max()
                earliest_time = df['timestamp'].min()
                
                fig.add_trace(
                    go.Scatter(
                        x=[earliest_time, latest_time],
                        y=[config['min_safe'], config['min_safe']],
                        mode='lines',
                        line=dict(color='orange', dash='dash'),
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[earliest_time, latest_time],
                        y=[config['max_safe'], config['max_safe']],
                        mode='lines',
                        line=dict(color='orange', dash='dash'),
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_agent_dashboard(self):
        """Render agent system dashboard"""
        if not st.session_state.systems_active['agents']:
            st.warning("Agent system is not active. Click 'Agents' button to start.")
            return
        
        st.subheader("🤖 Agent Status")
        
        # Agent cards
        cols = st.columns(3)
        
        for i, (agent_id, agent) in enumerate(st.session_state.agents.items()):
            with cols[i % 3]:
                status_class = "agent-busy" if agent.status.value == "busy" else "agent-card"
                
                st.markdown(f"""
                <div class="{status_class}">
                    <h4>{agent_id.replace('_', ' ').title()}</h4>
                    <p>Status: {agent.status.value}</p>
                    <p>Queue: {agent.task_queue.qsize()} tasks</p>
                    {f'<p>Current: {agent.current_task.name}</p>' if agent.current_task else ''}
                </div>
                """, unsafe_allow_html=True)
        
        # Message bus activity
        st.subheader("📨 Recent Messages")
        
        if st.session_state.message_bus.message_history:
            recent_messages = st.session_state.message_bus.message_history[-10:]
            
            for msg in reversed(recent_messages):
                st.caption(
                    f"{msg.timestamp.strftime('%H:%M:%S')} | "
                    f"{msg.sender} → {msg.recipient} | "
                    f"Priority: {msg.priority.name}"
                )
                if isinstance(msg.content, dict):
                    st.json(msg.content)
                else:
                    st.write(msg.content)
    
    def render_protocol_dashboard(self):
        """Render protocol automation dashboard"""
        if not st.session_state.protocol:
            st.warning("Protocol not loaded.")
            return
        
        status = st.session_state.protocol.get_protocol_status()
        
        # Protocol controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Progress", f"{status['completed_steps']}/{status['total_steps']}")
        
        with col2:
            if status['elapsed_time']:
                elapsed = int(status['elapsed_time'])
                st.metric("Elapsed Time", f"{elapsed // 60}:{elapsed % 60:02d}")
        
        with col3:
            if st.button("Export Protocol Data"):
                data = st.session_state.protocol.export_protocol_data()
                st.download_button(
                    "Download JSON",
                    json.dumps(data, indent=2, default=str),
                    f"protocol_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
        
        # Step list
        st.subheader("Protocol Steps")
        
        for i, step in enumerate(st.session_state.protocol.protocol_steps):
            if i == st.session_state.protocol.current_step_index:
                # Current step - highlighted
                st.markdown(f"""
                <div class="protocol-step-active">
                    <h4>{i+1}. {step.name}</h4>
                    <p>{step.description}</p>
                    <p>Status: {step.status.value}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Data recording interface for current step
                if step.status == StepStatus.READY and step.data_to_record:
                    st.write("Record data:")
                    for data_type in step.data_to_record:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            value = st.number_input(
                                data_type.replace('_', ' ').title(),
                                key=f"{step.id}_{data_type}",
                                format="%.4f"
                            )
                        with col2:
                            if st.button("Record", key=f"btn_{step.id}_{data_type}"):
                                st.session_state.protocol.record_data(step.id, data_type, value)
                                st.session_state.experiment_data['measurements'][data_type] = {
                                    'value': value,
                                    'timestamp': datetime.now().isoformat()
                                }
                                st.rerun()
            
            elif step.status == StepStatus.COMPLETED:
                st.success(f"✅ {i+1}. {step.name}")
            
            elif step.status == StepStatus.SKIPPED:
                st.warning(f"⏭️ {i+1}. {step.name} (Skipped)")
            
            else:
                st.caption(f"○ {i+1}. {step.name}")
    
    def render_message_log(self):
        """Render message and event log"""
        st.subheader("📜 System Messages")
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_voice = st.checkbox("Voice", value=True)
        with col2:
            show_protocol = st.checkbox("Protocol", value=True)
        with col3:
            show_system = st.checkbox("System", value=True)
        
        # Display messages
        messages_to_show = []
        
        for msg in st.session_state.messages[-50:]:
            if (msg['type'] == 'voice' and show_voice) or \
               (msg['type'] == 'protocol' and show_protocol) or \
               (msg['type'] == 'system' and show_system):
                messages_to_show.append(msg)
        
        for msg in reversed(messages_to_show):
            icon = {'voice': '🎤', 'protocol': '📋', 'system': '⚙️'}.get(msg['type'], '💬')
            st.write(f"{icon} **{msg['timestamp'].strftime('%H:%M:%S')}** - {msg['content']}")
    
    def render_sidebar(self):
        """Render sidebar with summary and controls"""
        with st.sidebar:
            st.header("📊 Experiment Summary")
            
            # System status
            st.subheader("System Status")
            for system, active in st.session_state.systems_active.items():
                st.write(f"{'🟢' if active else '🔴'} {system.title()}")
            
            st.divider()
            
            # Key metrics
            if st.session_state.experiment_data['measurements']:
                st.subheader("Key Measurements")
                
                # Calculate yield if possible
                if 'mass_gold' in st.session_state.experiment_data['measurements'] and \
                   'mass_final' in st.session_state.experiment_data['measurements']:
                    gold = st.session_state.experiment_data['measurements']['mass_gold']['value']
                    final = st.session_state.experiment_data['measurements']['mass_final']['value']
                    if gold > 0:
                        yield_calc = calculate_percent_yield(gold, final)
                        st.metric("Yield", f"{yield_calc['percent_yield']:.1f}%")
                
                # Show recent measurements
                for key, data in list(st.session_state.experiment_data['measurements'].items())[-5:]:
                    st.metric(key.replace('_', ' ').title(), f"{data['value']:.4f}")
            
            st.divider()
            
            # Safety summary
            if st.session_state.experiment_data['alerts']:
                st.subheader("⚠️ Recent Alerts")
                recent_alerts = st.session_state.experiment_data['alerts'][-3:]
                for alert in recent_alerts:
                    st.warning(f"{alert['parameter']}: {alert['level']}")
            
            st.divider()
            
            # Export all data
            if st.button("Export All Data", type="primary", key="export_all"):
                all_data = {
                    'timestamp': datetime.now().isoformat(),
                    'experiment_data': st.session_state.experiment_data,
                    'protocol_data': st.session_state.protocol.export_protocol_data() if st.session_state.protocol else None,
                    'system_status': {
                        'active_systems': st.session_state.systems_active,
                        'agent_statuses': {
                            agent_id: agent.status.value 
                            for agent_id, agent in st.session_state.agents.items()
                        } if st.session_state.agents else {}
                    }
                }
                
                st.download_button(
                    "Download Complete Dataset",
                    json.dumps(all_data, indent=2, default=str),
                    f"lab_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json",
                    key="download_all"
                )
    
    def run(self):
        """Run the integrated system"""
        self.render_header()
        self.render_sidebar()
        
        # System controls
        with st.container():
            self.render_system_controls()
            st.divider()
        
        # Main dashboard
        self.render_main_dashboard()

if __name__ == "__main__":
    # Create and run the integrated system
    system = IntegratedLabSystem()
    system.run()