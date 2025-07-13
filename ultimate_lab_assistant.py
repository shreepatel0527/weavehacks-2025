"""
Ultimate Lab Assistant: Complete integrated system with all advanced features
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
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
import base64
from io import BytesIO
from PIL import Image

# Import all our modules
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "weavehacks_flow-1/src"))

# Core systems from previous iterations
from realtime_voice_processor import RealtimeVoiceProcessor, VoiceCommandProcessor
from realtime_safety_monitor import RealtimeSafetyMonitor, SafetyLevel
from enhanced_agent_system import (
    MessageBus, EnhancedAgent, DataCollectionAgent, 
    SafetyMonitoringAgent, CoordinatorAgent, Priority, Task
)
from protocol_automation import ProtocolAutomation, StepStatus

# Advanced systems from iteration 3
from video_monitoring_system import VideoMonitoringSystem, VideoExperimentMonitor
from claude_flow_integration import ClaudeFlowReasoner, ClaudeFlowOrchestrator, ReasoningContext
from predictive_models import (
    YieldPredictor, ReactionOptimizer, AnomalyDetector, 
    ExperimentForecaster, ExperimentFeatures
)

# Chemistry calculations
from weavehacks_flow.utils.chemistry_calculations import (
    calculate_sulfur_amount, calculate_nabh4_amount, calculate_percent_yield
)

# Configure page
st.set_page_config(
    page_title="Ultimate Lab Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional UI
st.markdown("""
<style>
    /* Main theme */
    :root {
        --primary-color: #1f77b4;
        --success-color: #2ca02c;
        --warning-color: #ff7f0e;
        --danger-color: #d62728;
        --bg-color: #f8f9fa;
        --card-bg: #ffffff;
    }
    
    /* Cards and containers */
    .system-card {
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid var(--primary-color);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .alert-card {
        background: var(--danger-color);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        animation: pulse 2s infinite;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    .status-active {
        background: var(--success-color);
        animation: glow 2s infinite;
    }
    
    .status-inactive {
        background: #ccc;
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 5px var(--success-color); }
        50% { box-shadow: 0 0 20px var(--success-color); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    /* Video feed styling */
    .video-container {
        position: relative;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .video-overlay {
        position: absolute;
        top: 10px;
        left: 10px;
        background: rgba(0,0,0,0.7);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
    }
    
    /* AI reasoning display */
    .reasoning-result {
        background: #f0f8ff;
        border: 1px solid #4682b4;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .confidence-bar {
        height: 20px;
        background: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff7f0e 0%, #2ca02c 100%);
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)

class UltimateLabAssistant:
    """Complete integrated lab assistant system"""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_all_systems()
    
    def initialize_session_state(self):
        """Initialize comprehensive session state"""
        if 'ultimate_initialized' not in st.session_state:
            st.session_state.ultimate_initialized = True
            
            # System states
            st.session_state.systems = {
                'voice': False,
                'safety': False,
                'agents': False,
                'protocol': False,
                'video': False,
                'ai_reasoning': False,
                'ml_prediction': False
            }
            
            # System components
            st.session_state.voice_processor = None
            st.session_state.safety_monitor = None
            st.session_state.message_bus = None
            st.session_state.agents = {}
            st.session_state.protocol = None
            st.session_state.video_monitor = None
            st.session_state.claude_reasoner = None
            st.session_state.ml_predictor = None
            
            # Data storage
            st.session_state.experiment_data = {
                'measurements': {},
                'observations': [],
                'calculations': {},
                'alerts': [],
                'video_events': [],
                'ai_insights': [],
                'ml_predictions': []
            }
            
            # UI state
            st.session_state.current_view = 'dashboard'
            st.session_state.video_frame = None
            st.session_state.last_reasoning = None
    
    def setup_all_systems(self):
        """Setup all integrated systems"""
        
        # Voice system
        if st.session_state.voice_processor is None:
            self.setup_voice_system()
        
        # Safety monitoring
        if st.session_state.safety_monitor is None:
            self.setup_safety_system()
        
        # Agent system
        if st.session_state.message_bus is None:
            self.setup_agent_system()
        
        # Protocol automation
        if st.session_state.protocol is None:
            self.setup_protocol_system()
        
        # Video monitoring
        if st.session_state.video_monitor is None:
            self.setup_video_system()
        
        # AI reasoning
        if st.session_state.claude_reasoner is None:
            self.setup_ai_reasoning()
        
        # ML prediction
        if st.session_state.ml_predictor is None:
            self.setup_ml_prediction()
    
    def setup_voice_system(self):
        """Setup voice processing with integration"""
        command_processor = VoiceCommandProcessor()
        
        def voice_callback(text):
            # Log voice input
            st.session_state.experiment_data['observations'].append({
                'type': 'voice_command',
                'text': text,
                'timestamp': datetime.now()
            })
            
            # Process through systems
            asyncio.run(self.process_voice_input(text))
        
        st.session_state.voice_processor = RealtimeVoiceProcessor(callback=voice_callback)
    
    def setup_safety_system(self):
        """Setup safety monitoring with advanced alerts"""
        monitor = RealtimeSafetyMonitor()
        
        def safety_callback(event):
            # Store alert
            alert_data = {
                'timestamp': event.timestamp,
                'level': event.level.value,
                'parameter': event.parameter,
                'value': event.value,
                'message': event.message
            }
            st.session_state.experiment_data['alerts'].append(alert_data)
            
            # Trigger AI reasoning for critical alerts
            if event.level in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY]:
                asyncio.run(self.handle_critical_safety_event(alert_data))
        
        monitor.register_alert_callback(safety_callback)
        st.session_state.safety_monitor = monitor
    
    def setup_agent_system(self):
        """Setup enhanced agent system"""
        bus = MessageBus()
        st.session_state.message_bus = bus
        
        # Create specialized agents
        data_agent = DataCollectionAgent("data_agent_1", bus)
        safety_agent = SafetyMonitoringAgent("safety_agent_1", bus)
        coordinator = CoordinatorAgent("coordinator", bus)
        
        st.session_state.agents = {
            'data': data_agent,
            'safety': safety_agent,
            'coordinator': coordinator
        }
        
        # Register agents
        coordinator.register_agent(data_agent)
        coordinator.register_agent(safety_agent)
    
    def setup_protocol_system(self):
        """Setup protocol with ML integration"""
        protocol = ProtocolAutomation()
        
        # Enhanced callbacks
        protocol.register_callback('step_completed', self.on_protocol_step_completed)
        protocol.register_callback('step_ready', self.on_protocol_step_ready)
        
        st.session_state.protocol = protocol
    
    def setup_video_system(self):
        """Setup video monitoring"""
        video_monitor = VideoExperimentMonitor()
        
        def video_event_callback(event):
            st.session_state.experiment_data['video_events'].append({
                'timestamp': event.timestamp,
                'type': event.event_type.value,
                'description': event.description,
                'confidence': event.confidence
            })
            
            # Trigger AI analysis for significant events
            if event.confidence > 0.8:
                asyncio.run(self.analyze_video_event(event))
        
        video_monitor.video_system.register_event_callback(video_event_callback)
        st.session_state.video_monitor = video_monitor
    
    def setup_ai_reasoning(self):
        """Setup Claude-flow reasoning"""
        st.session_state.claude_reasoner = ClaudeFlowReasoner()
        st.session_state.claude_orchestrator = ClaudeFlowOrchestrator(
            st.session_state.claude_reasoner
        )
    
    def setup_ml_prediction(self):
        """Setup ML prediction models"""
        st.session_state.ml_predictor = YieldPredictor()
        st.session_state.reaction_optimizer = ReactionOptimizer(st.session_state.ml_predictor)
        st.session_state.anomaly_detector = AnomalyDetector()
        st.session_state.experiment_forecaster = ExperimentForecaster()
    
    # Integration methods
    
    async def process_voice_input(self, text):
        """Process voice input through all systems"""
        # Create reasoning context
        context = self.create_reasoning_context()
        context.user_query = text
        
        # Get AI reasoning
        reasoning_result = await st.session_state.claude_reasoner.reason(context)
        
        # Store insight
        st.session_state.experiment_data['ai_insights'].append({
            'timestamp': datetime.now(),
            'query': text,
            'conclusion': reasoning_result.conclusion,
            'confidence': reasoning_result.confidence,
            'recommendations': reasoning_result.recommendations
        })
        
        # Execute recommended actions
        for action in reasoning_result.next_actions[:3]:
            self.execute_action(action)
    
    async def handle_critical_safety_event(self, alert):
        """Handle critical safety events with AI"""
        context = self.create_reasoning_context()
        
        # Safety-critical reasoning
        reasoning_result = await st.session_state.claude_reasoner.reason(
            context, 
            reasoning_type=ReasoningType.SAFETY_CRITICAL
        )
        
        # Execute emergency actions
        for action in reasoning_result.next_actions:
            if action.get('priority') == 'emergency':
                self.execute_emergency_action(action)
    
    async def analyze_video_event(self, event):
        """Analyze video events with AI"""
        context = self.create_reasoning_context()
        context.recent_events.append({
            'type': 'video_event',
            'description': event.description,
            'confidence': event.confidence
        })
        
        # Diagnostic reasoning
        reasoning_result = await st.session_state.claude_reasoner.reason(
            context,
            reasoning_type=ReasoningType.DIAGNOSTIC
        )
        
        # Store insights
        if reasoning_result.confidence > 0.7:
            st.session_state.experiment_data['ai_insights'].append({
                'timestamp': datetime.now(),
                'source': 'video_analysis',
                'conclusion': reasoning_result.conclusion,
                'evidence': reasoning_result.supporting_evidence
            })
    
    def on_protocol_step_completed(self, step):
        """Handle protocol step completion with ML"""
        # Update ML features
        features = self.extract_current_features()
        
        # Predict yield
        if st.session_state.ml_predictor and features:
            prediction = st.session_state.ml_predictor.predict(features)
            
            st.session_state.experiment_data['ml_predictions'].append({
                'timestamp': datetime.now(),
                'step': step.name,
                'predicted_yield': prediction.prediction,
                'confidence_interval': prediction.confidence_interval,
                'feature_importance': prediction.feature_importance
            })
            
            # Check for anomalies
            current_data = {
                'temperature': features.temperature,
                'pressure': features.pressure,
                'ph': features.ph
            }
            
            anomalies = st.session_state.anomaly_detector.detect_anomalies(current_data)
            if anomalies:
                asyncio.run(self.handle_anomalies(anomalies))
    
    def on_protocol_step_ready(self, step):
        """Prepare for protocol step with optimization"""
        if step.step_type == StepType.CALCULATION:
            # Use ML to optimize parameters
            features = self.extract_current_features()
            if features:
                optimization = st.session_state.reaction_optimizer.optimize(
                    features, {}, n_iterations=10
                )
                
                # Display recommendations
                for rec in optimization['recommendations']:
                    st.info(f"💡 Optimization: {rec}")
    
    # Helper methods
    
    def create_reasoning_context(self) -> ReasoningContext:
        """Create current reasoning context"""
        return ReasoningContext(
            experiment_state=self.get_experiment_state(),
            historical_data=st.session_state.experiment_data.get('measurements', {}),
            sensor_readings=self.get_current_sensors(),
            recent_events=self.get_recent_events(),
            safety_status=self.get_safety_status(),
            protocol_step=self.get_current_protocol_step()
        )
    
    def get_experiment_state(self) -> Dict[str, Any]:
        """Get current experiment state"""
        state = {
            'active_systems': sum(st.session_state.systems.values()),
            'measurements_count': len(st.session_state.experiment_data['measurements']),
            'alerts_count': len(st.session_state.experiment_data['alerts']),
            'phase': 'unknown'
        }
        
        if st.session_state.protocol:
            status = st.session_state.protocol.get_protocol_status()
            state.update({
                'phase': 'synthesis',
                'progress': status['progress_percentage'],
                'current_step': status.get('current_step', {}).get('name', 'unknown')
            })
        
        return state
    
    def get_current_sensors(self) -> Dict[str, float]:
        """Get current sensor readings"""
        sensors = {}
        
        if st.session_state.safety_monitor and st.session_state.systems['safety']:
            status = st.session_state.safety_monitor.get_current_status()
            for param, data in status.get('current_values', {}).items():
                sensors[param] = data['value']
        
        return sensors
    
    def get_recent_events(self) -> List[Dict[str, Any]]:
        """Get recent events from all sources"""
        events = []
        
        # Add recent alerts
        events.extend(st.session_state.experiment_data['alerts'][-5:])
        
        # Add recent video events
        events.extend(st.session_state.experiment_data['video_events'][-5:])
        
        # Sort by timestamp
        events.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return events[:10]
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status"""
        if st.session_state.safety_monitor and st.session_state.systems['safety']:
            status = st.session_state.safety_monitor.get_current_status()
            return {
                'is_safe': len(status.get('active_alerts', {})) == 0,
                'active_alerts': status.get('active_alerts', {})
            }
        
        return {'is_safe': True, 'active_alerts': {}}
    
    def get_current_protocol_step(self) -> Optional[Dict[str, Any]]:
        """Get current protocol step"""
        if st.session_state.protocol:
            step = st.session_state.protocol.get_current_step()
            if step:
                return {
                    'name': step.name,
                    'type': step.step_type.value,
                    'status': step.status.value
                }
        return None
    
    def extract_current_features(self) -> Optional[ExperimentFeatures]:
        """Extract current features for ML"""
        measurements = st.session_state.experiment_data['measurements']
        sensors = self.get_current_sensors()
        
        # Check if we have minimum required data
        if not measurements or not sensors:
            return None
        
        return ExperimentFeatures(
            gold_mass=measurements.get('mass_gold', {}).get('value', 0.1576),
            toab_mass=measurements.get('mass_toab', {}).get('value', 0.25),
            sulfur_mass=measurements.get('mass_sulfur', {}).get('value', 0.052),
            nabh4_mass=measurements.get('mass_nabh4', {}).get('value', 0.015),
            temperature=sensors.get('temperature', 23.0),
            stirring_rpm=1100,  # Default
            reaction_time=3600,  # Default
            ph=7.0,  # Default
            ambient_temp=22.0,
            humidity=50.0,
            pressure=sensors.get('pressure', 101.3)
        )
    
    def execute_action(self, action: Dict[str, Any]):
        """Execute an action from AI reasoning"""
        action_type = action.get('type', '')
        
        if action_type == 'adjust_parameter':
            # Send to control agent
            task = Task(
                id=f"ai_action_{int(time.time())}",
                name="adjust_parameter",
                agent_id="control_agent_1",
                priority=Priority.HIGH,
                payload=action.get('payload', {})
            )
            # Would send to agent if control agent existed
        
        elif action_type == 'alert_user':
            st.warning(f"AI Alert: {action.get('message', 'Check system')}")
        
        elif action_type == 'record_observation':
            st.session_state.experiment_data['observations'].append({
                'type': 'ai_observation',
                'text': action.get('observation', ''),
                'timestamp': datetime.now()
            })
    
    def execute_emergency_action(self, action: Dict[str, Any]):
        """Execute emergency actions"""
        st.error(f"EMERGENCY: {action.get('action', 'Unknown action')}")
        
        # In production, would trigger actual safety systems
        # For now, log the action
        st.session_state.experiment_data['alerts'].append({
            'timestamp': datetime.now(),
            'level': 'emergency',
            'action': action.get('action'),
            'reason': action.get('reason', 'AI-triggered')
        })
    
    # UI Rendering Methods
    
    def render(self):
        """Render the complete UI"""
        self.render_header()
        self.render_sidebar()
        
        # Main content area
        if st.session_state.current_view == 'dashboard':
            self.render_dashboard()
        elif st.session_state.current_view == 'video':
            self.render_video_view()
        elif st.session_state.current_view == 'ai_insights':
            self.render_ai_insights()
        elif st.session_state.current_view == 'ml_analytics':
            self.render_ml_analytics()
        elif st.session_state.current_view == 'protocol':
            self.render_protocol_view()
    
    def render_header(self):
        """Render enhanced header"""
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            st.markdown("""
            <h1 style='color: #1f77b4; margin: 0;'>
                🧬 Ultimate Lab Assistant
            </h1>
            <p style='color: #666; margin: 0;'>
                AI-Powered Research Laboratory System
            </p>
            """, unsafe_allow_html=True)
        
        with col2:
            active = sum(st.session_state.systems.values())
            total = len(st.session_state.systems)
            st.metric("Active Systems", f"{active}/{total}")
        
        with col3:
            # Last update time
            st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))
        
        with col4:
            # Auto-refresh
            st_autorefresh(interval=1000, limit=None, key="ultimate_refresh")
    
    def render_sidebar(self):
        """Render enhanced sidebar"""
        with st.sidebar:
            st.markdown("## 🎛️ System Controls")
            
            # System toggles
            for system, active in st.session_state.systems.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{'🟢' if active else '🔴'} {system.replace('_', ' ').title()}")
                with col2:
                    if st.button("Toggle", key=f"toggle_{system}"):
                        self.toggle_system(system)
            
            st.divider()
            
            # Navigation
            st.markdown("## 📍 Navigation")
            views = {
                'dashboard': '📊 Dashboard',
                'video': '📹 Video Monitor',
                'ai_insights': '🧠 AI Insights',
                'ml_analytics': '📈 ML Analytics',
                'protocol': '📋 Protocol'
            }
            
            for view_id, view_name in views.items():
                if st.button(view_name, key=f"nav_{view_id}"):
                    st.session_state.current_view = view_id
            
            st.divider()
            
            # Quick stats
            st.markdown("## 📊 Quick Stats")
            
            # Yield prediction
            if st.session_state.experiment_data['ml_predictions']:
                latest_prediction = st.session_state.experiment_data['ml_predictions'][-1]
                st.metric(
                    "Predicted Yield",
                    f"{latest_prediction['predicted_yield']:.1f}%",
                    delta=f"±{(latest_prediction['confidence_interval'][1] - latest_prediction['confidence_interval'][0])/2:.1f}%"
                )
            
            # Safety status
            safety_status = self.get_safety_status()
            if safety_status['is_safe']:
                st.success("✅ All Systems Safe")
            else:
                st.error(f"⚠️ {len(safety_status['active_alerts'])} Active Alerts")
            
            # AI insights count
            insights_count = len(st.session_state.experiment_data['ai_insights'])
            st.metric("AI Insights", insights_count)
            
            st.divider()
            
            # Export functionality
            if st.button("📥 Export All Data", type="primary"):
                self.export_all_data()
    
    def render_dashboard(self):
        """Render main dashboard"""
        # System status overview
        st.markdown("### 🎯 System Status Overview")
        
        cols = st.columns(len(st.session_state.systems))
        for i, (system, active) in enumerate(st.session_state.systems.items()):
            with cols[i]:
                status_color = "#2ca02c" if active else "#cccccc"
                st.markdown(f"""
                <div class="system-card" style="border-color: {status_color};">
                    <span class="status-indicator {'status-active' if active else 'status-inactive'}"></span>
                    <strong>{system.replace('_', ' ').title()}</strong>
                </div>
                """, unsafe_allow_html=True)
        
        # Real-time metrics
        st.markdown("### 📊 Real-time Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Get current data
        sensors = self.get_current_sensors()
        
        with col1:
            temp = sensors.get('temperature', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>🌡️ Temperature</h3>
                <h1>{temp:.1f}°C</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            pressure = sensors.get('pressure', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔵 Pressure</h3>
                <h1>{pressure:.1f} kPa</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            progress = 0
            if st.session_state.protocol:
                status = st.session_state.protocol.get_protocol_status()
                progress = status['progress_percentage']
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈 Progress</h3>
                <h1>{progress:.0f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            alerts = len(st.session_state.experiment_data['alerts'])
            alert_color = "var(--danger-color)" if alerts > 0 else "var(--success-color)"
            
            st.markdown(f"""
            <div class="metric-card" style="background: {alert_color};">
                <h3>⚠️ Alerts</h3>
                <h1>{alerts}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # Live parameter trends
        st.markdown("### 📈 Live Parameter Trends")
        
        if st.session_state.safety_monitor and st.session_state.systems['safety']:
            self.render_parameter_trends()
        else:
            st.info("Enable safety monitoring to view live trends")
        
        # Recent events
        st.markdown("### 🔔 Recent Events")
        self.render_recent_events()
    
    def render_parameter_trends(self):
        """Render live parameter trend charts"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temperature', 'Pressure', 'Nitrogen', 'Oxygen'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        parameters = ['temperature', 'pressure', 'nitrogen', 'oxygen']
        colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728']
        
        for i, param in enumerate(parameters):
            row = i // 2 + 1
            col = i % 2 + 1
            
            # Get history
            history = st.session_state.safety_monitor.get_parameter_history(param, 15)
            
            if history:
                df = pd.DataFrame(history)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Add main trace
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['value'],
                        mode='lines+markers',
                        name=param.title(),
                        line=dict(color=colors[i], width=3),
                        marker=dict(size=8)
                    ),
                    row=row, col=col
                )
                
                # Add forecast if ML is active
                if st.session_state.systems['ml_prediction'] and st.session_state.experiment_forecaster:
                    forecast = st.session_state.experiment_forecaster.forecast_parameter_trends(
                        df[['timestamp', 'value']].set_index('timestamp'),
                        forecast_horizon=5
                    )
                    
                    if 'value' in forecast:
                        # Add forecast trace
                        future_times = pd.date_range(
                            start=df['timestamp'].max(),
                            periods=6,
                            freq='10S'
                        )[1:]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=future_times,
                                y=forecast['value']['forecast'],
                                mode='lines',
                                name=f"{param} forecast",
                                line=dict(color=colors[i], width=2, dash='dash'),
                                showlegend=False
                            ),
                            row=row, col=col
                        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_recent_events(self):
        """Render recent events timeline"""
        events = self.get_recent_events()
        
        if events:
            for event in events[:5]:
                # Determine event type and styling
                if event.get('level') == 'critical':
                    st.error(f"🚨 {event['timestamp'].strftime('%H:%M:%S')} - {event.get('message', event.get('description', 'Critical event'))}")
                elif event.get('type') == 'color_change':
                    st.warning(f"🎨 {event['timestamp'].strftime('%H:%M:%S')} - {event.get('description', 'Color change detected')}")
                elif event.get('type') == 'ai_insight':
                    st.info(f"🧠 {event['timestamp'].strftime('%H:%M:%S')} - {event.get('conclusion', 'AI insight generated')}")
                else:
                    st.write(f"📌 {event['timestamp'].strftime('%H:%M:%S')} - {event.get('description', event.get('message', 'Event occurred'))}")
        else:
            st.write("No recent events")
    
    def render_video_view(self):
        """Render video monitoring view"""
        st.markdown("### 📹 Video Monitoring")
        
        if not st.session_state.systems['video']:
            st.warning("Video monitoring is not active. Click 'Toggle' in the sidebar to enable.")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Video feed
            if st.session_state.video_monitor:
                frame = st.session_state.video_monitor.video_system.get_current_frame()
                
                if frame is not None:
                    # Convert frame to displayable format
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Add overlay information
                    overlay_text = f"Recording: {'Yes' if st.session_state.video_monitor.video_system.is_recording else 'No'}"
                    cv2.putText(frame_rgb, overlay_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Display frame
                    st.image(frame_rgb, channels="RGB", use_column_width=True)
                else:
                    st.info("No video feed available")
            
            # Video controls
            col1_1, col1_2, col1_3 = st.columns(3)
            
            with col1_1:
                if st.button("📹 Start Recording"):
                    st.session_state.video_monitor.video_system.start_recording()
            
            with col1_2:
                if st.button("⏹️ Stop Recording"):
                    st.session_state.video_monitor.video_system.stop_recording()
            
            with col1_3:
                if st.button("📸 Capture Frame"):
                    # Save current frame
                    pass
        
        with col2:
            st.markdown("#### 🎯 Video Events")
            
            # Recent video events
            video_events = st.session_state.experiment_data['video_events'][-10:]
            
            if video_events:
                for event in reversed(video_events):
                    confidence_color = "#2ca02c" if event['confidence'] > 0.8 else "#ff7f0e"
                    
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 0.5rem; margin: 0.5rem 0; border-radius: 5px; border-left: 3px solid {confidence_color};">
                        <strong>{event['type']}</strong><br>
                        {event['description']}<br>
                        <small>Confidence: {event['confidence']:.2f}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No video events detected")
    
    def render_ai_insights(self):
        """Render AI insights view"""
        st.markdown("### 🧠 AI Reasoning & Insights")
        
        if not st.session_state.systems['ai_reasoning']:
            st.warning("AI reasoning is not active. Click 'Toggle' in the sidebar to enable.")
            return
        
        # Manual reasoning trigger
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_query = st.text_input("Ask the AI assistant:", placeholder="What should I do next?")
        
        with col2:
            if st.button("🤔 Analyze", type="primary"):
                if user_query:
                    asyncio.run(self.process_voice_input(user_query))
        
        # Recent insights
        st.markdown("#### 💡 Recent AI Insights")
        
        insights = st.session_state.experiment_data['ai_insights'][-5:]
        
        if insights:
            for insight in reversed(insights):
                # Create insight card
                st.markdown(f"""
                <div class="reasoning-result">
                    <h4>{insight.get('query', 'Automated Analysis')}</h4>
                    <p><strong>Conclusion:</strong> {insight['conclusion']}</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {insight['confidence'] * 100}%;"></div>
                    </div>
                    <small>Confidence: {insight['confidence']:.2%}</small>
                """, unsafe_allow_html=True)
                
                # Recommendations
                if insight.get('recommendations'):
                    st.markdown("<strong>Recommendations:</strong>")
                    for rec in insight['recommendations'][:3]:
                        st.markdown(f"• {rec}")
                
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No AI insights generated yet. Try asking a question or wait for automated analysis.")
        
        # Reasoning history visualization
        if st.checkbox("Show reasoning history"):
            self.render_reasoning_history()
    
    def render_reasoning_history(self):
        """Render reasoning history chart"""
        if st.session_state.claude_reasoner:
            history = st.session_state.claude_reasoner.reasoning_history[-20:]
            
            if history:
                # Extract data for visualization
                timestamps = [h['timestamp'] for h in history]
                confidences = [h['result'].confidence for h in history]
                types = [h['result'].reasoning_type.value for h in history]
                
                # Create chart
                fig = go.Figure()
                
                # Group by reasoning type
                for reasoning_type in set(types):
                    mask = [t == reasoning_type for t in types]
                    fig.add_trace(go.Scatter(
                        x=[timestamps[i] for i, m in enumerate(mask) if m],
                        y=[confidences[i] for i, m in enumerate(mask) if m],
                        mode='markers+lines',
                        name=reasoning_type,
                        marker=dict(size=10)
                    ))
                
                fig.update_layout(
                    title="AI Reasoning Confidence Over Time",
                    xaxis_title="Time",
                    yaxis_title="Confidence",
                    yaxis=dict(range=[0, 1]),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def render_ml_analytics(self):
        """Render ML analytics view"""
        st.markdown("### 📈 Machine Learning Analytics")
        
        if not st.session_state.systems['ml_prediction']:
            st.warning("ML prediction is not active. Click 'Toggle' in the sidebar to enable.")
            return
        
        # Yield predictions
        st.markdown("#### 🎯 Yield Predictions")
        
        predictions = st.session_state.experiment_data['ml_predictions']
        
        if predictions:
            # Create prediction chart
            fig = go.Figure()
            
            # Extract data
            timestamps = [p['timestamp'] for p in predictions]
            yields = [p['predicted_yield'] for p in predictions]
            lower_bounds = [p['confidence_interval'][0] for p in predictions]
            upper_bounds = [p['confidence_interval'][1] for p in predictions]
            
            # Add prediction line
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=yields,
                mode='lines+markers',
                name='Predicted Yield',
                line=dict(color='#1f77b4', width=3)
            ))
            
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=timestamps + timestamps[::-1],
                y=upper_bounds + lower_bounds[::-1],
                fill='toself',
                fillcolor='rgba(31, 119, 180, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name='Confidence Interval'
            ))
            
            fig.update_layout(
                title="Yield Prediction Over Time",
                xaxis_title="Time",
                yaxis_title="Predicted Yield (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Latest prediction details
            latest = predictions[-1]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Latest Prediction",
                    f"{latest['predicted_yield']:.1f}%",
                    delta=f"±{(latest['confidence_interval'][1] - latest['confidence_interval'][0])/2:.1f}%"
                )
            
            with col2:
                st.metric(
                    "Confidence Range",
                    f"{latest['confidence_interval'][0]:.1f}-{latest['confidence_interval'][1]:.1f}%"
                )
            
            with col3:
                # Most important feature
                if latest.get('feature_importance'):
                    top_feature = max(latest['feature_importance'].items(), 
                                    key=lambda x: x[1])
                    st.metric(
                        "Key Factor",
                        top_feature[0].replace('_', ' ').title(),
                        f"{top_feature[1]:.2f}"
                    )
        else:
            st.info("No yield predictions available yet. Complete more protocol steps to generate predictions.")
        
        # Optimization suggestions
        st.markdown("#### 🔧 Optimization Suggestions")
        
        if st.button("🎯 Optimize Reaction Conditions"):
            features = self.extract_current_features()
            if features and st.session_state.reaction_optimizer:
                with st.spinner("Optimizing..."):
                    optimization = st.session_state.reaction_optimizer.optimize(
                        features, {}, n_iterations=20
                    )
                    
                    st.success(f"Optimization complete! Predicted yield: {optimization['predicted_yield']:.1f}%")
                    
                    st.markdown("**Recommended Actions:**")
                    for rec in optimization['recommendations']:
                        st.markdown(f"• {rec}")
        
        # Anomaly detection
        st.markdown("#### 🔍 Anomaly Detection")
        
        if st.session_state.anomaly_detector:
            # Check current parameters
            sensors = self.get_current_sensors()
            if sensors:
                anomalies = st.session_state.anomaly_detector.detect_anomalies(sensors)
                
                if anomalies:
                    for anomaly in anomalies:
                        severity_color = "#d62728" if anomaly['severity'] == 'high' else "#ff7f0e"
                        st.markdown(f"""
                        <div class="alert-card" style="background: {severity_color};">
                            <strong>Anomaly Detected: {anomaly['parameter']}</strong><br>
                            Value: {anomaly['value']:.2f} (Z-score: {anomaly['z_score']:.2f})<br>
                            Expected range: {anomaly['expected_range'][0]:.2f} - {anomaly['expected_range'][1]:.2f}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ No anomalies detected")
    
    def render_protocol_view(self):
        """Render protocol view"""
        st.markdown("### 📋 Protocol Management")
        
        if not st.session_state.protocol:
            st.warning("Protocol system not initialized.")
            return
        
        # Protocol controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("▶️ Start" if not st.session_state.systems['protocol'] else "⏸️ Pause"):
                if not st.session_state.systems['protocol']:
                    st.session_state.protocol.start_protocol()
                    st.session_state.systems['protocol'] = True
                else:
                    st.session_state.protocol.pause_protocol()
        
        with col2:
            if st.button("⏭️ Skip Step"):
                current_step = st.session_state.protocol.get_current_step()
                if current_step:
                    st.session_state.protocol.skip_step(current_step.id, "User skipped")
        
        with col3:
            status = st.session_state.protocol.get_protocol_status()
            st.metric("Progress", f"{status['completed_steps']}/{status['total_steps']}")
        
        with col4:
            if status['elapsed_time']:
                elapsed = int(status['elapsed_time'])
                st.metric("Time", f"{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}")
        
        # Current step details
        current_step = st.session_state.protocol.get_current_step()
        
        if current_step:
            st.markdown(f"""
            <div class="system-card" style="border-color: #ff7f0e;">
                <h4>Current Step: {current_step.name}</h4>
                <p>{current_step.description}</p>
                <p><strong>Type:</strong> {current_step.step_type.value}</p>
                <p><strong>Status:</strong> {current_step.status.value}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Time remaining for timed steps
            if 'time_remaining' in status.get('current_step', {}):
                remaining = int(status['current_step']['time_remaining'])
                st.progress(1 - (remaining / current_step.duration_seconds))
                st.write(f"Time remaining: {remaining // 60}:{remaining % 60:02d}")
            
            # Data recording interface
            if current_step.status == StepStatus.READY and current_step.data_to_record:
                st.markdown("#### 📝 Record Data")
                
                for data_type in current_step.data_to_record:
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.write(data_type.replace('_', ' ').title())
                    
                    with col2:
                        value = st.number_input(
                            "Value",
                            key=f"proto_{current_step.id}_{data_type}",
                            format="%.4f"
                        )
                    
                    with col3:
                        if st.button("Save", key=f"save_{current_step.id}_{data_type}"):
                            st.session_state.protocol.record_data(
                                current_step.id,
                                data_type,
                                value
                            )
                            st.session_state.experiment_data['measurements'][data_type] = {
                                'value': value,
                                'timestamp': datetime.now()
                            }
                            st.success("Saved!")
        
        # Protocol timeline
        st.markdown("#### 📊 Protocol Timeline")
        self.render_protocol_timeline()
    
    def render_protocol_timeline(self):
        """Render protocol timeline visualization"""
        steps = st.session_state.protocol.protocol_steps
        
        # Create timeline chart
        fig = go.Figure()
        
        # Define colors for different states
        colors = {
            StepStatus.COMPLETED: "#2ca02c",
            StepStatus.IN_PROGRESS: "#ff7f0e",
            StepStatus.READY: "#1f77b4",
            StepStatus.PENDING: "#cccccc",
            StepStatus.SKIPPED: "#888888",
            StepStatus.FAILED: "#d62728"
        }
        
        y_pos = 0
        for i, step in enumerate(steps):
            # Add step bar
            fig.add_trace(go.Scatter(
                x=[i, i+1],
                y=[y_pos, y_pos],
                mode='lines',
                line=dict(color=colors[step.status], width=20),
                showlegend=False,
                hovertext=step.name,
                hoverinfo='text'
            ))
            
            # Add step marker
            if step.status == StepStatus.IN_PROGRESS:
                fig.add_trace(go.Scatter(
                    x=[i+0.5],
                    y=[y_pos],
                    mode='markers',
                    marker=dict(size=15, color='white', line=dict(color='#ff7f0e', width=3)),
                    showlegend=False
                ))
        
        fig.update_layout(
            height=200,
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[-0.5, len(steps)+0.5]
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[-1, 1]
            ),
            plot_bgcolor='white',
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Legend
        cols = st.columns(6)
        for i, (status, color) in enumerate(colors.items()):
            with cols[i % 6]:
                st.markdown(f"""
                <span style="display: inline-block; width: 12px; height: 12px; 
                background: {color}; border-radius: 2px; margin-right: 5px;"></span>
                {status.value}
                """, unsafe_allow_html=True)
    
    def toggle_system(self, system_name: str):
        """Toggle a system on/off"""
        current_state = st.session_state.systems[system_name]
        
        if not current_state:
            # Turning on
            if system_name == 'voice' and st.session_state.voice_processor:
                st.session_state.voice_processor.start()
            elif system_name == 'safety' and st.session_state.safety_monitor:
                st.session_state.safety_monitor.start()
            elif system_name == 'agents':
                for agent in st.session_state.agents.values():
                    agent.start()
            elif system_name == 'protocol' and st.session_state.protocol:
                st.session_state.protocol.start_protocol()
            elif system_name == 'video' and st.session_state.video_monitor:
                st.session_state.video_monitor.start_monitoring()
            elif system_name == 'ai_reasoning':
                # AI reasoning is always ready
                pass
            elif system_name == 'ml_prediction':
                # ML prediction is always ready
                # Could load models here if needed
                pass
        else:
            # Turning off
            if system_name == 'voice' and st.session_state.voice_processor:
                st.session_state.voice_processor.stop()
            elif system_name == 'safety' and st.session_state.safety_monitor:
                st.session_state.safety_monitor.stop()
            elif system_name == 'agents':
                for agent in st.session_state.agents.values():
                    agent.stop()
            elif system_name == 'protocol' and st.session_state.protocol:
                st.session_state.protocol.pause_protocol()
            elif system_name == 'video' and st.session_state.video_monitor:
                st.session_state.video_monitor.stop_monitoring()
        
        # Toggle state
        st.session_state.systems[system_name] = not current_state
        st.rerun()
    
    def export_all_data(self):
        """Export all experiment data"""
        export_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'active_systems': st.session_state.systems,
                'experiment_id': f"exp_{int(time.time())}"
            },
            'measurements': st.session_state.experiment_data['measurements'],
            'observations': st.session_state.experiment_data['observations'],
            'calculations': st.session_state.experiment_data['calculations'],
            'alerts': st.session_state.experiment_data['alerts'],
            'video_events': st.session_state.experiment_data['video_events'],
            'ai_insights': st.session_state.experiment_data['ai_insights'],
            'ml_predictions': st.session_state.experiment_data['ml_predictions'],
            'protocol_data': st.session_state.protocol.export_protocol_data() if st.session_state.protocol else None
        }
        
        # Convert to JSON
        json_str = json.dumps(export_data, indent=2, default=str)
        
        # Create download button
        st.download_button(
            label="Download Complete Dataset",
            data=json_str,
            file_name=f"lab_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Main execution
def main():
    """Main entry point"""
    assistant = UltimateLabAssistant()
    assistant.render()

if __name__ == "__main__":
    main()