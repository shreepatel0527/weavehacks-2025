"""
Ultimate Lab Assistant Superset - Complete implementation with all features from iterations 0, 1, and 2
"""
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import asyncio
import threading
import time
from datetime import datetime, timedelta
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
import weave
from typing import Dict, List, Optional, Any, Tuple
import logging
from collections import deque
import queue

# Add all module paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "weavehacks_flow-1/src"))

# Import all modules from iterations 0, 1, and 2
# Iteration 0 modules
from weavehacks_flow.config.settings import Settings, get_chemistry_config, get_safety_config
from weavehacks_flow.utils.chemistry_calculations import (
    calculate_sulfur_amount, calculate_nabh4_amount, calculate_percent_yield,
    calculate_toab_ratio, calculate_all_reagents
)
from weavehacks_flow.utils.error_handling import (
    ErrorHandler, safe_execute, ErrorSeverity, LabAssistantError
)
from weavehacks_flow.main import ExperimentFlow

# Iteration 1 modules
from thread_safe_safety_monitor import ThreadSafeRealtimeSafetyMonitor, SafetyLevel, SafetyEvent
from enhanced_voice_accuracy import EnhancedVoiceProcessor, LabCommandProcessor
from optimized_ui_components import OptimizedUI, DataCache, PerformanceMonitor
from batch_api_operations import LabAPIClient, CachedBatchAPIClient
from enhanced_agent_system import (
    MessageBus, EnhancedAgent, DataCollectionAgent, 
    SafetyMonitoringAgent, CoordinatorAgent, Priority, Task
)
from protocol_automation import ProtocolAutomation, StepStatus

# Iteration 2 modules
from enhanced_video_monitoring import EnhancedVideoMonitoringSystem, VideoEvent, EventType
from claude_flow_integration import ClaudeFlowReasoner, ClaudeFlowOrchestrator, ReasoningContext
from predictive_models import (
    YieldPredictor, ReactionOptimizer, AnomalyDetector, 
    ExperimentForecaster, ExperimentFeatures
)
from cloud_connectivity import CloudConnector, CollaborationManager, RemoteMonitoringService
from error_recovery import ErrorRecoveryManager, StateManager, CircuitBreaker
from connection_pool_manager import ConnectionPoolManager, get_pool_manager
from advanced_caching_layer import CacheManager, get_cache_manager, cached

# Configure page
st.set_page_config(
    page_title="Ultimate Lab Assistant Superset",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ultimate_lab_assistant')

class UltimateLabAssistantSuperset:
    """Complete lab assistant system with all features integrated"""
    
    def __init__(self):
        # Core settings
        self.settings = Settings()
        
        # Error handling
        self.error_handler = ErrorHandler()
        self.recovery_manager = ErrorRecoveryManager()
        
        # Safety monitoring
        self.safety_monitor = ThreadSafeRealtimeSafetyMonitor(
            check_interval=0.5
        )
        
        # Voice processing
        self.voice_processor = EnhancedVoiceProcessor(
            use_noise_reduction=True
        )
        self.command_processor = LabCommandProcessor()
        
        # Video monitoring
        self.video_system = EnhancedVideoMonitoringSystem(
            enable_ml=True,
            enable_recording=True
        )
        
        # Agent system
        self.message_bus = MessageBus()
        self.agents = self._initialize_agents()
        
        # Protocol automation
        self.protocol_automation = ProtocolAutomation()
        
        # AI reasoning
        self.claude_reasoner = ClaudeFlowReasoner()
        self.claude_orchestrator = ClaudeFlowOrchestrator(self.claude_reasoner)
        
        # Predictive models
        self.yield_predictor = YieldPredictor()
        self.reaction_optimizer = ReactionOptimizer(self.yield_predictor)
        self.anomaly_detector = AnomalyDetector()
        self.forecaster = ExperimentForecaster()
        
        # Cloud connectivity
        self.cloud_config = {
            'user_id': 'lab_user_001',
            'api_key': 'demo_key',
            'websocket_url': 'wss://lab-cloud.example.com/ws',
            'auth_url': 'https://lab-cloud.example.com/auth',
            'storage_backend': 'azure'
        }
        self.cloud_connector = CloudConnector(self.cloud_config)
        self.collaboration_manager = CollaborationManager(self.cloud_connector)
        self.remote_monitor = RemoteMonitoringService(self.cloud_connector)
        
        # Connection pooling
        self.pool_manager = None
        
        # Caching
        self.cache_manager = get_cache_manager()
        self._setup_caches()
        
        # API client
        self.api_client = LabAPIClient("https://api.lab-assistant.com")
        
        # UI components
        self.ui = OptimizedUI()
        
        # Experiment state
        self.experiment_state = {
            'id': None,
            'status': 'idle',
            'start_time': None,
            'protocol': None,
            'current_step': None,
            'measurements': deque(maxlen=1000),
            'events': deque(maxlen=500),
            'gold_mass': 0.1576  # Default
        }
        
        # Statistics
        self.stats = {
            'experiments_run': 0,
            'total_runtime': 0,
            'successful_completions': 0,
            'safety_incidents': 0
        }
        
        # Initialize W&B
        weave.init('ultimate-lab-assistant-superset')
        
        logger.info("Ultimate Lab Assistant Superset initialized")
    
    def _initialize_agents(self) -> Dict[str, EnhancedAgent]:
        """Initialize all agents"""
        agents = {}
        
        # Data collection agent
        agents['data_collector'] = DataCollectionAgent(
            name="DataCollector",
            message_bus=self.message_bus
        )
        
        # Safety monitoring agent
        agents['safety_monitor'] = SafetyMonitoringAgent(
            name="SafetyMonitor",
            message_bus=self.message_bus
        )
        
        # Coordinator agent
        agents['coordinator'] = CoordinatorAgent(
            name="Coordinator",
            message_bus=self.message_bus,
            agents=[agents['data_collector'], agents['safety_monitor']]
        )
        
        return agents
    
    def _setup_caches(self):
        """Setup caching layers"""
        # Default cache for general use
        self.cache_manager.create_cache(
            'default',
            memory_size=2000,
            memory_mb=200
        )
        
        # Chemistry calculations cache
        self.cache_manager.create_cache(
            'chemistry',
            memory_size=1000,
            memory_mb=50
        )
        
        # Sensor data cache
        self.cache_manager.create_cache(
            'sensors',
            memory_size=5000,
            memory_mb=100
        )
        
        # ML predictions cache
        self.cache_manager.create_cache(
            'predictions',
            memory_size=500,
            memory_mb=50
        )
    
    async def initialize(self):
        """Initialize all async components"""
        try:
            # Initialize connection pools
            pool_config = {
                'database': {
                    'dsn': 'postgresql://user:password@localhost/lab_assistant',
                    'min_size': 10,
                    'max_size': 20
                },
                'redis': {
                    'url': 'redis://localhost:6379',
                    'min_connections': 5,
                    'max_connections': 10
                },
                'http': {
                    'connector_limit': 100,
                    'connector_limit_per_host': 30
                }
            }
            
            self.pool_manager = await get_pool_manager()
            await self.pool_manager.initialize(pool_config)
            
            # Initialize cloud connection
            await self.cloud_connector.connect()
            
            # Initialize API client
            await self.api_client.start()
            
            # Start cache manager
            await self.cache_manager.start()
            
            # Start all agents
            for agent in self.agents.values():
                agent.start()
            
            logger.info("All async components initialized")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            self.error_handler.handle_error(e, "initialization")
    
    def start_experiment(self, protocol_name: str = 'nanoparticle_synthesis'):
        """Start new experiment"""
        try:
            # Generate experiment ID
            self.experiment_state['id'] = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.experiment_state['status'] = 'running'
            self.experiment_state['start_time'] = datetime.now()
            
            # Load protocol
            self.protocol_automation.load_protocol(protocol_name)
            self.protocol_automation.start_protocol()
            self.experiment_state['protocol'] = protocol_name
            
            # Start monitoring
            self.safety_monitor.start_monitoring()
            self.video_system.start()
            self.voice_processor.start()
            
            # Start remote monitoring
            self.remote_monitor.start_monitoring(self.experiment_state['id'])
            
            # Update stats
            self.stats['experiments_run'] += 1
            
            # Log to W&B
            weave.log({
                'experiment_started': {
                    'id': self.experiment_state['id'],
                    'protocol': protocol_name,
                    'timestamp': datetime.now().isoformat()
                }
            })
            
            logger.info(f"Experiment {self.experiment_state['id']} started")
            
        except Exception as e:
            self.error_handler.handle_error(e, "experiment_start", ErrorSeverity.HIGH)
    
    def stop_experiment(self):
        """Stop current experiment"""
        try:
            if self.experiment_state['status'] != 'running':
                return
            
            # Calculate runtime
            if self.experiment_state['start_time']:
                runtime = (datetime.now() - self.experiment_state['start_time']).total_seconds()
                self.stats['total_runtime'] += runtime
            
            # Stop monitoring
            self.safety_monitor.stop_monitoring()
            self.video_system.stop()
            self.voice_processor.stop()
            self.remote_monitor.stop_monitoring()
            
            # Update state
            self.experiment_state['status'] = 'completed'
            self.stats['successful_completions'] += 1
            
            # Generate final report
            report = self._generate_experiment_report()
            
            # Log completion
            weave.log({
                'experiment_completed': {
                    'id': self.experiment_state['id'],
                    'runtime_seconds': runtime,
                    'events_count': len(self.experiment_state['events']),
                    'timestamp': datetime.now().isoformat()
                }
            })
            
            logger.info(f"Experiment {self.experiment_state['id']} completed")
            
            return report
            
        except Exception as e:
            self.error_handler.handle_error(e, "experiment_stop", ErrorSeverity.MEDIUM)
    
    @cached(cache_name='chemistry', ttl=3600)
    def calculate_reagents(self, gold_mass: float) -> Dict[str, Any]:
        """Calculate all reagent amounts with caching"""
        try:
            # Calculate all reagents
            reagents = calculate_all_reagents(gold_mass)
            
            # Add safety margins
            reagents['safety_margins'] = {
                'temperature': '±2°C',
                'ph': '±0.5',
                'time': '+10%'
            }
            
            return reagents
            
        except Exception as e:
            return self.error_handler.handle_error(
                e, "reagent_calculation", 
                ErrorSeverity.HIGH,
                {'gold_mass': gold_mass}
            )
    
    async def predict_yield(self, features: ExperimentFeatures) -> Dict[str, Any]:
        """Predict experiment yield with ML"""
        try:
            # Check cache first
            cache_key = self.cache_manager.cache_key(
                'yield_prediction',
                features.__dict__
            )
            
            cached_result = await self.cache_manager.get_cache('predictions').get(cache_key)
            if cached_result:
                return cached_result
            
            # Make prediction
            prediction = self.yield_predictor.predict(features)
            
            # Optimize conditions
            optimization = await self.reaction_optimizer.optimize(
                features,
                {},
                n_iterations=20
            )
            
            result = {
                'predicted_yield': prediction.prediction,
                'confidence_interval': prediction.confidence_interval,
                'optimization_suggestions': optimization['recommendations'],
                'feature_importance': prediction.feature_importance
            }
            
            # Cache result
            await self.cache_manager.get_cache('predictions').put(
                cache_key, result
            )
            
            return result
            
        except Exception as e:
            return self.error_handler.handle_error(
                e, "yield_prediction",
                ErrorSeverity.MEDIUM
            )
    
    async def process_voice_command(self, command_text: str) -> Dict[str, Any]:
        """Process voice command with AI reasoning"""
        try:
            # Process command
            command = self.command_processor.process(command_text)
            
            if not command['command']:
                return {'status': 'not_understood', 'text': command_text}
            
            # Create reasoning context
            context = ReasoningContext(
                experiment_state=self.experiment_state,
                historical_data=list(self.experiment_state['measurements']),
                sensor_readings=self.safety_monitor.current_readings,
                recent_events=list(self.experiment_state['events'][-10:]),
                safety_status={'is_safe': self.safety_monitor.current_safety_level == SafetyLevel.SAFE},
                user_query=command_text
            )
            
            # Get AI reasoning
            reasoning_result = await self.claude_reasoner.reason(context)
            
            # Execute command
            result = await self._execute_command(command, reasoning_result)
            
            return {
                'status': 'success',
                'command': command,
                'reasoning': reasoning_result.conclusion,
                'result': result
            }
            
        except Exception as e:
            return self.error_handler.handle_error(
                e, "voice_command",
                ErrorSeverity.MEDIUM
            )
    
    async def _execute_command(self, command: Dict, reasoning: Any) -> Any:
        """Execute processed command"""
        cmd_type = command['command']
        
        if cmd_type == 'start_experiment':
            self.start_experiment()
            return "Experiment started"
            
        elif cmd_type == 'stop_experiment':
            report = self.stop_experiment()
            return f"Experiment stopped. {report.get('summary', '')}"
            
        elif cmd_type == 'emergency_stop':
            await self._emergency_stop()
            return "EMERGENCY STOP EXECUTED"
            
        elif cmd_type == 'status_report':
            return self._get_status_report()
            
        elif cmd_type in ['read_temperature', 'read_pressure']:
            param = cmd_type.split('_')[1]
            value = self.safety_monitor.current_readings.get(param, 'Unknown')
            return f"Current {param}: {value}"
            
        else:
            return f"Command {cmd_type} not implemented"
    
    async def _emergency_stop(self):
        """Execute emergency stop"""
        logger.critical("EMERGENCY STOP INITIATED")
        
        # Stop all operations
        self.stop_experiment()
        
        # Send alerts
        await self.remote_monitor.send_alert(
            'emergency_stop',
            'Emergency stop executed',
            'critical'
        )
        
        # Log incident
        self.stats['safety_incidents'] += 1
        
        weave.log({
            'emergency_stop': {
                'timestamp': datetime.now().isoformat(),
                'experiment_id': self.experiment_state['id']
            }
        })
    
    def _get_status_report(self) -> str:
        """Generate status report"""
        if self.experiment_state['status'] != 'running':
            return "No experiment running"
        
        runtime = (datetime.now() - self.experiment_state['start_time']).total_seconds()
        
        report = f"""
        Experiment Status:
        - ID: {self.experiment_state['id']}
        - Runtime: {runtime/60:.1f} minutes
        - Current Step: {self.protocol_automation.get_current_step().get('name', 'Unknown')}
        - Safety Level: {self.safety_monitor.current_safety_level.value}
        - Events Detected: {len(self.experiment_state['events'])}
        - Temperature: {self.safety_monitor.current_readings.get('temperature', 'N/A')}°C
        - Pressure: {self.safety_monitor.current_readings.get('pressure', 'N/A')} kPa
        """
        
        return report.strip()
    
    def _generate_experiment_report(self) -> Dict[str, Any]:
        """Generate comprehensive experiment report"""
        # Calculate metrics
        runtime = 0
        if self.experiment_state['start_time']:
            runtime = (datetime.now() - self.experiment_state['start_time']).total_seconds()
        
        # Get event summary
        event_summary = {}
        for event in self.experiment_state['events']:
            event_type = event.get('type', 'unknown')
            event_summary[event_type] = event_summary.get(event_type, 0) + 1
        
        # Get video summary if available
        video_summary = {}
        if self.video_system:
            video_summary = self.video_system.get_event_summary()
        
        report = {
            'experiment_id': self.experiment_state['id'],
            'protocol': self.experiment_state['protocol'],
            'runtime_minutes': runtime / 60,
            'completion_status': self.experiment_state['status'],
            'measurements_collected': len(self.experiment_state['measurements']),
            'events': event_summary,
            'video_analysis': video_summary,
            'safety_incidents': self.stats['safety_incidents'],
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def render_ui(self):
        """Render the complete UI"""
        st.title("🧬 Ultimate Lab Assistant Superset")
        
        # Sidebar
        with st.sidebar:
            st.header("⚙️ Controls")
            
            # Experiment controls
            if st.button("▶️ Start Experiment", disabled=self.experiment_state['status'] == 'running'):
                self.start_experiment()
                st.success("Experiment started!")
            
            if st.button("⏹️ Stop Experiment", disabled=self.experiment_state['status'] != 'running'):
                report = self.stop_experiment()
                st.info("Experiment stopped")
                st.json(report)
            
            if st.button("🚨 Emergency Stop"):
                asyncio.run(self._emergency_stop())
                st.error("EMERGENCY STOP EXECUTED")
            
            st.divider()
            
            # Reagent calculator
            st.header("🧪 Reagent Calculator")
            gold_mass = st.number_input(
                "Gold mass (g)",
                min_value=0.001,
                max_value=1.0,
                value=self.experiment_state['gold_mass'],
                step=0.001,
                format="%.4f"
            )
            
            if st.button("Calculate Reagents"):
                reagents = self.calculate_reagents(gold_mass)
                st.json(reagents)
            
            st.divider()
            
            # Statistics
            st.header("📊 Statistics")
            st.metric("Experiments Run", self.stats['experiments_run'])
            st.metric("Success Rate", 
                     f"{self.stats['successful_completions'] / max(1, self.stats['experiments_run']) * 100:.1f}%")
            st.metric("Total Runtime", f"{self.stats['total_runtime'] / 3600:.1f} hours")
            st.metric("Safety Incidents", self.stats['safety_incidents'])
        
        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔬 Experiment", "📊 Analytics", "🎥 Video", "🤖 AI Assistant", "☁️ Cloud"
        ])
        
        with tab1:
            self._render_experiment_tab()
        
        with tab2:
            self._render_analytics_tab()
        
        with tab3:
            self._render_video_tab()
        
        with tab4:
            self._render_ai_tab()
        
        with tab5:
            self._render_cloud_tab()
    
    def _render_experiment_tab(self):
        """Render experiment monitoring tab"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🌡️ Real-time Monitoring")
            
            # Safety status
            safety_level = self.safety_monitor.current_safety_level
            safety_color = {
                SafetyLevel.SAFE: "green",
                SafetyLevel.WARNING: "orange",
                SafetyLevel.CRITICAL: "red",
                SafetyLevel.EMERGENCY: "red"
            }.get(safety_level, "gray")
            
            st.markdown(
                f"Safety Status: <span style='color:{safety_color}; font-size:24px'>■</span> {safety_level.value.upper()}",
                unsafe_allow_html=True
            )
            
            # Current readings
            readings = self.safety_monitor.current_readings
            
            if readings:
                metrics = st.columns(3)
                with metrics[0]:
                    st.metric("Temperature", f"{readings.get('temperature', 0):.1f}°C")
                with metrics[1]:
                    st.metric("Pressure", f"{readings.get('pressure', 0):.1f} kPa")
                with metrics[2]:
                    st.metric("Stirring", f"{readings.get('stirring_rpm', 0):.0f} RPM")
                
                # Real-time chart
                if hasattr(st.session_state, 'chart_data'):
                    chart_data = st.session_state.chart_data
                else:
                    chart_data = {'temperature': [], 'pressure': [], 'stirring_rpm': []}
                    st.session_state.chart_data = chart_data
                
                # Update chart data
                for param, value in readings.items():
                    if param in chart_data:
                        chart_data[param].append({
                            'timestamp': datetime.now(),
                            'value': value
                        })
                        # Keep last 100 points
                        if len(chart_data[param]) > 100:
                            chart_data[param].pop(0)
                
                # Create chart
                fig = self.ui.charts.create_realtime_chart(
                    chart_data,
                    "Real-time Sensor Data"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📋 Protocol Status")
            
            if self.experiment_state['status'] == 'running':
                current_step = self.protocol_automation.get_current_step()
                if current_step:
                    st.info(f"Current Step: {current_step.get('name', 'Unknown')}")
                    
                    # Progress
                    progress = self.protocol_automation.get_progress()
                    st.progress(progress)
                    st.caption(f"{int(progress * 100)}% complete")
                    
                    # Step details
                    with st.expander("Step Details"):
                        st.json(current_step)
                
                # Recent events
                st.subheader("📌 Recent Events")
                events = list(self.experiment_state['events'])[-5:]
                for event in reversed(events):
                    event_type = event.get('type', 'unknown')
                    event_time = event.get('timestamp', '')
                    st.caption(f"{event_time}: {event_type}")
            else:
                st.info("No experiment running")
    
    def _render_analytics_tab(self):
        """Render analytics tab"""
        st.subheader("📈 Experiment Analytics")
        
        # Yield prediction
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Yield Prediction")
            
            # Get current conditions
            features = ExperimentFeatures(
                gold_mass=self.experiment_state['gold_mass'],
                toab_mass=0.25,
                sulfur_mass=0.052,
                nabh4_mass=0.015,
                temperature=self.safety_monitor.current_readings.get('temperature', 23.0),
                stirring_rpm=self.safety_monitor.current_readings.get('stirring_rpm', 1100),
                reaction_time=3600,
                ph=self.safety_monitor.current_readings.get('ph', 7.0),
                ambient_temp=22.0,
                humidity=50.0,
                pressure=self.safety_monitor.current_readings.get('pressure', 101.3)
            )
            
            if st.button("Predict Yield"):
                with st.spinner("Calculating..."):
                    prediction = asyncio.run(self.predict_yield(features))
                    
                    st.metric(
                        "Predicted Yield",
                        f"{prediction['predicted_yield']:.1f}%",
                        f"±{(prediction['confidence_interval'][1] - prediction['confidence_interval'][0])/2:.1f}%"
                    )
                    
                    # Optimization suggestions
                    if prediction.get('optimization_suggestions'):
                        st.markdown("**Optimization Suggestions:**")
                        for suggestion in prediction['optimization_suggestions']:
                            st.caption(f"• {suggestion}")
        
        with col2:
            st.markdown("### Anomaly Detection")
            
            # Check for anomalies
            if hasattr(self.anomaly_detector, 'baseline_stats') and self.anomaly_detector.baseline_stats:
                anomalies = self.anomaly_detector.detect_anomalies(
                    self.safety_monitor.current_readings
                )
                
                if anomalies:
                    st.warning(f"{len(anomalies)} anomalies detected!")
                    for anomaly in anomalies:
                        st.caption(
                            f"• {anomaly['parameter']}: {anomaly['value']:.2f} "
                            f"(Z-score: {anomaly['z_score']:.2f})"
                        )
                else:
                    st.success("No anomalies detected")
            else:
                st.info("Anomaly detection not calibrated")
        
        # Historical data
        st.markdown("### Historical Data")
        
        if self.experiment_state['measurements']:
            # Convert to DataFrame
            df = pd.DataFrame(list(self.experiment_state['measurements']))
            
            # Display table
            st.dataframe(
                df.tail(50),
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                "Download Data",
                csv,
                f"experiment_data_{self.experiment_state['id']}.csv",
                "text/csv"
            )
    
    def _render_video_tab(self):
        """Render video monitoring tab"""
        st.subheader("🎥 Video Monitoring")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Video feed placeholder
            video_placeholder = st.empty()
            
            # Get current frame
            if self.video_system.is_monitoring:
                frame = self.video_system.get_current_frame()
                if frame is not None:
                    # Convert to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Display
                    video_placeholder.image(
                        frame_rgb,
                        caption="Live Video Feed",
                        use_column_width=True
                    )
            else:
                video_placeholder.info("Video monitoring not active")
        
        with col2:
            st.markdown("### Video Events")
            
            # Get event summary
            if self.video_system:
                summary = self.video_system.get_event_summary()
                
                st.metric("Total Events", summary.get('total_events', 0))
                
                # Event breakdown
                if summary.get('event_counts'):
                    st.markdown("**Event Types:**")
                    for event_type, count in summary['event_counts'].items():
                        st.caption(f"• {event_type}: {count}")
                
                # Recording status
                if self.video_system.video_writer:
                    st.success("🔴 Recording active")
                else:
                    st.info("Recording not active")
    
    def _render_ai_tab(self):
        """Render AI assistant tab"""
        st.subheader("🤖 AI Laboratory Assistant")
        
        # Voice input status
        if self.voice_processor.is_recording:
            st.success("🎤 Listening...")
        else:
            st.info("🎤 Voice input ready")
        
        # Chat interface
        st.markdown("### Chat with AI Assistant")
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        user_input = st.chat_input("Ask me anything about your experiment...")
        
        if user_input:
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Process with AI
            with st.spinner("Thinking..."):
                response = asyncio.run(self.process_voice_command(user_input))
                
                # Format response
                if response['status'] == 'success':
                    ai_response = f"{response['reasoning']}\n\nResult: {response['result']}"
                else:
                    ai_response = "I'm sorry, I didn't understand that command."
                
                # Add AI response
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": ai_response
                })
                
                # Rerun to update chat
                st.rerun()
        
        # AI Insights
        st.markdown("### AI Insights")
        
        if st.button("Generate Experiment Analysis"):
            with st.spinner("Analyzing experiment..."):
                # Create full context
                context = {
                    'experiment_state': self.experiment_state,
                    'sensor_readings': self.safety_monitor.current_readings,
                    'recent_events': list(self.experiment_state['events'][-20:]),
                    'safety_status': {
                        'is_safe': self.safety_monitor.current_safety_level == SafetyLevel.SAFE,
                        'level': self.safety_monitor.current_safety_level.value
                    }
                }
                
                # Get AI analysis
                analysis = asyncio.run(
                    self.claude_orchestrator.orchestrate_experiment_analysis(context)
                )
                
                # Display results
                st.json(analysis)
    
    def _render_cloud_tab(self):
        """Render cloud connectivity tab"""
        st.subheader("☁️ Cloud Connectivity")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Connection Status")
            
            if self.cloud_connector.is_connected:
                st.success("✅ Connected to cloud")
                
                # Show active users
                if self.collaboration_manager.active_users:
                    st.markdown("**Active Users:**")
                    for user_id, info in self.collaboration_manager.active_users.items():
                        st.caption(f"• {info['name']} ({info['role']})")
            else:
                st.error("❌ Not connected to cloud")
                
                if st.button("Connect to Cloud"):
                    asyncio.run(self.cloud_connector.connect())
                    st.rerun()
        
        with col2:
            st.markdown("### Data Sync")
            
            # Sync status
            if self.remote_monitor and hasattr(self.remote_monitor, 'experiment_id'):
                st.info(f"Monitoring: {self.remote_monitor.experiment_id}")
                
                # Manual sync
                if st.button("Force Sync"):
                    self.remote_monitor.send_sensor_data()
                    self.remote_monitor.send_buffered_events()
                    st.success("Data synchronized")
            else:
                st.info("Remote monitoring not active")
        
        # Collaboration
        st.markdown("### Collaboration")
        
        # Send message
        message = st.text_input("Send message to team:")
        if st.button("Send") and message:
            self.collaboration_manager.broadcast_update(
                'chat_message',
                {'user': self.cloud_connector.user_id, 'message': message},
                self.experiment_state['id']
            )
            st.success("Message sent")

def main():
    """Main application entry point"""
    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = UltimateLabAssistantSuperset()
        
        # Initialize async components
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(st.session_state.assistant.initialize())
    
    assistant = st.session_state.assistant
    
    # Auto-refresh for real-time updates
    st_autorefresh(interval=1000, limit=10000, key="assistant_refresh")
    
    # Render UI
    assistant.render_ui()
    
    # Register callbacks
    if not hasattr(st.session_state, 'callbacks_registered'):
        # Safety alert callback
        def safety_alert_handler(event: SafetyEvent):
            st.session_state.assistant.experiment_state['events'].append({
                'type': 'safety_alert',
                'level': event.level.value,
                'message': event.message,
                'timestamp': event.timestamp.isoformat()
            })
        
        assistant.safety_monitor.register_alert_callback(safety_alert_handler)
        
        # Video event callback
        def video_event_handler(event: VideoEvent):
            st.session_state.assistant.experiment_state['events'].append({
                'type': f'video_{event.event_type.value}',
                'description': event.description,
                'confidence': event.confidence,
                'timestamp': event.timestamp.isoformat()
            })
        
        assistant.video_system.register_callback(video_event_handler)
        
        # Voice command callback
        def voice_command_handler(result: Dict):
            if result.get('transcript'):
                asyncio.run(assistant.process_voice_command(result['transcript']))
        
        assistant.voice_processor.register_speech_callback(voice_command_handler)
        
        st.session_state.callbacks_registered = True

if __name__ == "__main__":
    main()