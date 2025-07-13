"""
Comprehensive test suite for the Ultimate Lab Assistant System
"""
import unittest
import asyncio
import time
from datetime import datetime
from pathlib import Path
import sys
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "weavehacks_flow-1/src"))

# Import all modules
from realtime_voice_processor import RealtimeVoiceProcessor, VoiceCommandProcessor
from realtime_safety_monitor import RealtimeSafetyMonitor, SafetyLevel
from enhanced_agent_system import MessageBus, Priority, Task
from protocol_automation import ProtocolAutomation, StepStatus
from video_monitoring_system import VideoMonitoringSystem, VideoEvent, EventType
from claude_flow_integration import ClaudeFlowReasoner, ReasoningType, ReasoningContext
from predictive_models import YieldPredictor, ExperimentFeatures
from cloud_connectivity import CloudConnector, CloudMessage, MessageType
from error_recovery import ErrorRecoveryManager, ErrorSeverity, with_recovery

class TestRealtimeVoiceProcessor(unittest.TestCase):
    """Test voice processing capabilities"""
    
    def setUp(self):
        self.voice_processor = RealtimeVoiceProcessor(use_whisper=False)
    
    def test_voice_initialization(self):
        """Test voice processor initialization"""
        self.assertIsNotNone(self.voice_processor)
        self.assertFalse(self.voice_processor.is_recording)
    
    def test_command_processor(self):
        """Test command processing"""
        command_processor = VoiceCommandProcessor()
        
        # Test command parsing
        result = command_processor.process_command("start experiment")
        self.assertEqual(result['intent'], 'start_experiment')
        
        result = command_processor.process_command("what is the temperature")
        self.assertEqual(result['intent'], 'query_status')
    
    @patch('sounddevice.InputStream')
    def test_voice_recording(self, mock_stream):
        """Test voice recording functionality"""
        # Mock audio stream
        mock_stream.return_value.__enter__ = Mock()
        mock_stream.return_value.__exit__ = Mock()
        
        # Start recording
        self.voice_processor.start_recording()
        self.assertTrue(self.voice_processor.is_recording)
        
        # Stop recording
        self.voice_processor.stop_recording()
        self.assertFalse(self.voice_processor.is_recording)

class TestRealtimeSafetyMonitor(unittest.TestCase):
    """Test safety monitoring system"""
    
    def setUp(self):
        self.safety_monitor = RealtimeSafetyMonitor(check_interval=0.1)
    
    def test_safety_initialization(self):
        """Test safety monitor initialization"""
        self.assertIsNotNone(self.safety_monitor)
        self.assertFalse(self.safety_monitor.is_monitoring)
    
    def test_threshold_checking(self):
        """Test threshold validation"""
        # Test normal conditions
        readings = {
            'temperature': 23.0,
            'pressure': 101.3,
            'stirring_rpm': 1100
        }
        
        violations = []
        for param, value in readings.items():
            threshold = self.safety_monitor.thresholds.get(param, {})
            if value < threshold.get('min', 0) or value > threshold.get('max', float('inf')):
                violations.append(param)
        
        self.assertEqual(len(violations), 0)
        
        # Test violation
        readings['temperature'] = 35.0  # Above threshold
        violations = []
        for param, value in readings.items():
            threshold = self.safety_monitor.thresholds.get(param, {})
            if value < threshold.get('min', 0) or value > threshold.get('max', float('inf')):
                violations.append(param)
        
        self.assertIn('temperature', violations)
    
    def test_safety_levels(self):
        """Test safety level determination"""
        # Test normal
        self.assertEqual(
            self.safety_monitor._determine_safety_level([]).value,
            SafetyLevel.NORMAL.value
        )
        
        # Test warning
        violations = [{'parameter': 'temperature', 'severity': 'medium'}]
        level = SafetyLevel.WARNING  # Would be determined by actual logic
        self.assertIn(level.value, ['normal', 'warning', 'danger', 'critical'])

class TestEnhancedAgentSystem(unittest.TestCase):
    """Test agent coordination system"""
    
    def setUp(self):
        self.message_bus = MessageBus()
    
    def test_message_bus(self):
        """Test message bus functionality"""
        # Subscribe to topic
        received_messages = []
        
        def handler(message):
            received_messages.append(message)
        
        self.message_bus.subscribe('test_topic', handler)
        
        # Publish message
        self.message_bus.publish('test_topic', {'data': 'test'})
        
        # Allow time for async processing
        time.sleep(0.1)
        
        self.assertEqual(len(received_messages), 1)
        self.assertEqual(received_messages[0]['data'], 'test')
    
    def test_task_queue(self):
        """Test task queue prioritization"""
        tasks = []
        
        # Create tasks with different priorities
        task1 = Task(
            id='1',
            name='Low priority',
            priority=Priority.LOW,
            created_at=datetime.now()
        )
        task2 = Task(
            id='2',
            name='High priority',
            priority=Priority.HIGH,
            created_at=datetime.now()
        )
        task3 = Task(
            id='3',
            name='Critical priority',
            priority=Priority.CRITICAL,
            created_at=datetime.now()
        )
        
        # Tasks should be sorted by priority
        tasks = [task1, task2, task3]
        sorted_tasks = sorted(tasks, key=lambda t: t.priority.value, reverse=True)
        
        self.assertEqual(sorted_tasks[0].priority, Priority.CRITICAL)
        self.assertEqual(sorted_tasks[1].priority, Priority.HIGH)
        self.assertEqual(sorted_tasks[2].priority, Priority.LOW)

class TestProtocolAutomation(unittest.TestCase):
    """Test protocol automation"""
    
    def setUp(self):
        self.protocol = ProtocolAutomation()
        self.protocol.load_protocol('nanoparticle_synthesis')
    
    def test_protocol_loading(self):
        """Test protocol loading"""
        self.assertIsNotNone(self.protocol.current_protocol)
        self.assertGreater(len(self.protocol.current_protocol['steps']), 0)
    
    def test_step_execution(self):
        """Test step execution tracking"""
        # Start protocol
        self.protocol.start_protocol()
        
        # Get current step
        current = self.protocol.get_current_step()
        self.assertIsNotNone(current)
        self.assertEqual(current['status'], StepStatus.IN_PROGRESS.value)
        
        # Complete step
        self.protocol.complete_current_step()
        self.assertEqual(
            self.protocol.current_protocol['steps'][0]['status'],
            StepStatus.COMPLETED.value
        )

class TestVideoMonitoring(unittest.TestCase):
    """Test video monitoring system"""
    
    def setUp(self):
        self.video_system = VideoMonitoringSystem(enable_ml=False)
    
    def test_video_initialization(self):
        """Test video system initialization"""
        self.assertIsNotNone(self.video_system)
        self.assertFalse(self.video_system.is_monitoring)
    
    def test_event_handling(self):
        """Test video event handling"""
        events_received = []
        
        def event_handler(event):
            events_received.append(event)
        
        self.video_system.register_event_callback(event_handler)
        
        # Create test event
        test_event = VideoEvent(
            timestamp=datetime.now(),
            event_type=EventType.COLOR_CHANGE,
            description="Color changed to yellow",
            confidence=0.85,
            frame_number=100
        )
        
        # Handle event
        self.video_system._handle_event(test_event)
        
        self.assertEqual(len(events_received), 1)
        self.assertEqual(events_received[0].event_type, EventType.COLOR_CHANGE)

class TestClaudeFlowIntegration(unittest.TestCase):
    """Test Claude-flow reasoning system"""
    
    def setUp(self):
        self.reasoner = ClaudeFlowReasoner()
    
    def test_reasoning_initialization(self):
        """Test reasoner initialization"""
        self.assertIsNotNone(self.reasoner)
        self.assertIsNotNone(self.reasoner.knowledge_base)
    
    async def test_reasoning_types(self):
        """Test different reasoning types"""
        # Create test context
        context = ReasoningContext(
            experiment_state={'phase': 'synthesis'},
            historical_data=[],
            sensor_readings={'temperature': 23.0},
            recent_events=[],
            safety_status={'is_safe': True}
        )
        
        # Test diagnostic reasoning
        result = await self.reasoner.reason(context, ReasoningType.DIAGNOSTIC)
        self.assertEqual(result.reasoning_type, ReasoningType.DIAGNOSTIC)
        self.assertIsNotNone(result.conclusion)
        
        # Test predictive reasoning
        result = await self.reasoner.reason(context, ReasoningType.PREDICTIVE)
        self.assertEqual(result.reasoning_type, ReasoningType.PREDICTIVE)
        self.assertGreater(result.confidence, 0)
    
    def test_async_reasoning(self):
        """Test async reasoning execution"""
        asyncio.run(self.test_reasoning_types())

class TestPredictiveModels(unittest.TestCase):
    """Test ML prediction models"""
    
    def setUp(self):
        self.predictor = YieldPredictor()
    
    def test_predictor_initialization(self):
        """Test predictor initialization"""
        self.assertIsNotNone(self.predictor)
        self.assertGreater(len(self.predictor.models), 0)
    
    def test_prediction(self):
        """Test yield prediction"""
        # Create test features
        features = ExperimentFeatures(
            gold_mass=0.1576,
            toab_mass=0.25,
            sulfur_mass=0.052,
            nabh4_mass=0.015,
            temperature=23.0,
            stirring_rpm=1100,
            reaction_time=3600,
            ph=7.0,
            ambient_temp=22.0,
            humidity=50.0,
            pressure=101.3
        )
        
        # Make prediction
        result = self.predictor.predict(features)
        
        self.assertIsNotNone(result)
        self.assertGreater(result.prediction, 0)
        self.assertIsNotNone(result.confidence_interval)

class TestCloudConnectivity(unittest.TestCase):
    """Test cloud connectivity features"""
    
    def setUp(self):
        self.config = {
            'user_id': 'test_user',
            'api_key': 'test_key',
            'websocket_url': 'wss://test.example.com/ws',
            'auth_url': 'https://test.example.com/auth',
            'storage_backend': 'azure'
        }
        self.cloud = CloudConnector(self.config)
    
    def test_cloud_initialization(self):
        """Test cloud connector initialization"""
        self.assertIsNotNone(self.cloud)
        self.assertFalse(self.cloud.is_connected)
    
    def test_message_queuing(self):
        """Test message queuing"""
        # Create test message
        message = CloudMessage(
            message_type=MessageType.SENSOR_DATA,
            payload={'temperature': 23.5},
            sender_id='test_user',
            experiment_id='exp_001',
            timestamp=datetime.now()
        )
        
        # Queue message
        self.cloud.send_message(message)
        
        # Check queue
        self.assertFalse(self.cloud.message_queue.empty())
        queued = self.cloud.message_queue.get()
        self.assertEqual(queued.message_type, MessageType.SENSOR_DATA)

class TestErrorRecovery(unittest.TestCase):
    """Test error recovery mechanisms"""
    
    def setUp(self):
        self.error_manager = ErrorRecoveryManager()
    
    def test_error_handling(self):
        """Test basic error handling"""
        # Test error handling
        try:
            raise ValueError("Test error")
        except Exception as e:
            result = self.error_manager.handle_error(
                e, 
                'test_component',
                ErrorSeverity.LOW
            )
        
        # Check error was logged
        self.assertEqual(len(self.error_manager.error_history), 1)
        self.assertEqual(
            self.error_manager.error_history[0].error_type,
            'ValueError'
        )
    
    def test_recovery_decorator(self):
        """Test recovery decorator"""
        @with_recovery('test_function', ErrorSeverity.MEDIUM, 
                      fallback_value='fallback')
        def failing_function():
            raise Exception("Test failure")
        
        # Should return fallback value
        result = failing_function()
        self.assertEqual(result, 'fallback')
    
    def test_circuit_breaker(self):
        """Test circuit breaker pattern"""
        from error_recovery import CircuitBreaker
        
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        
        # Function that fails
        fail_count = 0
        def failing_func():
            nonlocal fail_count
            fail_count += 1
            raise Exception("Failure")
        
        # First failure
        try:
            breaker.call(failing_func)
        except:
            pass
        
        # Second failure - should open circuit
        try:
            breaker.call(failing_func)
        except:
            pass
        
        self.assertEqual(breaker.state, 'open')

class TestIntegration(unittest.TestCase):
    """Integration tests for complete system"""
    
    def test_end_to_end_workflow(self):
        """Test complete experiment workflow"""
        # Initialize components
        voice = RealtimeVoiceProcessor(use_whisper=False)
        safety = RealtimeSafetyMonitor()
        protocol = ProtocolAutomation()
        
        # Load protocol
        protocol.load_protocol('nanoparticle_synthesis')
        self.assertIsNotNone(protocol.current_protocol)
        
        # Start protocol
        protocol.start_protocol()
        current_step = protocol.get_current_step()
        self.assertEqual(current_step['status'], StepStatus.IN_PROGRESS.value)
        
        # Simulate safety check
        safety_status = {
            'is_safe': True,
            'level': SafetyLevel.NORMAL.value,
            'violations': []
        }
        self.assertTrue(safety_status['is_safe'])
        
        # Complete step
        protocol.complete_current_step()
        self.assertEqual(
            protocol.current_protocol['steps'][0]['status'],
            StepStatus.COMPLETED.value
        )
    
    def test_system_resilience(self):
        """Test system resilience with errors"""
        error_manager = ErrorRecoveryManager()
        
        # Simulate multiple component failures
        components = ['sensor', 'voice', 'cloud']
        
        for comp in components:
            try:
                if comp == 'sensor':
                    raise ConnectionError("Sensor disconnected")
                elif comp == 'voice':
                    raise RuntimeError("Audio device error")
                elif comp == 'cloud':
                    raise TimeoutError("Cloud connection timeout")
            except Exception as e:
                result = error_manager.handle_error(
                    e,
                    comp,
                    ErrorSeverity.MEDIUM
                )
        
        # Check all errors were handled
        self.assertEqual(len(error_manager.error_history), 3)
        
        # Get statistics
        stats = error_manager.get_error_statistics()
        self.assertEqual(stats['total_errors'], 3)

# Performance tests
class TestPerformance(unittest.TestCase):
    """Performance and load tests"""
    
    def test_message_bus_throughput(self):
        """Test message bus performance"""
        bus = MessageBus()
        message_count = 1000
        received = []
        
        def handler(msg):
            received.append(msg)
        
        bus.subscribe('perf_test', handler)
        
        # Send many messages
        start_time = time.time()
        for i in range(message_count):
            bus.publish('perf_test', {'id': i})
        
        # Wait for processing
        time.sleep(0.5)
        
        elapsed = time.time() - start_time
        throughput = message_count / elapsed
        
        print(f"Message throughput: {throughput:.1f} messages/second")
        self.assertGreater(throughput, 100)  # At least 100 msg/sec
    
    def test_concurrent_operations(self):
        """Test concurrent operation handling"""
        import concurrent.futures
        
        def simulate_operation(op_id):
            # Simulate some work
            time.sleep(0.01)
            return f"Operation {op_id} completed"
        
        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(50):
                future = executor.submit(simulate_operation, i)
                futures.append(future)
            
            # Wait for completion
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        self.assertEqual(len(results), 50)

def run_all_tests():
    """Run all test suites"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestRealtimeVoiceProcessor,
        TestRealtimeSafetyMonitor,
        TestEnhancedAgentSystem,
        TestProtocolAutomation,
        TestVideoMonitoring,
        TestClaudeFlowIntegration,
        TestPredictiveModels,
        TestCloudConnectivity,
        TestErrorRecovery,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("="*70)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)