"""
Enhanced agent coordination system with message queue and parallel execution
"""
import asyncio
import threading
import queue
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import weave
from concurrent.futures import ThreadPoolExecutor, Future
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class AgentStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"

@dataclass
class Message:
    id: str
    sender: str
    recipient: str
    content: Any
    priority: Priority
    timestamp: datetime = field(default_factory=datetime.now)
    requires_response: bool = False
    correlation_id: Optional[str] = None

@dataclass
class Task:
    id: str
    name: str
    agent_id: str
    priority: Priority
    payload: Dict[str, Any]
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

class MessageBus:
    """Central message bus for agent communication"""
    
    def __init__(self):
        self.queues: Dict[str, queue.PriorityQueue] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history = []
        self.lock = threading.Lock()
        weave.init('agent-message-bus')
    
    def register_agent(self, agent_id: str):
        """Register an agent with the message bus"""
        with self.lock:
            if agent_id not in self.queues:
                self.queues[agent_id] = queue.PriorityQueue()
                logger.info(f"Registered agent: {agent_id}")
    
    @weave.op()
    def send_message(self, message: Message):
        """Send a message to an agent"""
        with self.lock:
            if message.recipient in self.queues:
                # Priority queue uses negative priority for correct ordering
                priority_value = -message.priority.value
                self.queues[message.recipient].put((priority_value, message))
                self.message_history.append(message)
                
                # Trigger subscribers
                if message.recipient in self.subscribers:
                    for callback in self.subscribers[message.recipient]:
                        threading.Thread(target=callback, args=(message,)).start()
                
                # Log to W&B
                weave.log({
                    'message_sent': {
                        'id': message.id,
                        'sender': message.sender,
                        'recipient': message.recipient,
                        'priority': message.priority.name,
                        'timestamp': message.timestamp.isoformat()
                    }
                })
                
                return True
            else:
                logger.warning(f"Recipient {message.recipient} not registered")
                return False
    
    def receive_message(self, agent_id: str, timeout: Optional[float] = None) -> Optional[Message]:
        """Receive a message for an agent"""
        if agent_id in self.queues:
            try:
                _, message = self.queues[agent_id].get(timeout=timeout)
                return message
            except queue.Empty:
                return None
        return None
    
    def subscribe(self, agent_id: str, callback: Callable):
        """Subscribe to messages for an agent"""
        with self.lock:
            if agent_id not in self.subscribers:
                self.subscribers[agent_id] = []
            self.subscribers[agent_id].append(callback)

class EnhancedAgent:
    """Base class for enhanced agents with coordination capabilities"""
    
    def __init__(self, agent_id: str, agent_type: str, message_bus: MessageBus):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.message_bus = message_bus
        self.status = AgentStatus.IDLE
        self.task_queue = queue.PriorityQueue()
        self.current_task: Optional[Task] = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.running = False
        self._register()
        
        # Agent-specific state
        self.state = {}
        self.capabilities = []
        
        # Initialize W&B
        weave.init(f'agent-{agent_type}')
    
    def _register(self):
        """Register with message bus"""
        self.message_bus.register_agent(self.agent_id)
        self.message_bus.subscribe(self.agent_id, self._handle_message)
        logger.info(f"Agent {self.agent_id} registered")
    
    @weave.op()
    def _handle_message(self, message: Message):
        """Handle incoming messages"""
        logger.info(f"{self.agent_id} received message from {message.sender}")
        
        # Convert message to task if it contains a task payload
        if isinstance(message.content, dict) and 'task' in message.content:
            task = Task(
                id=f"task_{message.id}",
                name=message.content['task'],
                agent_id=self.agent_id,
                priority=message.priority,
                payload=message.content.get('payload', {})
            )
            self.add_task(task)
        
        # Handle coordination messages
        elif isinstance(message.content, dict) and 'command' in message.content:
            self._handle_command(message)
        
        # If response required, send acknowledgment
        if message.requires_response:
            response = Message(
                id=f"resp_{message.id}",
                sender=self.agent_id,
                recipient=message.sender,
                content={'status': 'received', 'original_id': message.id},
                priority=Priority.MEDIUM,
                correlation_id=message.id
            )
            self.message_bus.send_message(response)
    
    def _handle_command(self, message: Message):
        """Handle coordination commands"""
        command = message.content.get('command')
        
        if command == 'status':
            self.send_status_update(message.sender)
        elif command == 'pause':
            self.pause()
        elif command == 'resume':
            self.resume()
        elif command == 'get_capabilities':
            self.send_capabilities(message.sender)
    
    def add_task(self, task: Task):
        """Add a task to the agent's queue"""
        priority_value = -task.priority.value
        self.task_queue.put((priority_value, task))
        
        weave.log({
            'task_added': {
                'agent_id': self.agent_id,
                'task_id': task.id,
                'task_name': task.name,
                'priority': task.priority.name
            }
        })
    
    @weave.op()
    def process_task(self, task: Task) -> Any:
        """Process a task - override in subclasses"""
        raise NotImplementedError("Subclasses must implement process_task")
    
    def _task_worker(self):
        """Worker thread for processing tasks"""
        while self.running:
            try:
                # Get next task with timeout
                _, task = self.task_queue.get(timeout=1.0)
                
                # Update status
                self.status = AgentStatus.BUSY
                self.current_task = task
                task.status = "in_progress"
                
                # Log task start
                weave.log({
                    'task_started': {
                        'agent_id': self.agent_id,
                        'task_id': task.id,
                        'task_name': task.name
                    }
                })
                
                # Process task
                try:
                    result = self.process_task(task)
                    task.result = result
                    task.status = "completed"
                    task.completed_at = datetime.now()
                    
                    # Send completion message
                    self._send_task_completion(task)
                    
                except Exception as e:
                    task.status = "failed"
                    task.error = str(e)
                    logger.error(f"Task {task.id} failed: {e}")
                    
                    # Send failure message
                    self._send_task_failure(task)
                
                finally:
                    self.current_task = None
                    self.status = AgentStatus.IDLE
                    
                    # Log task completion
                    weave.log({
                        'task_completed': {
                            'agent_id': self.agent_id,
                            'task_id': task.id,
                            'status': task.status,
                            'duration': (task.completed_at - task.created_at).total_seconds() 
                                      if task.completed_at else None
                        }
                    })
            
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in task worker: {e}")
                self.status = AgentStatus.ERROR
    
    def _send_task_completion(self, task: Task):
        """Send task completion notification"""
        message = Message(
            id=f"completion_{task.id}",
            sender=self.agent_id,
            recipient="coordinator",
            content={
                'type': 'task_completion',
                'task_id': task.id,
                'result': task.result
            },
            priority=Priority.MEDIUM
        )
        self.message_bus.send_message(message)
    
    def _send_task_failure(self, task: Task):
        """Send task failure notification"""
        message = Message(
            id=f"failure_{task.id}",
            sender=self.agent_id,
            recipient="coordinator",
            content={
                'type': 'task_failure',
                'task_id': task.id,
                'error': task.error
            },
            priority=Priority.HIGH
        )
        self.message_bus.send_message(message)
    
    def send_status_update(self, recipient: str = "coordinator"):
        """Send status update"""
        message = Message(
            id=f"status_{self.agent_id}_{int(time.time())}",
            sender=self.agent_id,
            recipient=recipient,
            content={
                'type': 'status_update',
                'status': self.status.value,
                'current_task': self.current_task.id if self.current_task else None,
                'queue_size': self.task_queue.qsize()
            },
            priority=Priority.LOW
        )
        self.message_bus.send_message(message)
    
    def send_capabilities(self, recipient: str):
        """Send agent capabilities"""
        message = Message(
            id=f"capabilities_{self.agent_id}_{int(time.time())}",
            sender=self.agent_id,
            recipient=recipient,
            content={
                'type': 'capabilities',
                'agent_type': self.agent_type,
                'capabilities': self.capabilities
            },
            priority=Priority.MEDIUM
        )
        self.message_bus.send_message(message)
    
    def start(self):
        """Start the agent"""
        self.running = True
        self.status = AgentStatus.IDLE
        
        # Start worker thread
        worker_thread = threading.Thread(target=self._task_worker)
        worker_thread.daemon = True
        worker_thread.start()
        
        logger.info(f"Agent {self.agent_id} started")
    
    def stop(self):
        """Stop the agent"""
        self.running = False
        self.status = AgentStatus.OFFLINE
        self.executor.shutdown(wait=True)
        logger.info(f"Agent {self.agent_id} stopped")
    
    def pause(self):
        """Pause task processing"""
        self.status = AgentStatus.IDLE
        logger.info(f"Agent {self.agent_id} paused")
    
    def resume(self):
        """Resume task processing"""
        if self.running:
            self.status = AgentStatus.IDLE
            logger.info(f"Agent {self.agent_id} resumed")

# Specific agent implementations

class DataCollectionAgent(EnhancedAgent):
    """Enhanced data collection agent"""
    
    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, "data_collection", message_bus)
        self.capabilities = [
            "record_mass",
            "record_volume",
            "record_temperature",
            "record_observation",
            "validate_data"
        ]
        self.collected_data = {}
    
    @weave.op()
    def process_task(self, task: Task) -> Any:
        """Process data collection tasks"""
        task_name = task.name
        payload = task.payload
        
        if task_name == "record_mass":
            return self.record_mass(
                payload.get('substance'),
                payload.get('value')
            )
        elif task_name == "record_volume":
            return self.record_volume(
                payload.get('liquid'),
                payload.get('value')
            )
        elif task_name == "record_observation":
            return self.record_observation(
                payload.get('observation'),
                payload.get('step')
            )
        else:
            raise ValueError(f"Unknown task: {task_name}")
    
    def record_mass(self, substance: str, value: float) -> Dict[str, Any]:
        """Record mass measurement"""
        self.collected_data[f"mass_{substance}"] = {
            'value': value,
            'unit': 'g',
            'timestamp': datetime.now().isoformat()
        }
        
        weave.log({
            'data_recorded': {
                'type': 'mass',
                'substance': substance,
                'value': value
            }
        })
        
        return {'status': 'recorded', 'substance': substance, 'value': value}
    
    def record_volume(self, liquid: str, value: float) -> Dict[str, Any]:
        """Record volume measurement"""
        self.collected_data[f"volume_{liquid}"] = {
            'value': value,
            'unit': 'mL',
            'timestamp': datetime.now().isoformat()
        }
        
        return {'status': 'recorded', 'liquid': liquid, 'value': value}
    
    def record_observation(self, observation: str, step: int) -> Dict[str, Any]:
        """Record qualitative observation"""
        obs_key = f"observation_step_{step}"
        if obs_key not in self.collected_data:
            self.collected_data[obs_key] = []
        
        self.collected_data[obs_key].append({
            'text': observation,
            'timestamp': datetime.now().isoformat()
        })
        
        return {'status': 'recorded', 'step': step}

class SafetyMonitoringAgent(EnhancedAgent):
    """Enhanced safety monitoring agent"""
    
    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, "safety_monitoring", message_bus)
        self.capabilities = [
            "monitor_parameters",
            "check_safety",
            "emergency_shutdown",
            "adjust_parameters"
        ]
        self.safety_thresholds = {
            'temperature': {'min': 15, 'max': 35},
            'pressure': {'min': 95, 'max': 110}
        }
        self.current_readings = {}
    
    @weave.op()
    def process_task(self, task: Task) -> Any:
        """Process safety monitoring tasks"""
        task_name = task.name
        payload = task.payload
        
        if task_name == "monitor_parameters":
            return self.monitor_parameters(payload.get('parameters', {}))
        elif task_name == "check_safety":
            return self.check_safety()
        elif task_name == "emergency_shutdown":
            return self.emergency_shutdown(payload.get('reason'))
        else:
            raise ValueError(f"Unknown task: {task_name}")
    
    def monitor_parameters(self, parameters: Dict[str, float]) -> Dict[str, Any]:
        """Monitor safety parameters"""
        self.current_readings.update(parameters)
        alerts = []
        
        for param, value in parameters.items():
            if param in self.safety_thresholds:
                threshold = self.safety_thresholds[param]
                if value < threshold['min'] or value > threshold['max']:
                    alerts.append({
                        'parameter': param,
                        'value': value,
                        'threshold': threshold,
                        'severity': 'critical' if abs(value - threshold['min']) > 10 
                                              or abs(value - threshold['max']) > 10 
                                              else 'warning'
                    })
        
        if alerts:
            # Send alert message
            alert_message = Message(
                id=f"safety_alert_{int(time.time())}",
                sender=self.agent_id,
                recipient="coordinator",
                content={
                    'type': 'safety_alert',
                    'alerts': alerts
                },
                priority=Priority.CRITICAL
            )
            self.message_bus.send_message(alert_message)
        
        return {'status': 'monitored', 'alerts': alerts}
    
    def check_safety(self) -> Dict[str, Any]:
        """Check overall safety status"""
        is_safe = True
        issues = []
        
        for param, value in self.current_readings.items():
            if param in self.safety_thresholds:
                threshold = self.safety_thresholds[param]
                if value < threshold['min'] or value > threshold['max']:
                    is_safe = False
                    issues.append(f"{param}: {value}")
        
        return {'is_safe': is_safe, 'issues': issues}
    
    def emergency_shutdown(self, reason: str) -> Dict[str, Any]:
        """Trigger emergency shutdown"""
        # Send shutdown command to all agents
        shutdown_message = Message(
            id=f"emergency_shutdown_{int(time.time())}",
            sender=self.agent_id,
            recipient="all",
            content={
                'command': 'emergency_shutdown',
                'reason': reason
            },
            priority=Priority.CRITICAL
        )
        self.message_bus.send_message(shutdown_message)
        
        weave.log({
            'emergency_shutdown': {
                'triggered_by': self.agent_id,
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }
        })
        
        return {'status': 'shutdown_triggered', 'reason': reason}

class CoordinatorAgent(EnhancedAgent):
    """Coordinator agent for orchestrating other agents"""
    
    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, "coordinator", message_bus)
        self.capabilities = [
            "orchestrate_experiment",
            "distribute_tasks",
            "monitor_agents",
            "handle_alerts"
        ]
        self.agent_registry = {}
        self.experiment_state = {
            'current_step': 0,
            'status': 'not_started',
            'agents': {}
        }
    
    def register_agent(self, agent: EnhancedAgent):
        """Register an agent with the coordinator"""
        self.agent_registry[agent.agent_id] = agent
        self.experiment_state['agents'][agent.agent_id] = {
            'type': agent.agent_type,
            'status': agent.status.value
        }
    
    @weave.op()
    def process_task(self, task: Task) -> Any:
        """Process coordination tasks"""
        task_name = task.name
        payload = task.payload
        
        if task_name == "orchestrate_experiment":
            return self.orchestrate_experiment(payload.get('protocol'))
        elif task_name == "distribute_tasks":
            return self.distribute_tasks(payload.get('tasks', []))
        else:
            raise ValueError(f"Unknown task: {task_name}")
    
    def orchestrate_experiment(self, protocol: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Orchestrate an experiment protocol"""
        self.experiment_state['status'] = 'in_progress'
        results = []
        
        for step in protocol:
            self.experiment_state['current_step'] = step.get('step_number', 0)
            
            # Distribute tasks for this step
            step_tasks = step.get('tasks', [])
            task_results = self.distribute_tasks(step_tasks)
            
            results.append({
                'step': step.get('step_number'),
                'results': task_results
            })
            
            # Check for safety after each step
            safety_check = self.check_safety_status()
            if not safety_check['is_safe']:
                self.experiment_state['status'] = 'halted'
                return {
                    'status': 'halted',
                    'reason': 'safety_violation',
                    'completed_steps': results
                }
        
        self.experiment_state['status'] = 'completed'
        return {'status': 'completed', 'results': results}
    
    def distribute_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Distribute tasks to appropriate agents"""
        futures = []
        
        for task_spec in tasks:
            agent_type = task_spec.get('agent_type')
            
            # Find available agent of the required type
            agent_id = self.find_available_agent(agent_type)
            
            if agent_id:
                # Create task message
                task_message = Message(
                    id=f"task_{int(time.time())}_{agent_type}",
                    sender=self.agent_id,
                    recipient=agent_id,
                    content={
                        'task': task_spec.get('task_name'),
                        'payload': task_spec.get('payload', {})
                    },
                    priority=Priority(task_spec.get('priority', Priority.MEDIUM.value))
                )
                
                # Send task
                self.message_bus.send_message(task_message)
                
                futures.append({
                    'task_id': task_message.id,
                    'agent_id': agent_id,
                    'task_name': task_spec.get('task_name')
                })
        
        return futures
    
    def find_available_agent(self, agent_type: str) -> Optional[str]:
        """Find an available agent of the specified type"""
        for agent_id, agent_info in self.experiment_state['agents'].items():
            if agent_info['type'] == agent_type and agent_info['status'] == AgentStatus.IDLE.value:
                return agent_id
        return None
    
    def check_safety_status(self) -> Dict[str, Any]:
        """Check safety status from safety agent"""
        # Find safety monitoring agent
        safety_agent_id = self.find_available_agent("safety_monitoring")
        
        if safety_agent_id:
            # Request safety check
            message = Message(
                id=f"safety_check_{int(time.time())}",
                sender=self.agent_id,
                recipient=safety_agent_id,
                content={
                    'task': 'check_safety',
                    'payload': {}
                },
                priority=Priority.HIGH,
                requires_response=True
            )
            self.message_bus.send_message(message)
            
            # Wait for response (simplified - in real system would be async)
            time.sleep(0.1)
            
            # Check for safety alerts
            response = self.message_bus.receive_message(self.agent_id, timeout=1.0)
            if response and response.content.get('type') == 'safety_alert':
                return {'is_safe': False, 'alerts': response.content.get('alerts', [])}
        
        return {'is_safe': True}

# Example usage
if __name__ == "__main__":
    # Create message bus
    bus = MessageBus()
    
    # Create agents
    data_agent = DataCollectionAgent("data_agent_1", bus)
    safety_agent = SafetyMonitoringAgent("safety_agent_1", bus)
    coordinator = CoordinatorAgent("coordinator", bus)
    
    # Register agents with coordinator
    coordinator.register_agent(data_agent)
    coordinator.register_agent(safety_agent)
    
    # Start all agents
    data_agent.start()
    safety_agent.start()
    coordinator.start()
    
    # Example experiment protocol
    protocol = [
        {
            'step_number': 1,
            'tasks': [
                {
                    'agent_type': 'data_collection',
                    'task_name': 'record_mass',
                    'payload': {'substance': 'gold', 'value': 0.1576},
                    'priority': Priority.HIGH.value
                },
                {
                    'agent_type': 'safety_monitoring',
                    'task_name': 'monitor_parameters',
                    'payload': {'parameters': {'temperature': 25.0, 'pressure': 101.3}},
                    'priority': Priority.HIGH.value
                }
            ]
        }
    ]
    
    # Start experiment
    coordinator.add_task(Task(
        id="exp_001",
        name="orchestrate_experiment",
        agent_id="coordinator",
        priority=Priority.HIGH,
        payload={'protocol': protocol}
    ))
    
    # Run for a bit
    try:
        time.sleep(5)
        
        # Check status
        print("\nAgent Status:")
        for agent_id in ["data_agent_1", "safety_agent_1", "coordinator"]:
            coordinator.agent_registry[agent_id].send_status_update()
        
        time.sleep(1)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        # Stop all agents
        data_agent.stop()
        safety_agent.stop()
        coordinator.stop()