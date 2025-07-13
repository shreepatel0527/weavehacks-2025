"""
Comprehensive error recovery and resilience mechanisms
"""
import asyncio
import functools
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, TypeVar, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import traceback
import json
import logging
from pathlib import Path
import weave
import pickle
import queue

T = TypeVar('T')

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryStrategy(Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"
    ROLLBACK = "rollback"
    MANUAL_INTERVENTION = "manual_intervention"

@dataclass
class ErrorContext:
    error_type: str
    error_message: str
    traceback: str
    timestamp: datetime
    component: str
    severity: ErrorSeverity
    recovery_attempts: int = 0
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecoveryAction:
    strategy: RecoveryStrategy
    action: Callable
    max_attempts: int = 3
    backoff_factor: float = 2.0
    timeout: float = 30.0
    fallback_value: Any = None

class ErrorRecoveryManager:
    """Comprehensive error recovery system"""
    
    def __init__(self, state_manager: Optional['StateManager'] = None):
        self.state_manager = state_manager or StateManager()
        self.error_history = []
        self.recovery_actions = {}
        self.circuit_breakers = {}
        self.error_handlers = {}
        
        # Recovery configuration
        self.max_error_history = 1000
        self.error_log_path = Path("error_logs")
        self.error_log_path.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize recovery strategies
        self._init_recovery_strategies()
        
        # Initialize W&B
        weave.init('error-recovery')
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger('error_recovery')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(
            self.error_log_path / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
        )
        fh.setLevel(logging.ERROR)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _init_recovery_strategies(self):
        """Initialize default recovery strategies"""
        # Voice processing errors
        self.register_recovery(
            'VoiceProcessingError',
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                action=self._retry_with_backoff,
                max_attempts=3,
                fallback_value={'text': '', 'confidence': 0.0}
            )
        )
        
        # Sensor reading errors
        self.register_recovery(
            'SensorReadError',
            RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                action=self._use_cached_value,
                fallback_value={'temperature': 23.0, 'pressure': 101.3}
            )
        )
        
        # API errors
        self.register_recovery(
            'APIError',
            RecoveryAction(
                strategy=RecoveryStrategy.CIRCUIT_BREAKER,
                action=self._circuit_breaker_handler,
                max_attempts=5,
                timeout=60.0
            )
        )
        
        # Database errors
        self.register_recovery(
            'DatabaseError',
            RecoveryAction(
                strategy=RecoveryStrategy.ROLLBACK,
                action=self._rollback_transaction
            )
        )
    
    def register_recovery(self, error_type: str, action: RecoveryAction):
        """Register recovery action for error type"""
        self.recovery_actions[error_type] = action
    
    def register_handler(self, error_type: str, handler: Callable):
        """Register custom error handler"""
        self.error_handlers[error_type] = handler
    
    @weave.op()
    def handle_error(self, error: Exception, component: str, 
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: Optional[Dict[str, Any]] = None) -> Any:
        """Main error handling entry point"""
        
        # Create error context
        error_context = ErrorContext(
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
            timestamp=datetime.now(),
            component=component,
            severity=severity,
            additional_data=context or {}
        )
        
        # Log error
        self._log_error(error_context)
        
        # Store in history
        self._store_error(error_context)
        
        # Check for custom handler
        if error_context.error_type in self.error_handlers:
            try:
                return self.error_handlers[error_context.error_type](
                    error, error_context
                )
            except Exception as handler_error:
                self.logger.error(f"Error handler failed: {handler_error}")
        
        # Apply recovery strategy
        recovery_action = self.recovery_actions.get(
            error_context.error_type,
            self._get_default_recovery()
        )
        
        return self._execute_recovery(error_context, recovery_action)
    
    def _log_error(self, context: ErrorContext):
        """Log error with context"""
        log_data = {
            'error': {
                'type': context.error_type,
                'message': context.error_message,
                'component': context.component,
                'severity': context.severity.value,
                'timestamp': context.timestamp.isoformat()
            }
        }
        
        # Log to W&B
        weave.log(log_data)
        
        # Log to file
        if context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.error(json.dumps(log_data))
        else:
            self.logger.warning(json.dumps(log_data))
    
    def _store_error(self, context: ErrorContext):
        """Store error in history"""
        self.error_history.append(context)
        
        # Limit history size
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)
        
        # Persist critical errors
        if context.severity == ErrorSeverity.CRITICAL:
            self._persist_error(context)
    
    def _persist_error(self, context: ErrorContext):
        """Persist error to disk"""
        filename = f"error_{context.timestamp.strftime('%Y%m%d_%H%M%S')}_{context.error_type}.json"
        filepath = self.error_log_path / filename
        
        with open(filepath, 'w') as f:
            json.dump({
                'error_type': context.error_type,
                'message': context.error_message,
                'traceback': context.traceback,
                'component': context.component,
                'severity': context.severity.value,
                'timestamp': context.timestamp.isoformat(),
                'additional_data': context.additional_data
            }, f, indent=2)
    
    def _execute_recovery(self, context: ErrorContext, 
                         action: RecoveryAction) -> Any:
        """Execute recovery action"""
        if action.strategy == RecoveryStrategy.RETRY:
            return self._retry_with_backoff(context, action)
        elif action.strategy == RecoveryStrategy.FALLBACK:
            return self._use_fallback(context, action)
        elif action.strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return self._circuit_breaker_handler(context, action)
        elif action.strategy == RecoveryStrategy.ROLLBACK:
            return self._rollback_transaction(context, action)
        elif action.strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return self._graceful_degradation(context, action)
        else:
            return self._manual_intervention(context, action)
    
    def _retry_with_backoff(self, context: ErrorContext, 
                           action: RecoveryAction) -> Any:
        """Retry with exponential backoff"""
        for attempt in range(action.max_attempts):
            try:
                # Update attempt count
                context.recovery_attempts = attempt + 1
                
                # Wait with backoff
                if attempt > 0:
                    wait_time = action.backoff_factor ** attempt
                    time.sleep(min(wait_time, action.timeout))
                
                # Retry the action
                if callable(action.action):
                    return action.action()
                else:
                    return action.fallback_value
                    
            except Exception as e:
                if attempt == action.max_attempts - 1:
                    # Final attempt failed
                    return action.fallback_value
                continue
    
    def _use_fallback(self, context: ErrorContext, 
                     action: RecoveryAction) -> Any:
        """Use fallback value"""
        # Try to get cached value first
        cached = self._get_cached_value(context.component)
        if cached is not None:
            return cached
        
        return action.fallback_value
    
    def _use_cached_value(self, context: ErrorContext, 
                         action: RecoveryAction) -> Any:
        """Use cached value from state manager"""
        return self.state_manager.get_cached_value(
            context.component,
            action.fallback_value
        )
    
    def _circuit_breaker_handler(self, context: ErrorContext, 
                               action: RecoveryAction) -> Any:
        """Handle with circuit breaker pattern"""
        breaker_key = f"{context.component}:{context.error_type}"
        
        if breaker_key not in self.circuit_breakers:
            self.circuit_breakers[breaker_key] = CircuitBreaker(
                failure_threshold=action.max_attempts,
                recovery_timeout=action.timeout
            )
        
        breaker = self.circuit_breakers[breaker_key]
        
        try:
            return breaker.call(action.action)
        except Exception:
            return action.fallback_value
    
    def _rollback_transaction(self, context: ErrorContext, 
                            action: RecoveryAction) -> Any:
        """Rollback to previous state"""
        return self.state_manager.rollback(context.component)
    
    def _graceful_degradation(self, context: ErrorContext, 
                            action: RecoveryAction) -> Any:
        """Gracefully degrade functionality"""
        # Disable non-critical features
        degraded_features = self.state_manager.enable_degraded_mode(
            context.component
        )
        
        self.logger.warning(
            f"Graceful degradation enabled for {context.component}: "
            f"Disabled features: {degraded_features}"
        )
        
        return action.fallback_value
    
    def _manual_intervention(self, context: ErrorContext, 
                           action: RecoveryAction) -> Any:
        """Request manual intervention"""
        # Log critical error requiring manual intervention
        self.logger.critical(
            f"Manual intervention required for {context.component}: "
            f"{context.error_message}"
        )
        
        # Send alert (in production would integrate with alerting system)
        self._send_alert(context)
        
        return action.fallback_value
    
    def _send_alert(self, context: ErrorContext):
        """Send alert for manual intervention"""
        alert = {
            'type': 'manual_intervention_required',
            'component': context.component,
            'error': context.error_type,
            'message': context.error_message,
            'severity': context.severity.value,
            'timestamp': context.timestamp.isoformat()
        }
        
        # In production, would send to alerting service
        print(f"ALERT: {json.dumps(alert, indent=2)}")
    
    def _get_cached_value(self, component: str) -> Optional[Any]:
        """Get cached value for component"""
        return self.state_manager.get_cached_value(component)
    
    def _get_default_recovery(self) -> RecoveryAction:
        """Get default recovery action"""
        return RecoveryAction(
            strategy=RecoveryStrategy.FALLBACK,
            action=lambda: None,
            fallback_value=None
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        if not self.error_history:
            return {}
        
        # Group by error type
        error_counts = {}
        severity_counts = {s.value: 0 for s in ErrorSeverity}
        component_counts = {}
        
        for error in self.error_history:
            # Count by type
            error_counts[error.error_type] = error_counts.get(
                error.error_type, 0
            ) + 1
            
            # Count by severity
            severity_counts[error.severity.value] += 1
            
            # Count by component
            component_counts[error.component] = component_counts.get(
                error.component, 0
            ) + 1
        
        # Calculate recovery success rate
        total_recoveries = sum(e.recovery_attempts for e in self.error_history)
        successful_recoveries = len([e for e in self.error_history 
                                   if e.recovery_attempts > 0])
        
        recovery_rate = (successful_recoveries / len(self.error_history) * 100 
                        if self.error_history else 0)
        
        return {
            'total_errors': len(self.error_history),
            'error_types': error_counts,
            'severity_distribution': severity_counts,
            'component_errors': component_counts,
            'recovery_success_rate': recovery_rate,
            'total_recovery_attempts': total_recoveries
        }

class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(self, failure_threshold: int = 5, 
                 recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half_open
    
    def call(self, func: Callable) -> Any:
        """Call function with circuit breaker protection"""
        if self.state == 'open':
            if self._should_attempt_reset():
                self.state = 'half_open'
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if should attempt reset"""
        return (self.last_failure_time and 
                datetime.now() - self.last_failure_time > 
                timedelta(seconds=self.recovery_timeout))
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = 'closed'
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'

class StateManager:
    """Manages application state for recovery"""
    
    def __init__(self):
        self.state_history = {}
        self.cache = {}
        self.degraded_components = set()
        self.checkpoints = {}
        self.max_history = 10
    
    def save_state(self, component: str, state: Any):
        """Save component state"""
        if component not in self.state_history:
            self.state_history[component] = []
        
        self.state_history[component].append({
            'state': state,
            'timestamp': datetime.now()
        })
        
        # Limit history
        if len(self.state_history[component]) > self.max_history:
            self.state_history[component].pop(0)
        
        # Update cache
        self.cache[component] = state
    
    def rollback(self, component: str, steps: int = 1) -> Any:
        """Rollback to previous state"""
        if component not in self.state_history:
            return None
        
        history = self.state_history[component]
        if len(history) > steps:
            return history[-steps-1]['state']
        
        return None
    
    def get_cached_value(self, component: str, default: Any = None) -> Any:
        """Get cached value for component"""
        return self.cache.get(component, default)
    
    def enable_degraded_mode(self, component: str) -> List[str]:
        """Enable degraded mode for component"""
        self.degraded_components.add(component)
        
        # Define features to disable per component
        disabled_features = {
            'voice_processing': ['advanced_nlp', 'multi_language'],
            'video_monitoring': ['ml_detection', 'high_resolution'],
            'cloud_sync': ['real_time_sync', 'large_uploads'],
            'agent_coordination': ['parallel_execution', 'advanced_reasoning']
        }
        
        return disabled_features.get(component, [])
    
    def is_degraded(self, component: str) -> bool:
        """Check if component is in degraded mode"""
        return component in self.degraded_components
    
    def create_checkpoint(self, name: str):
        """Create system checkpoint"""
        self.checkpoints[name] = {
            'timestamp': datetime.now(),
            'state': dict(self.cache),
            'degraded': set(self.degraded_components)
        }
    
    def restore_checkpoint(self, name: str) -> bool:
        """Restore from checkpoint"""
        if name not in self.checkpoints:
            return False
        
        checkpoint = self.checkpoints[name]
        self.cache = dict(checkpoint['state'])
        self.degraded_components = set(checkpoint['degraded'])
        
        return True

# Decorator for automatic error recovery
def with_recovery(component: str, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 fallback_value: Any = None):
    """Decorator for automatic error recovery"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get or create error recovery manager
                if not hasattr(wrapper, '_error_manager'):
                    wrapper._error_manager = ErrorRecoveryManager()
                
                # Handle error
                return wrapper._error_manager.handle_error(
                    e, 
                    component, 
                    severity,
                    context={'args': args, 'kwargs': kwargs}
                ) or fallback_value
        
        return wrapper
    return decorator

# Example usage
class ResilientLabSystem:
    """Example of resilient lab system using error recovery"""
    
    def __init__(self):
        self.error_manager = ErrorRecoveryManager()
        self.state_manager = self.error_manager.state_manager
    
    @with_recovery('sensor_reading', ErrorSeverity.MEDIUM, 
                  fallback_value={'temperature': 23.0})
    def read_temperature(self) -> Dict[str, float]:
        """Read temperature with automatic recovery"""
        # Simulate potential error
        import random
        if random.random() < 0.3:
            raise Exception("Sensor communication error")
        
        return {'temperature': 23.5 + random.random()}
    
    @with_recovery('voice_processing', ErrorSeverity.HIGH)
    def process_voice_command(self, audio_data: bytes) -> str:
        """Process voice with recovery"""
        # Simulate processing
        if not audio_data:
            raise ValueError("Empty audio data")
        
        return "Processed command"
    
    def demonstrate_recovery(self):
        """Demonstrate error recovery capabilities"""
        print("Testing resilient lab system...")
        
        # Test sensor reading with failures
        for i in range(5):
            result = self.read_temperature()
            print(f"Temperature reading {i+1}: {result}")
        
        # Test voice processing
        try:
            command = self.process_voice_command(b"")
        except:
            print("Voice processing failed, recovery handled")
        
        # Get error statistics
        stats = self.error_manager.get_error_statistics()
        print(f"\nError Statistics: {json.dumps(stats, indent=2)}")

if __name__ == "__main__":
    # Demonstrate error recovery
    system = ResilientLabSystem()
    system.demonstrate_recovery()