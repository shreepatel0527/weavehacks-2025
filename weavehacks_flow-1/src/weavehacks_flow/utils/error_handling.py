"""
Comprehensive error handling and recovery system for Lab Assistant
"""
import functools
import logging
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
from enum import Enum
import json
from pathlib import Path
try:
    import weave
except ImportError:
    weave = None

try:
    import wandb
except ImportError:
    wandb = None

def safe_wandb_log(data: dict):
    """Safely log to wandb, handling cases where wandb is not initialized"""
    if wandb is None:
        return
    try:
        wandb.log(data)
    except wandb.errors.UsageError:
        # wandb not initialized, try to initialize minimally
        try:
            wandb.init(project="lab-assistant-errors", mode="disabled")
            wandb.log(data)
        except Exception:
            # If all else fails, just skip logging
            pass
    except Exception:
        # Any other wandb error, skip logging
        pass

T = TypeVar('T')

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for better organization"""
    SENSOR = "sensor"
    CALCULATION = "calculation"
    API = "api"
    SAFETY = "safety"
    UI = "ui"
    AGENT = "agent"
    SYSTEM = "system"

class LabAssistantError(Exception):
    """Base exception for Lab Assistant"""
    def __init__(self, message: str, category: ErrorCategory, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.timestamp = datetime.now()

class SensorError(LabAssistantError):
    """Sensor-related errors"""
    def __init__(self, message: str, sensor_type: str, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        super().__init__(
            message, 
            ErrorCategory.SENSOR, 
            severity,
            {'sensor_type': sensor_type}
        )

class CalculationError(LabAssistantError):
    """Calculation-related errors"""
    def __init__(self, message: str, calculation_type: str,
                 input_values: Optional[Dict[str, Any]] = None):
        super().__init__(
            message,
            ErrorCategory.CALCULATION,
            ErrorSeverity.HIGH,
            {
                'calculation_type': calculation_type,
                'input_values': input_values or {}
            }
        )

class SafetyError(LabAssistantError):
    """Safety-critical errors"""
    def __init__(self, message: str, parameter: str, 
                 current_value: float, threshold: float):
        super().__init__(
            message,
            ErrorCategory.SAFETY,
            ErrorSeverity.CRITICAL,
            {
                'parameter': parameter,
                'current_value': current_value,
                'threshold': threshold
            }
        )

class APIError(LabAssistantError):
    """API-related errors"""
    def __init__(self, message: str, api_name: str,
                 status_code: Optional[int] = None):
        super().__init__(
            message,
            ErrorCategory.API,
            ErrorSeverity.MEDIUM,
            {
                'api_name': api_name,
                'status_code': status_code
            }
        )

class ErrorHandler:
    """Centralized error handling system"""
    
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Error statistics
        self.error_counts = {
            category: {severity: 0 for severity in ErrorSeverity}
            for category in ErrorCategory
        }
        
        # Recovery strategies
        self.recovery_strategies = {}
        self._register_default_strategies()
        
        # Initialize W&B if available
        try:
            weave.init('lab-assistant-errors')
            self.use_weave = True
        except:
            self.use_weave = False
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger('lab_assistant')
        logger.setLevel(logging.DEBUG)
        
        # File handler for errors
        error_handler = logging.FileHandler(
            self.log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
        )
        error_handler.setLevel(logging.ERROR)
        
        # File handler for all logs
        general_handler = logging.FileHandler(
            self.log_dir / f"lab_assistant_{datetime.now().strftime('%Y%m%d')}.log"
        )
        general_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        for handler in [error_handler, general_handler, console_handler]:
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _register_default_strategies(self):
        """Register default recovery strategies"""
        # Sensor errors: use cached values
        self.recovery_strategies[ErrorCategory.SENSOR] = self._sensor_recovery
        
        # Calculation errors: return safe defaults
        self.recovery_strategies[ErrorCategory.CALCULATION] = self._calculation_recovery
        
        # API errors: retry with backoff
        self.recovery_strategies[ErrorCategory.API] = self._api_recovery
        
        # Safety errors: trigger emergency protocol
        self.recovery_strategies[ErrorCategory.SAFETY] = self._safety_recovery
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Any:
        """Main error handling method"""
        # Create error record
        error_record = self._create_error_record(error, context)
        
        # Log error
        self._log_error(error_record)
        
        # Update statistics
        self._update_statistics(error_record)
        
        # Execute recovery strategy
        if isinstance(error, LabAssistantError):
            recovery_result = self._execute_recovery(error)
        else:
            # Handle unexpected errors
            recovery_result = self._handle_unexpected_error(error, context)
        
        return recovery_result
    
    def _create_error_record(self, error: Exception, 
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create detailed error record"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        if isinstance(error, LabAssistantError):
            record.update({
                'category': error.category.value,
                'severity': error.severity.value,
                'details': error.details
            })
        else:
            record.update({
                'category': ErrorCategory.SYSTEM.value,
                'severity': ErrorSeverity.HIGH.value
            })
        
        return record
    
    def _log_error(self, error_record: Dict[str, Any]):
        """Log error to multiple destinations"""
        # Log to file
        if error_record['severity'] == ErrorSeverity.CRITICAL.value:
            self.logger.critical(json.dumps(error_record))
        elif error_record['severity'] == ErrorSeverity.HIGH.value:
            self.logger.error(json.dumps(error_record))
        else:
            self.logger.warning(json.dumps(error_record))
        
        # Log to W&B if available
        if self.use_weave:
            try:
                safe_wandb_log({'error': error_record})
            except:
                pass
        
        # Save critical errors to separate file
        if error_record['severity'] == ErrorSeverity.CRITICAL.value:
            self._save_critical_error(error_record)
    
    def _save_critical_error(self, error_record: Dict[str, Any]):
        """Save critical errors to separate file"""
        critical_log = self.log_dir / "critical_errors.jsonl"
        
        with open(critical_log, 'a') as f:
            f.write(json.dumps(error_record) + '\n')
    
    def _update_statistics(self, error_record: Dict[str, Any]):
        """Update error statistics"""
        category = ErrorCategory(error_record['category'])
        severity = ErrorSeverity(error_record['severity'])
        
        self.error_counts[category][severity] += 1
    
    def _execute_recovery(self, error: LabAssistantError) -> Any:
        """Execute recovery strategy based on error category"""
        if error.category in self.recovery_strategies:
            return self.recovery_strategies[error.category](error)
        else:
            return None
    
    def _sensor_recovery(self, error: SensorError) -> Dict[str, float]:
        """Recovery strategy for sensor errors"""
        self.logger.info(f"Executing sensor recovery for: {error.details['sensor_type']}")
        
        # Return safe default values
        defaults = {
            'temperature': 23.0,
            'pressure': 101.3,
            'stirring_rpm': 1100,
            'ph': 7.0
        }
        
        sensor_type = error.details.get('sensor_type', '')
        return {sensor_type: defaults.get(sensor_type, 0.0)}
    
    def _calculation_recovery(self, error: CalculationError) -> Any:
        """Recovery strategy for calculation errors"""
        self.logger.info(f"Executing calculation recovery for: {error.details['calculation_type']}")
        
        # Return safe defaults based on calculation type
        calc_type = error.details.get('calculation_type', '')
        
        if 'yield' in calc_type:
            return 0.0  # 0% yield is safe
        elif 'mass' in calc_type or 'amount' in calc_type:
            return 0.001  # Small safe amount
        else:
            return None
    
    def _api_recovery(self, error: APIError) -> Any:
        """Recovery strategy for API errors"""
        self.logger.info(f"Executing API recovery for: {error.details['api_name']}")
        
        # Return cached or default response
        return {
            'status': 'error',
            'message': 'API temporarily unavailable',
            'cached': True
        }
    
    def _safety_recovery(self, error: SafetyError) -> Dict[str, Any]:
        """Recovery strategy for safety errors"""
        self.logger.critical(f"SAFETY VIOLATION: {error.message}")
        
        # Trigger emergency protocol
        return {
            'action': 'emergency_stop',
            'parameter': error.details['parameter'],
            'current_value': error.details['current_value'],
            'threshold': error.details['threshold'],
            'timestamp': datetime.now().isoformat()
        }
    
    def _handle_unexpected_error(self, error: Exception, 
                               context: Optional[Dict[str, Any]] = None) -> Any:
        """Handle unexpected errors"""
        self.logger.error(f"Unexpected error: {type(error).__name__}: {str(error)}")
        
        # Log full details
        self._save_critical_error({
            'timestamp': datetime.now().isoformat(),
            'error_type': 'UnexpectedError',
            'original_type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        })
        
        return None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        total_errors = sum(
            count
            for category_counts in self.error_counts.values()
            for count in category_counts.values()
        )
        
        return {
            'total_errors': total_errors,
            'by_category': {
                cat.value: sum(counts.values())
                for cat, counts in self.error_counts.items()
            },
            'by_severity': {
                sev.value: sum(
                    counts[sev]
                    for counts in self.error_counts.values()
                )
                for sev in ErrorSeverity
            },
            'detailed_counts': {
                cat.value: {sev.value: count for sev, count in counts.items()}
                for cat, counts in self.error_counts.items()
            }
        }

# Global error handler instance
error_handler = ErrorHandler()

def safe_execute(func: Callable[..., T], 
                default_return: Optional[T] = None,
                error_category: ErrorCategory = ErrorCategory.SYSTEM,
                error_severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> Callable[..., T]:
    """Decorator for safe function execution with error handling"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except LabAssistantError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            # Handle unexpected errors
            context = {
                'function': func.__name__,
                'args': str(args)[:100],  # Limit size
                'kwargs': str(kwargs)[:100]
            }
            
            # Create custom error
            lab_error = LabAssistantError(
                f"Error in {func.__name__}: {str(e)}",
                error_category,
                error_severity,
                context
            )
            
            # Handle error
            result = error_handler.handle_error(lab_error, context)
            
            # Return default or handled result
            return result if result is not None else default_return
    
    return wrapper

def validate_input(validation_rules: Dict[str, Callable]) -> Callable:
    """Decorator for input validation"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            
            # Get function signature to map positional args to parameter names
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            
            # Create a combined dict of all arguments
            all_args = {}
            
            # Add positional arguments
            for i, arg in enumerate(args):
                if i < len(param_names):
                    all_args[param_names[i]] = arg
            
            # Add keyword arguments
            all_args.update(kwargs)
            
            # Validate based on rules
            for param_name, validator in validation_rules.items():
                if param_name in all_args:
                    value = all_args[param_name]
                    if not validator(value):
                        raise CalculationError(
                            f"Invalid {param_name}: {value}",
                            func.__name__,
                            all_args
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

# Example usage functions
def get_error_summary() -> str:
    """Get a summary of recent errors"""
    stats = error_handler.get_error_statistics()
    
    summary = f"""
Error Summary:
- Total errors: {stats['total_errors']}
- Critical errors: {stats['by_severity'].get('critical', 0)}
- High severity: {stats['by_severity'].get('high', 0)}
- Most common category: {max(stats['by_category'].items(), key=lambda x: x[1])[0] if stats['by_category'] else 'None'}
"""
    
    return summary