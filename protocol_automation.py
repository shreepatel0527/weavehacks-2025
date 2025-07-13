"""
Protocol automation system with step-by-step guidance and timing
"""
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
import weave
from pathlib import Path

class StepType(Enum):
    MEASUREMENT = "measurement"
    MIXING = "mixing"
    HEATING = "heating"
    COOLING = "cooling"
    STIRRING = "stirring"
    WAITING = "waiting"
    OBSERVATION = "observation"
    SAFETY_CHECK = "safety_check"
    CALCULATION = "calculation"

class StepStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"

@dataclass
class ProtocolStep:
    id: str
    name: str
    description: str
    step_type: StepType
    duration_seconds: Optional[int] = None
    temperature: Optional[float] = None
    speed_rpm: Optional[int] = None
    required_materials: List[str] = field(default_factory=list)
    safety_requirements: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    data_to_record: List[str] = field(default_factory=list)
    calculations: Dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    recorded_data: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

class ProtocolAutomation:
    """Automated protocol execution with guidance and monitoring"""
    
    def __init__(self, protocol_file: Optional[Path] = None):
        self.protocol_steps: List[ProtocolStep] = []
        self.current_step_index = 0
        self.is_running = False
        self.is_paused = False
        self.step_callbacks: Dict[str, List[Callable]] = {}
        self.timer_thread: Optional[threading.Thread] = None
        self.start_time: Optional[datetime] = None
        self.completion_time: Optional[datetime] = None
        
        # Load protocol if provided
        if protocol_file:
            self.load_protocol(protocol_file)
        else:
            self._load_default_protocol()
        
        # Initialize W&B
        weave.init('protocol-automation')
    
    def _load_default_protocol(self):
        """Load default Au25 nanoparticle synthesis protocol"""
        self.protocol_steps = [
            ProtocolStep(
                id="step_001",
                name="Weigh HAuCl₄·3H₂O",
                description="Weigh 0.1576g of gold(III) chloride trihydrate",
                step_type=StepType.MEASUREMENT,
                required_materials=["HAuCl₄·3H₂O", "analytical balance", "weighing paper"],
                data_to_record=["mass_gold"],
                safety_requirements=["gloves", "lab coat"]
            ),
            ProtocolStep(
                id="step_002",
                name="Measure Nanopure Water",
                description="Measure 5 mL of nanopure water",
                step_type=StepType.MEASUREMENT,
                required_materials=["nanopure water", "graduated cylinder"],
                data_to_record=["volume_water"],
                dependencies=["step_001"]
            ),
            ProtocolStep(
                id="step_003",
                name="Dissolve Gold Compound",
                description="Dissolve HAuCl₄·3H₂O in nanopure water",
                step_type=StepType.MIXING,
                duration_seconds=120,
                required_materials=["stir bar", "beaker"],
                dependencies=["step_001", "step_002"]
            ),
            ProtocolStep(
                id="step_004",
                name="Weigh TOAB",
                description="Weigh ~0.25g of tetraoctylammonium bromide",
                step_type=StepType.MEASUREMENT,
                required_materials=["TOAB", "analytical balance"],
                data_to_record=["mass_toab"],
                dependencies=["step_003"]
            ),
            ProtocolStep(
                id="step_005",
                name="Measure Toluene",
                description="Measure 10 mL of toluene",
                step_type=StepType.MEASUREMENT,
                required_materials=["toluene", "graduated cylinder"],
                data_to_record=["volume_toluene"],
                safety_requirements=["fume hood", "gloves"],
                dependencies=["step_004"]
            ),
            ProtocolStep(
                id="step_006",
                name="Dissolve TOAB",
                description="Dissolve TOAB in toluene",
                step_type=StepType.MIXING,
                duration_seconds=180,
                required_materials=["stir bar", "beaker"],
                dependencies=["step_004", "step_005"]
            ),
            ProtocolStep(
                id="step_007",
                name="Combine Solutions",
                description="Combine gold and TOAB solutions in round-bottom flask",
                step_type=StepType.MIXING,
                required_materials=["25 mL round-bottom flask", "stir bar"],
                dependencies=["step_003", "step_006"]
            ),
            ProtocolStep(
                id="step_008",
                name="Vigorous Stirring",
                description="Stir vigorously at ~1100 rpm for 15 minutes",
                step_type=StepType.STIRRING,
                duration_seconds=900,
                speed_rpm=1100,
                required_materials=["stir plate"],
                safety_requirements=["fume hood"],
                dependencies=["step_007"]
            ),
            ProtocolStep(
                id="step_009",
                name="Remove Aqueous Layer",
                description="Remove aqueous layer with syringe",
                step_type=StepType.MIXING,
                required_materials=["10 mL syringe"],
                dependencies=["step_008"]
            ),
            ProtocolStep(
                id="step_010",
                name="Nitrogen Purge",
                description="Purge flask with nitrogen gas",
                step_type=StepType.MIXING,
                duration_seconds=60,
                required_materials=["N2 gas", "gas needle"],
                safety_requirements=["gas handling training"],
                dependencies=["step_009"]
            ),
            ProtocolStep(
                id="step_011",
                name="Cool to 0°C",
                description="Cool in ice bath to 0°C over 30 minutes",
                step_type=StepType.COOLING,
                duration_seconds=1800,
                temperature=0.0,
                required_materials=["ice bath", "thermometer"],
                dependencies=["step_010"]
            ),
            ProtocolStep(
                id="step_012",
                name="Calculate Sulfur Amount",
                description="Calculate amount of PhCH₂CH₂SH needed (3 eq)",
                step_type=StepType.CALCULATION,
                calculations={"type": "sulfur_amount", "equivalents": 3},
                dependencies=["step_011"]
            ),
            ProtocolStep(
                id="step_013",
                name="Add Sulfur Compound",
                description="Add calculated amount of 2-phenylethanethiol",
                step_type=StepType.MIXING,
                required_materials=["PhCH₂CH₂SH", "micropipette"],
                data_to_record=["mass_sulfur"],
                dependencies=["step_012"]
            ),
            ProtocolStep(
                id="step_014",
                name="Observe Color Change",
                description="Observe color change: deep red → faint yellow → clear",
                step_type=StepType.OBSERVATION,
                duration_seconds=3600,
                data_to_record=["color_observations"],
                dependencies=["step_013"]
            ),
            ProtocolStep(
                id="step_015",
                name="Calculate NaBH₄ Amount",
                description="Calculate amount of NaBH₄ needed (10 eq)",
                step_type=StepType.CALCULATION,
                calculations={"type": "nabh4_amount", "equivalents": 10},
                dependencies=["step_014"]
            ),
            ProtocolStep(
                id="step_016",
                name="Prepare NaBH₄ Solution",
                description="Dissolve NaBH₄ in 7 mL ice-cold nanopure water",
                step_type=StepType.MIXING,
                required_materials=["NaBH₄", "ice-cold water"],
                data_to_record=["mass_nabh4", "volume_water_cold"],
                dependencies=["step_015"]
            ),
            ProtocolStep(
                id="step_017",
                name="Add NaBH₄ Solution",
                description="Add NaBH₄ solution to reaction",
                step_type=StepType.MIXING,
                safety_requirements=["hydrogen gas evolution"],
                dependencies=["step_016"]
            ),
            ProtocolStep(
                id="step_018",
                name="Stir Overnight",
                description="Stir overnight under N₂ atmosphere",
                step_type=StepType.STIRRING,
                duration_seconds=28800,  # 8 hours
                speed_rpm=600,
                dependencies=["step_017"]
            ),
            ProtocolStep(
                id="step_019",
                name="Final Workup",
                description="Remove aqueous layer and add ethanol to precipitate",
                step_type=StepType.MIXING,
                required_materials=["ethanol", "centrifuge"],
                data_to_record=["mass_final"],
                dependencies=["step_018"]
            )
        ]
    
    @weave.op()
    def start_protocol(self):
        """Start protocol execution"""
        if self.is_running:
            return {"error": "Protocol already running"}
        
        self.is_running = True
        self.is_paused = False
        self.start_time = datetime.now()
        self.current_step_index = 0
        
        # Log protocol start
        weave.log({
            'protocol_event': {
                'type': 'started',
                'timestamp': self.start_time.isoformat(),
                'total_steps': len(self.protocol_steps)
            }
        })
        
        # Start with first step
        self._execute_current_step()
        
        return {"status": "started", "first_step": self.protocol_steps[0].name}
    
    @weave.op()
    def _execute_current_step(self):
        """Execute the current protocol step"""
        if not self.is_running or self.current_step_index >= len(self.protocol_steps):
            self._complete_protocol()
            return
        
        current_step = self.protocol_steps[self.current_step_index]
        
        # Check dependencies
        if not self._check_dependencies(current_step):
            current_step.status = StepStatus.FAILED
            current_step.notes.append("Dependencies not met")
            return
        
        # Mark as in progress
        current_step.status = StepStatus.IN_PROGRESS
        current_step.start_time = datetime.now()
        
        # Log step start
        weave.log({
            'step_event': {
                'type': 'started',
                'step_id': current_step.id,
                'step_name': current_step.name,
                'step_type': current_step.step_type.value,
                'timestamp': current_step.start_time.isoformat()
            }
        })
        
        # Execute callbacks
        self._trigger_callbacks('step_started', current_step)
        
        # Handle timed steps
        if current_step.duration_seconds:
            self._start_timer(current_step)
        else:
            # For non-timed steps, mark ready for user action
            current_step.status = StepStatus.READY
            self._trigger_callbacks('step_ready', current_step)
    
    def _start_timer(self, step: ProtocolStep):
        """Start timer for timed steps"""
        def timer_callback():
            time.sleep(step.duration_seconds)
            
            if not self.is_paused and step.status == StepStatus.IN_PROGRESS:
                self._complete_step(step)
        
        self.timer_thread = threading.Thread(target=timer_callback)
        self.timer_thread.daemon = True
        self.timer_thread.start()
    
    @weave.op()
    def _complete_step(self, step: ProtocolStep):
        """Complete a protocol step"""
        step.status = StepStatus.COMPLETED
        step.end_time = datetime.now()
        
        # Log completion
        duration = (step.end_time - step.start_time).total_seconds() if step.start_time else 0
        
        weave.log({
            'step_event': {
                'type': 'completed',
                'step_id': step.id,
                'step_name': step.name,
                'duration_seconds': duration,
                'timestamp': step.end_time.isoformat()
            }
        })
        
        # Trigger callbacks
        self._trigger_callbacks('step_completed', step)
        
        # Move to next step
        self.current_step_index += 1
        if self.is_running:
            self._execute_current_step()
    
    def _complete_protocol(self):
        """Complete the entire protocol"""
        self.is_running = False
        self.completion_time = datetime.now()
        
        # Calculate total duration
        total_duration = (self.completion_time - self.start_time).total_seconds() if self.start_time else 0
        
        # Log completion
        weave.log({
            'protocol_event': {
                'type': 'completed',
                'total_duration_seconds': total_duration,
                'completed_steps': sum(1 for s in self.protocol_steps if s.status == StepStatus.COMPLETED),
                'timestamp': self.completion_time.isoformat()
            }
        })
        
        self._trigger_callbacks('protocol_completed', None)
    
    def _check_dependencies(self, step: ProtocolStep) -> bool:
        """Check if all dependencies are met"""
        for dep_id in step.dependencies:
            dep_step = next((s for s in self.protocol_steps if s.id == dep_id), None)
            if not dep_step or dep_step.status != StepStatus.COMPLETED:
                return False
        return True
    
    def record_data(self, step_id: str, data_type: str, value: Any):
        """Record data for a step"""
        step = next((s for s in self.protocol_steps if s.id == step_id), None)
        if step:
            step.recorded_data[data_type] = {
                'value': value,
                'timestamp': datetime.now().isoformat()
            }
            
            # Log data recording
            weave.log({
                'data_recorded': {
                    'step_id': step_id,
                    'data_type': data_type,
                    'value': value
                }
            })
            
            # Check if all required data is recorded
            if all(dt in step.recorded_data for dt in step.data_to_record):
                if step.status == StepStatus.READY:
                    self._complete_step(step)
    
    def add_note(self, step_id: str, note: str):
        """Add a note to a step"""
        step = next((s for s in self.protocol_steps if s.id == step_id), None)
        if step:
            step.notes.append({
                'text': note,
                'timestamp': datetime.now().isoformat()
            })
    
    def pause_protocol(self):
        """Pause protocol execution"""
        self.is_paused = True
        self._trigger_callbacks('protocol_paused', None)
    
    def resume_protocol(self):
        """Resume protocol execution"""
        self.is_paused = False
        self._trigger_callbacks('protocol_resumed', None)
    
    def skip_step(self, step_id: str, reason: str):
        """Skip a protocol step"""
        step = next((s for s in self.protocol_steps if s.id == step_id), None)
        if step and step.status in [StepStatus.PENDING, StepStatus.READY]:
            step.status = StepStatus.SKIPPED
            step.notes.append(f"Skipped: {reason}")
            
            # Move to next step if this was current
            if self.protocol_steps[self.current_step_index].id == step_id:
                self.current_step_index += 1
                self._execute_current_step()
    
    def get_current_step(self) -> Optional[ProtocolStep]:
        """Get the current protocol step"""
        if 0 <= self.current_step_index < len(self.protocol_steps):
            return self.protocol_steps[self.current_step_index]
        return None
    
    def get_protocol_status(self) -> Dict[str, Any]:
        """Get overall protocol status"""
        completed_steps = sum(1 for s in self.protocol_steps if s.status == StepStatus.COMPLETED)
        
        current_step = self.get_current_step()
        
        status = {
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'total_steps': len(self.protocol_steps),
            'completed_steps': completed_steps,
            'progress_percentage': (completed_steps / len(self.protocol_steps)) * 100,
            'current_step': {
                'id': current_step.id,
                'name': current_step.name,
                'status': current_step.status.value,
                'type': current_step.step_type.value
            } if current_step else None,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'elapsed_time': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        }
        
        # Add time remaining for timed steps
        if current_step and current_step.duration_seconds and current_step.start_time:
            elapsed = (datetime.now() - current_step.start_time).total_seconds()
            remaining = max(0, current_step.duration_seconds - elapsed)
            status['current_step']['time_remaining'] = remaining
        
        return status
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register a callback for protocol events"""
        if event_type not in self.step_callbacks:
            self.step_callbacks[event_type] = []
        self.step_callbacks[event_type].append(callback)
    
    def _trigger_callbacks(self, event_type: str, step: Optional[ProtocolStep]):
        """Trigger callbacks for an event"""
        if event_type in self.step_callbacks:
            for callback in self.step_callbacks[event_type]:
                try:
                    callback(step)
                except Exception as e:
                    print(f"Error in callback: {e}")
    
    def export_protocol_data(self) -> Dict[str, Any]:
        """Export all protocol data"""
        return {
            'protocol_name': 'Au25 Nanoparticle Synthesis',
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'completion_time': self.completion_time.isoformat() if self.completion_time else None,
            'steps': [
                {
                    'id': step.id,
                    'name': step.name,
                    'type': step.step_type.value,
                    'status': step.status.value,
                    'start_time': step.start_time.isoformat() if step.start_time else None,
                    'end_time': step.end_time.isoformat() if step.end_time else None,
                    'recorded_data': step.recorded_data,
                    'notes': step.notes
                }
                for step in self.protocol_steps
            ]
        }

# Example usage
if __name__ == "__main__":
    # Create protocol automation
    protocol = ProtocolAutomation()
    
    # Register callbacks
    def on_step_started(step):
        print(f"\n🔵 Started: {step.name}")
        print(f"   Type: {step.step_type.value}")
        if step.required_materials:
            print(f"   Required: {', '.join(step.required_materials)}")
        if step.duration_seconds:
            print(f"   Duration: {step.duration_seconds}s")
    
    def on_step_ready(step):
        print(f"\n🟡 Ready for input: {step.name}")
        if step.data_to_record:
            print(f"   Record: {', '.join(step.data_to_record)}")
    
    def on_step_completed(step):
        print(f"\n✅ Completed: {step.name}")
    
    protocol.register_callback('step_started', on_step_started)
    protocol.register_callback('step_ready', on_step_ready)
    protocol.register_callback('step_completed', on_step_completed)
    
    # Start protocol
    print("Starting Au25 Nanoparticle Synthesis Protocol")
    protocol.start_protocol()
    
    # Simulate data recording
    time.sleep(2)
    current = protocol.get_current_step()
    if current and current.id == "step_001":
        protocol.record_data("step_001", "mass_gold", 0.1576)
    
    # Run for a bit
    try:
        while protocol.is_running:
            time.sleep(1)
            status = protocol.get_protocol_status()
            if status['current_step'] and 'time_remaining' in status['current_step']:
                remaining = status['current_step']['time_remaining']
                print(f"\r⏱️  {status['current_step']['name']}: {remaining:.0f}s remaining", end="")
    
    except KeyboardInterrupt:
        print("\nProtocol interrupted")
        protocol.pause_protocol()