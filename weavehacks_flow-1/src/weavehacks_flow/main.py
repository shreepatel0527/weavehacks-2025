#!/usr/bin/env python
from random import randint
from pydantic import BaseModel
from datetime import datetime
# CrewAI Flow compatibility layer for Python 3.9
class FlowCompatibility:
    """Simple compatibility layer to replace crewai.flow for Python 3.9"""
    def __init__(self, state_class):
        self.state = state_class()
        self._step_methods = []
        
    def kickoff(self):
        """Execute the workflow steps in sequence"""
        for method in self._step_methods:
            method()
    
    def add_step(self, method):
        """Add a step method to the workflow"""
        self._step_methods.append(method)

def start():
    """Decorator to mark start method"""
    def decorator(func):
        func._is_start = True
        return func
    return decorator

def listen(previous_method):
    """Decorator to mark method dependencies"""
    def decorator(func):
        func._listens_to = previous_method
        return func
    return decorator

# Use compatibility class instead of Flow
Flow = FlowCompatibility
import weave
from agents.data_collection_agent import DataCollectionAgent
from agents.lab_control_agent import LabControlAgent
from agents.safety_monitoring_agent import EnhancedSafetyMonitoringAgent
from agents.safety_monitoring_agent import SafetyMonitoringAgent
from agents.video_monitoring_agent import VideoMonitoringAgent
from agents.voice_recognition_agent import SpeechRecognizerAgent
# from .crews.data_collection_crew.data_collection_crew import DataCollectionCrew
# from .crews.lab_control_crew.lab_control_crew import LabControlCrew
# from .crews.safety_monitoring_crew.safety_monitoring_crew import SafetyMonitoringCrew
from .utils.chemistry_calculations import (
    calculate_sulfur_amount,
    calculate_nabh4_amount,
    calculate_percent_yield
)

# Initialize W&B Weave (optional - will work without API key)
try:
    weave.init('weavehacks-lab-assistant')
    print("Weave initialized successfully")
except Exception as e:
    print(f"Weave initialization failed (this is optional): {e}")
    print("Continuing without W&B logging...")

class ExperimentState(BaseModel):
    # experiment tracking
    step_num: int = 0
    exp_status: str = "not started" # not started, in progress, complete, halted
    # solids mass
    mass_gold: float = 0.0
    mass_toab: float = 0.0
    mass_sulfur: float = 0.0
    mass_nabh4: float = 0.0
    mass_final: float= 0.0
    # liquids volume
    volume_toluene: float = 0.0
    volume_nanopure_rt: float = 0.0
    volume_nanopure_cold: float = 0.0

    observations: str = ""
    # safety status
    safety_status: str = "safe" # safe or unsafe

class ExperimentFlow(FlowCompatibility):
    def __init__(self):
        super().__init__(ExperimentState)
        self.data_agent = DataCollectionAgent()
        self.lab_agent = LabControlAgent()
        self.safety_agent = EnhancedSafetyMonitoringAgent()
        #self.safety_agent = SafetyMonitoringAgent()
        self.video_agent = VideoMonitoringAgent()
        self.voice_agent = SpeechRecognizerAgent(model_size="base")
        
        # Initialize video agent with graceful fallback if OpenCV not available
        try:
            self.video_agent = VideoMonitoringAgent()
        except ImportError as e:
            print(f"Video monitoring not available: {e}")
            self.video_agent = None
        
        self._setup_workflow()
    
    def _setup_workflow(self):
        """Set up the workflow execution order"""
        # Define the workflow steps in order
        workflow_steps = [
            self.initialize_experiment,
            self.weigh_gold,
            self.measure_nanopure_rt,
            self.weigh_toab,
            self.measure_toluene,
            self.calculate_sulfur_amount,
            self.weigh_sulfur,
            self.initialize_overnight_monitoring,
            self.calculate_nabh4_amount,
            self.weigh_nabh4,
            self.measure_nanopure_cold,
            self.weigh_final,
            self.calculate_percent_yield,
            self.finalize_experiment
        ]
        
        for step in workflow_steps:
            self.add_step(step)

    @start()
    def initialize_experiment(self):
        print("Initializing experiment...")

    @weave.op()
    def update_step(self):
        self.state.step_num += 1
        print(f"Step {self.state.step_num} completed.")
    
    
    # need to add a step prompting method 

    @listen(initialize_experiment)
    @weave.op()
    def weigh_gold(self):
        self.state.mass_gold = self.data_agent.record_data(
            "Weigh HAuCl₄·3H₂O (0.1576g) -- record mass", use_voice=True
        )
        print(f"Gold mass recorded: {self.state.mass_gold}g")
        self.update_step()

    @listen(weigh_gold)
    @weave.op()
    def measure_nanopure_rt(self):
        self.state.volume_nanopure_rt = self.data_agent.record_data(
            "Measure water (10mL) -- record vol", use_voice=True
        )
        print(f"Room Temp Nanopure Volume recorded: {self.state.volume_nanopure_rt}mL")
        self.update_step()

    @listen(measure_nanopure_rt)
    @weave.op()
    def weigh_toab(self):
        self.state.mass_toab = self.data_agent.record_data(
            "Weigh TOAB (~0.25g) -- record mass", use_voice=True
        )
        print(f"TOAB mass recorded: {self.state.mass_toab}g")
        self.update_step()

    @listen(weigh_toab)
    @weave.op()
    def measure_toluene(self):
        self.state.volume_toluene = self.data_agent.record_data(
            "Measure toluene (10mL) -- record vol", use_voice=True
        )
        print(f"Toluene volume recorded: {self.state.volume_toluene}mL")
        self.update_step()

    @listen(measure_toluene)
    @weave.op()
    def calculate_sulfur_amount(self):
        """Calculate amount of PhCH₂CH₂SH (3 eq. relative to gold)"""
        calc_result = calculate_sulfur_amount(self.state.mass_gold)
        
        print(f"\nCalculation for Sulfur (PhCH₂CH₂SH):")
        print(f"Gold mass: {self.state.mass_gold:.4f}g")
        print(f"Moles of gold: {calc_result['moles_gold']:.6f} mol")
        print(f"Moles of sulfur needed (3 eq): {calc_result['moles_sulfur']:.6f} mol")
        print(f"Mass of sulfur needed: {calc_result['mass_sulfur_g']:.4f}g")
        
        return calc_result['mass_sulfur_g'] 

    @listen(calculate_sulfur_amount)
    @weave.op()
    def weigh_sulfur(self):
        mass_needed = self.calculate_sulfur_amount()
        prompt = f"Weigh PhCH₂CH₂SH (~{mass_needed:.3f}g) -- record mass"
        self.state.mass_sulfur = self.data_agent.record_data(prompt)
        print(f"Sulfur mass recorded: {self.state.mass_sulfur}g")
        self.update_step()

    # DO NOT MODIFY ABOVE THIS !!! 

    @listen(weigh_sulfur)
    @weave.op()
    def initialize_overnight_monitoring(self):
        """Initialize overnight monitoring system with video, safety, and lab control integration"""
        print("\n" + "="*50)
        print("INITIALIZING OVERNIGHT MONITORING SYSTEM")
        print("="*50)
        
        # Initialize monitoring status
        monitoring_status = {
            'video_monitoring': False,
            'video_recording': False,
            'safety_monitoring': False,
            'lab_control_alert': False
        }
        
        # Start video monitoring for the experiment
        print("Starting video monitoring...")
        video_result = self.video_agent.start_monitoring()
        if video_result['status'] == 'success':
            print("✓ Video monitoring started successfully")
            monitoring_status['video_monitoring'] = True
        else:
            print(f"✗ Warning: Video monitoring failed to start: {video_result['message']}")
        
        # Start video recording
        print("Starting video recording...")
        recording_result = self.video_agent.start_recording()
        if recording_result['status'] == 'success':
            print("✓ Video recording started")
            monitoring_status['video_recording'] = True
        else:
            print(f"✗ Warning: Video recording failed to start: {recording_result['message']}")
        
        # Register safety callback for video monitoring
        self.video_agent.register_callback(self._handle_video_safety_event)
        
        # Start safety monitoring
        print("Activating safety monitoring...")
        self.safety_agent.monitor_parameters()
        monitoring_status['safety_monitoring'] = True
        print("✓ Safety monitoring activated")
        
        # Put lab control agent on alert
        print("Setting lab control agent to alert mode...")
        self.lab_agent.turn_on("emergency_monitoring")
        self.lab_agent.turn_on("safety_systems")
        monitoring_status['lab_control_alert'] = True
        print("✓ Lab control agent on alert")
        
        # Start continuous monitoring thread
        import threading
        self.monitoring_thread = threading.Thread(
            target=self._continuous_monitoring_loop,
            name="OvernightMonitoringThread",
            daemon=True
        )
        self.monitoring_active = True
        self.monitoring_thread.start()
        print("✓ Continuous monitoring thread started")
        
        print("\n" + "="*50)
        print("OVERNIGHT MONITORING SYSTEM ACTIVE")
        print("System will monitor for safety violations and halt experiment if needed")
        print("="*50 + "\n")
        
        # Log to Weave
        try:
            import wandb
            def safe_wandb_log(data: dict):
                """Safely log to wandb, handling cases where wandb is not initialized"""
                try:
                    wandb.log(data)
                except wandb.errors.UsageError:
                    try:
                        wandb.init(project="lab-assistant-agents", mode="disabled")
                        wandb.log(data)
                    except Exception:
                        pass
                except Exception:
                    pass
            
            safe_wandb_log({
                'overnight_monitoring': {
                    'action': 'initialize',
                    'status': monitoring_status,
                    'timestamp': datetime.now().isoformat()
                }
            })
        except ImportError:
            print("W&B not available, skipping logging")
        
        self.update_step()
        return monitoring_status
    
    def _handle_video_safety_event(self, event):
        """Handle safety events detected by video monitoring"""
        from .agents.video_monitoring_agent import EventType
        
        if event.event_type == EventType.SAFETY_VIOLATION:
            print(f"\n⚠️ VIDEO SAFETY VIOLATION DETECTED: {event.description}")
            self._trigger_emergency_halt("Video monitoring detected safety violation", event.description)
    
    def _continuous_monitoring_loop(self):
        """Continuous monitoring loop for overnight operation"""
        import time
        
        while self.monitoring_active and hasattr(self, 'monitoring_active'):
            try:
                # Check safety parameters
                self.safety_agent.monitor_parameters()
                
                if not self.safety_agent.is_safe():
                    self.state.safety_status = "unsafe"
                    self.safety_agent.notify_scientist()
                    print("Safety status: Unsafe! Notifying scientist.")
                    self._trigger_emergency_halt("Safety parameters out of range", "Environmental conditions unsafe")
                    break
                
                # Check for video safety violations
                if self.video_agent.has_safety_violations():
                    violations = self.video_agent.get_safety_violations()
                    latest_violation = violations[-1] if violations else None
                    if latest_violation:
                        self._trigger_emergency_halt("Video safety violation", latest_violation.description)
                        break
                
                # Sleep for monitoring interval (check every 30 seconds)
                time.sleep(30)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                self._trigger_emergency_halt("Monitoring system error", str(e))
                break
    
    def _trigger_emergency_halt(self, reason, details):
        """Emergency halt procedure for safety violations"""
        print("\n" + "="*60)
        print("🚨 EMERGENCY HALT TRIGGERED 🚨")
        print("="*60)
        print(f"Reason: {reason}")
        print(f"Details: {details}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Stop experiment
        self.state.exp_status = "halted"
        self.monitoring_active = False
        
        # Emergency shutdown procedures
        print("\nInitiating emergency shutdown procedures...")
        
        # Emergency shutdown all lab instruments
        if hasattr(self.lab_agent, 'emergency_shutdown_all'):
            shutdown_count = self.lab_agent.emergency_shutdown_all()
            print(f"✓ Emergency shutdown completed: {shutdown_count} instruments stopped")
        else:
            # Fallback to individual shutdown
            if hasattr(self.lab_agent, 'instruments'):
                for instrument in list(self.lab_agent.instruments.keys()):
                    if self.lab_agent.is_on(instrument):
                        self.lab_agent.turn_off(instrument)
                        print(f"✓ {instrument} turned off")
        
        # Stop video monitoring
        if self.video_agent.is_monitoring:
            video_stop = self.video_agent.stop_monitoring()
            print(f"✓ Video monitoring stopped: {video_stop['message']}")
        
        # Log emergency halt
        try:
            import wandb
            def safe_wandb_log(data: dict):
                """Safely log to wandb, handling cases where wandb is not initialized"""
                try:
                    wandb.log(data)
                except wandb.errors.UsageError:
                    try:
                        wandb.init(project="lab-assistant-agents", mode="disabled")
                        wandb.log(data)
                    except Exception:
                        pass
                except Exception:
                    pass
            
            safe_wandb_log({
                'emergency_halt': {
                    'reason': reason,
                    'details': details,
                    'timestamp': datetime.now().isoformat(),
                    'experiment_status': self.state.exp_status
                }
            })
        except ImportError:
            pass
        
        print("\n⚠️ EXPERIMENT HALTED - MANUAL INTERVENTION REQUIRED ⚠️")
        print("="*60 + "\n")
        
        # Raise exception to stop workflow
        raise RuntimeError(f"Emergency halt triggered: {reason} - {details}")

    ## DO NOT MODIFY BELOW THIS !!! 

    @listen(initialize_overnight_monitoring)
    @weave.op()
    def calculate_nabh4_amount(self):
        """Calculate amount of NaBH4 (10 eq. relative to gold)"""
        calc_result = calculate_nabh4_amount(self.state.mass_gold)
        
        print(f"\nCalculation for NaBH4:")
        print(f"Gold mass: {self.state.mass_gold:.4f}g")
        print(f"Moles of gold: {calc_result['moles_gold']:.6f} mol")
        print(f"Moles of NaBH4 needed (10 eq): {calc_result['moles_nabh4']:.6f} mol")
        print(f"Mass of NaBH4 needed: {calc_result['mass_nabh4_g']:.4f}g")
        
        return calc_result['mass_nabh4_g']

    @listen(calculate_nabh4_amount)
    @weave.op()
    def weigh_nabh4(self):
        mass_needed = self.calculate_nabh4_amount()
        prompt = f"Weigh NaBH4 (~{mass_needed:.3f}g) -- record mass"
        self.state.mass_nabh4 = self.data_agent.record_data(prompt)
        print(f"NaBH4 mass recorded: {self.state.mass_nabh4}g")
        self.update_step()

    @listen(weigh_nabh4)
    @weave.op()
    def measure_nanopure_cold(self):
        self.state.volume_nanopure_cold = self.data_agent.record_data("Measure ice-cold Nanopure water (7mL) -- record vol")
        print(f"Cold Nanopure Volume recorded: {self.state.volume_nanopure_cold}mL")
        self.update_step()
    
    @listen(measure_nanopure_cold)
    @weave.op()
    def weigh_final(self):
        self.state.mass_final = self.data_agent.record_data("Weigh final Au₂₅ nanoparticles -- record mass")
        print(f"Nanoparticle mass recorded: {self.state.mass_final}g")
        self.update_step()

    @listen(weigh_final)
    @weave.op()
    def calculate_percent_yield(self):
        """Calculate percent yield of the experiment based on the initial HAuCl4 content"""
        calc_result = calculate_percent_yield(self.state.mass_gold, self.state.mass_final)
        
        print(f"\nPercent Yield Calculation:")
        print(f"Starting HAuCl4·3H2O: {calc_result['starting_mass_g']:.4f}g")
        print(f"Gold content in starting material: {calc_result['gold_content_g']:.4f}g")
        print(f"Actual yield (Au₂₅ nanoparticles): {calc_result['actual_yield_g']:.4f}g")
        print(f"Percent yield: {calc_result['percent_yield']:.2f}%")
        
        return calc_result['percent_yield']

    @listen(calculate_percent_yield)
    @weave.op()
    def finalize_experiment(self):
        self.state.exp_status = "complete"
        percent_yield = self.calculate_percent_yield()
        
        # Stop overnight monitoring if active
        if hasattr(self, 'monitoring_active') and self.monitoring_active:
            print("Stopping overnight monitoring system...")
            self.monitoring_active = False
            if hasattr(self, 'monitoring_thread'):
                self.monitoring_thread.join(timeout=2.0)
        
        # Stop video monitoring and recording
        if hasattr(self.video_agent, 'is_monitoring') and self.video_agent.is_monitoring:
            stop_result = self.video_agent.stop_monitoring()
            print(f"Video monitoring stopped: {stop_result['message']}")
        
        # Get video monitoring summary
        video_summary = self.video_agent.get_event_summary()
        
        print("\n" + "="*50)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"\nFinal Experiment Summary:")
        print(f"- Total steps completed: {self.state.step_num}")
        print(f"- Gold compound mass: {self.state.mass_gold}g")
        print(f"- TOAB mass: {self.state.mass_toab}g")
        print(f"- Sulfur compound mass: {self.state.mass_sulfur}g")
        print(f"- NaBH4 mass: {self.state.mass_nabh4}g")
        print(f"- Final nanoparticle mass: {self.state.mass_final}g")
        print(f"- Percent yield: {percent_yield:.2f}%")
        print(f"- Safety status: {self.state.safety_status}")
        print(f"\nVideo Monitoring Summary:")
        print(f"- Frames processed: {video_summary['frames_processed']}")
        print(f"- Events detected: {video_summary['total_events']}")
        print(f"- Safety violations: {video_summary['safety_violations']}")
        if video_summary['event_counts']:
            print("- Event breakdown:")
            for event_type, count in video_summary['event_counts'].items():
                print(f"  * {event_type}: {count}")
        print("\n" + "="*50)
    '''
    def measure_solvent(self):
        self.state.volume_solvent = self.data_agent.record_data("Measure water -- record vol")
        print(f"Solvent volume recorded: {self.state.volume_solvent}mL")
    @listen(measure_solvent)
    '''

    @weave.op()
    def control_lab_instruments(self):
        self.lab_agent.turn_on("centrifuge")
        self.lab_agent.turn_on("UV-Vis")
        print("Lab instruments turned on.")

    @listen(control_lab_instruments)
    @weave.op()
    def monitor_safety(self):
        self.safety_agent.monitor_parameters()
        if self.safety_agent.is_safe():
            print("Safety status: Safe")
        else:
            self.state.safety_status = "unsafe"
            self.safety_agent.notify_scientist()
            print("Safety status: Unsafe! Notifying scientist.")

@weave.op()
def kickoff():
    experiment_flow = ExperimentFlow()
    experiment_flow.kickoff()

if __name__ == "__main__":
    kickoff()
