try:
    import weave
except ImportError:
    weave = None

try:
    import wandb
except ImportError:
    wandb = None

def weave_op():
    """Optional weave decorator that works when weave is not available"""
    def decorator(func):
        if weave is not None:
            return weave.op()(func)
        return func
    return decorator

def safe_wandb_log(data: dict):
    """Safely log to wandb, handling cases where wandb is not initialized"""
    if wandb is None:
        return
    try:
        wandb.log(data)
    except Exception:
        # wandb not initialized, try to initialize minimally
        try:
            wandb.init(project="lab-assistant-agents", mode="disabled")
            wandb.log(data)
        except Exception:
            # If all else fails, just skip logging
            pass
    except Exception:
        # Any other wandb error, skip logging
        pass

class LabControlAgent:
    def __init__(self):
        self.instruments = {}
        self.alert_mode = False
        self.emergency_shutdown_active = False

    @weave_op()
    def turn_on(self, instrument_name):
        if self.emergency_shutdown_active:
            print(f"Cannot turn on {instrument_name}: Emergency shutdown active")
            return False
            
        self.instruments[instrument_name] = True
        
        # Check for special monitoring instruments
        if instrument_name in ["emergency_monitoring", "safety_systems"]:
            self.alert_mode = True
            print(f"Lab control agent: {instrument_name} activated - Alert mode enabled")
        else:
            print(f"{instrument_name} turned on.")
            
        safe_wandb_log({'instrument_control': {'action': 'turn_on', 'instrument': instrument_name}})
        return True

    @weave_op()
    def turn_off(self, instrument_name):
        if instrument_name in self.instruments:
            self.instruments[instrument_name] = False
            print(f"{instrument_name} turned off.")
            safe_wandb_log({'instrument_control': {'action': 'turn_off', 'instrument': instrument_name}})
            return True
        else:
            print(f"{instrument_name} is not currently on.")
            return False

    @weave_op()
    def is_on(self, instrument_name):
        status = self.instruments.get(instrument_name, False)
        safe_wandb_log({'instrument_status': {'instrument': instrument_name, 'is_on': status}})
        return status
    
    @weave_op()
    def emergency_shutdown_all(self):
        """Emergency shutdown of all lab instruments"""
        self.emergency_shutdown_active = True
        shutdown_count = 0
        
        print("\n🚨 INITIATING EMERGENCY SHUTDOWN OF ALL INSTRUMENTS 🚨")
        
        # Turn off all instruments
        for instrument_name in list(self.instruments.keys()):
            if self.instruments[instrument_name]:  # Only turn off if currently on
                self.instruments[instrument_name] = False
                print(f"✓ Emergency shutdown: {instrument_name}")
                shutdown_count += 1
        
        # Log emergency shutdown
        safe_wandb_log({
            'emergency_shutdown': {
                'instruments_shutdown': shutdown_count,
                'timestamp': 'emergency_active'
            }
        })
        
        print(f"Emergency shutdown complete: {shutdown_count} instruments turned off")
        return shutdown_count
    
    @weave_op()
    def reset_emergency_mode(self):
        """Reset emergency shutdown mode (for testing/recovery)"""
        self.emergency_shutdown_active = False
        self.alert_mode = False
        print("Emergency mode reset - Normal operations can resume")
        safe_wandb_log({'lab_control': {'action': 'emergency_reset'}})
    
    def get_status(self):
        """Get current status of lab control agent"""
        return {
            'alert_mode': self.alert_mode,
            'emergency_shutdown_active': self.emergency_shutdown_active,
            'active_instruments': {k: v for k, v in self.instruments.items() if v},
            'total_instruments': len(self.instruments)
        }