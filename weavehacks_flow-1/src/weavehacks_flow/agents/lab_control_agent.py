import weave
import wandb

def safe_wandb_log(data: dict):
    """Safely log to wandb, handling cases where wandb is not initialized"""
    try:
        wandb.log(data)
    except wandb.errors.UsageError:
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

    @weave.op()
    def turn_on(self, instrument_name):
        self.instruments[instrument_name] = True
        print(f"{instrument_name} turned on.")
        safe_wandb_log({'instrument_control': {'action': 'turn_on', 'instrument': instrument_name}})

    @weave.op()
    def turn_off(self, instrument_name):
        if instrument_name in self.instruments:
            self.instruments[instrument_name] = False
            print(f"{instrument_name} turned off.")
            safe_wandb_log({'instrument_control': {'action': 'turn_off', 'instrument': instrument_name}})
        else:
            print(f"{instrument_name} is not currently on.")

    @weave.op()
    def is_on(self, instrument_name):
        status = self.instruments.get(instrument_name, False)
        safe_wandb_log({'instrument_status': {'instrument': instrument_name, 'is_on': status}})
        return status