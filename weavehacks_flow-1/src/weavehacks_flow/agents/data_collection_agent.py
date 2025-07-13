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

class DataCollectionAgent:
    @weave.op()
    def record_data(self, prompt):
        # Simulate data collection based on user input
        value = float(input(f"{prompt}: "))
        # Log the data collection to W&B
        safe_wandb_log({'data_collected': {'prompt': prompt, 'value': value}})
        return value

    @weave.op()
    def clarify_reagent(self):
        # Prompt the user for clarification on reagents
        clarification = input("Please clarify the reagent information: ")
        safe_wandb_log({'clarification_requested': clarification})
        return clarification

class LabControlAgent:
    def turn_on(self, instrument):
        print(f"{instrument} is now ON.")

    def turn_off(self, instrument):
        print(f"{instrument} is now OFF.")

class SafetyMonitoringAgent:
    def monitor_parameters(self):
        # Simulate monitoring safety parameters
        print("Monitoring safety parameters...")

    def is_safe(self):
        # Simulate safety check
        return True

    def notify_scientist(self):
        print("Safety alert! Please check the instruments.")