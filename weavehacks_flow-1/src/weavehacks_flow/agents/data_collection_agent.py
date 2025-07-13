import weave

class DataCollectionAgent:
    @weave.op()
    def record_data(self, prompt):
        # Simulate data collection based on user input
        value = float(input(f"{prompt}: "))
        # Log the data collection to W&B
        weave.log({'data_collected': {'prompt': prompt, 'value': value}})
        return value

    @weave.op()
    def clarify_reagent(self):
        # Prompt the user for clarification on reagents
        clarification = input("Please clarify the reagent information: ")
        weave.log({'clarification_requested': clarification})
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