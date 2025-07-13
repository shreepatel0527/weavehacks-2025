from crewai import Crew
from weavehacks_flow.agents.safety_monitoring_agent import SafetyMonitoringAgent

class SafetyMonitoringCrew:
    def __init__(self):
        self.agent = SafetyMonitoringAgent()

    def monitor_safety(self):
        print("Monitoring safety parameters...")
        self.agent.monitor_parameters()
        if self.agent.is_safe():
            print("All safety parameters are within acceptable limits.")
        else:
            print("Safety alert! Parameters exceed safe limits.")
            self.agent.notify_scientist()

    def integrate_with_flow(self, flow):
        flow.listen(self.monitor_safety)

# The SafetyMonitoringCrew class oversees safety monitoring tasks using the SafetyMonitoringAgent.
# It integrates safety checks into the main flow of the application.