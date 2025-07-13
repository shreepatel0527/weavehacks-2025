#!/usr/bin/env python
from random import randint
from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from weavehacks_flow.agents.data_collection_agent import DataCollectionAgent
from weavehacks_flow.agents.lab_control_agent import LabControlAgent
from weavehacks_flow.agents.safety_monitoring_agent import SafetyMonitoringAgent
from weavehacks_flow.crews.data_collection_crew.data_collection_crew import DataCollectionCrew
from weavehacks_flow.crews.lab_control_crew.lab_control_crew import LabControlCrew
from weavehacks_flow.crews.safety_monitoring_crew.safety_monitoring_crew import SafetyMonitoringCrew

class ExperimentState(BaseModel):
    mass_gold: float = 0.0
    mass_sulfur: float = 0.0
    volume_solvent: float = 0.0
    observations: str = ""
    safety_status: str = "safe"

class ExperimentFlow(Flow[ExperimentState]):
    data_agent = DataCollectionAgent()
    lab_agent = LabControlAgent()
    safety_agent = SafetyMonitoringAgent()

    @start()
    def initialize_experiment(self):
        print("Initializing experiment...")
        # self.state = ExperimentState()

    @listen(initialize_experiment)
    def weigh_gold(self):
        self.state.mass_gold = self.data_agent.record_data("Weigh HAuCl₄·3H₂O (0.1576g) -- record mass")
        print(f"Gold mass recorded: {self.state.mass_gold}g")

    @listen(weigh_gold)
    def weigh_sulfur(self):
        self.state.mass_sulfur = self.data_agent.record_data("Weigh TOAB (~0.25g) -- record mass")
        print(f"Sulfur mass recorded: {self.state.mass_sulfur}g")

    @listen(weigh_sulfur)
    def measure_solvent(self):
        self.state.volume_solvent = self.data_agent.record_data("Measure water -- record vol")
        print(f"Solvent volume recorded: {self.state.volume_solvent}mL")

    @listen(measure_solvent)
    def control_lab_instruments(self):
        self.lab_agent.turn_on("centrifuge")
        self.lab_agent.turn_on("UV-Vis")
        print("Lab instruments turned on.")

    @listen(control_lab_instruments)
    def monitor_safety(self):
        self.safety_agent.monitor_parameters()
        if self.safety_agent.is_safe():
            print("Safety status: Safe")
        else:
            self.state.safety_status = "unsafe"
            self.safety_agent.notify_scientist()
            print("Safety status: Unsafe! Notifying scientist.")

def kickoff():
    experiment_flow = ExperimentFlow()
    experiment_flow.kickoff()

if __name__ == "__main__":
    kickoff()