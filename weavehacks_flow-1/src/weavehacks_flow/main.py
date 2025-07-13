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

class ExperimentFlow(Flow[ExperimentState]):
    data_agent = DataCollectionAgent()
    lab_agent = LabControlAgent()
    safety_agent = SafetyMonitoringAgent()

    @start()
    def initialize_experiment(self):
        print("Initializing experiment...")

    def update_step(self):
        self.state.step_num += 1
        print(f"Step {self.state.step_num} completed.")
    
    # need to add a step prompting method 

    @listen(initialize_experiment)
    def weigh_gold(self):
        self.state.mass_gold = self.data_agent.record_data("Weigh HAuCl₄·3H₂O (0.1576g) -- record mass")
        print(f"Gold mass recorded: {self.state.mass_gold}g")

    @listen(weigh_gold)
    def measure_nanopure_rt(self):
        self.state.volume_nanopure_rt = self.data_agent.record_data("Measure water (10mL) -- record vol")
        print(f"Room Temp Nanopure Volume recorded: {self.state.volume_nanopure_rt}mL")

    @listen(measure_nanopure_rt)
    def weigh_toab(self):
        self.state.mass_toab = self.data_agent.record_data("Weigh TOAB (~0.25g) -- record mass")
        print(f"TOAB mass recorded: {self.state.mass_toab}g")

    @listen(weigh_toab)
    def toluene(self):
        self.state.volume_toluene = self.data_agent.record_data("Measure water (10mL) -- record vol")
        print(f"Toluene volume recorded: {self.state.volume_toluene}mL")

    @listen(toluene)
    # need to do: a calculate amount of sulfur method 

    # need a lot of userprompting on the steps we are doing here 

    def weigh_sulfur(self):
        self.state.mass_sulfur = self.data_agent.record_data("Weigh Sulfur (~0.05g) -- record mass")
        print(f"Sulfur mass recorded: {self.state.mass_sulfur}g")

    @listen(weigh_sulfur)
    def weigh_nabh4(self):
        self.state.mass_nabh4 = self.data_agent.record_data("Weigh NaBH4 (~0.7g) -- record mass")
        print(f"NaBH4 mass recorded: {self.state.mass_nabh4}g")

    @listen(weigh_nabh4)
    def measure_nanopure_cold(self):
        self.state.volume_nanopure_cold = self.data_agent.record_data("Measure water (10mL) -- record vol")
        print(f"Cold Nanopure Volume recorded: {self.state.volume_nanopure_cold}mL")
    
    @listen(measure_nanopure_cold)
    def weigh_final(self):
        self.state.mass_final = self.data_agent.record_data("Weigh Final (~0.05g) -- record mass")
        print(f"Nanoparticle mass recorded: {self.state.mass_final}g")

    @listen(weigh_final)
    def finalize_experiment(self):
        self.state.exp_status = "complete"
        print("Experiment completed successfully.")
        print(f"Final state: {self.state}")
    '''
    def measure_solvent(self):
        self.state.volume_solvent = self.data_agent.record_data("Measure water -- record vol")
        print(f"Solvent volume recorded: {self.state.volume_solvent}mL")
    @listen(measure_solvent)
    '''

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