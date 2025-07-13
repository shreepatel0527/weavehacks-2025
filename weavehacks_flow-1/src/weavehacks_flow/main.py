#!/usr/bin/env python
from random import randint
from pydantic import BaseModel
from crewai.flow import Flow, listen, start
import weave
from weavehacks_flow.agents.data_collection_agent import DataCollectionAgent
from weavehacks_flow.agents.lab_control_agent import LabControlAgent
from weavehacks_flow.agents.safety_monitoring_agent import SafetyMonitoringAgent
from weavehacks_flow.crews.data_collection_crew.data_collection_crew import DataCollectionCrew
from weavehacks_flow.crews.lab_control_crew.lab_control_crew import LabControlCrew
from weavehacks_flow.crews.safety_monitoring_crew.safety_monitoring_crew import SafetyMonitoringCrew
from weavehacks_flow.utils.chemistry_calculations import (
    calculate_sulfur_amount,
    calculate_nabh4_amount,
    calculate_percent_yield
)

# Initialize W&B Weave
weave.init('weavehacks-lab-assistant')

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

    @weave.op()
    def update_step(self):
        self.state.step_num += 1
        print(f"Step {self.state.step_num} completed.")
    
    # need to add a step prompting method 

    @listen(initialize_experiment)
    @weave.op()
    def weigh_gold(self):
        self.state.mass_gold = self.data_agent.record_data("Weigh HAuCl₄·3H₂O (0.1576g) -- record mass")
        print(f"Gold mass recorded: {self.state.mass_gold}g")
        self.update_step()

    @listen(weigh_gold)
    @weave.op()
    def measure_nanopure_rt(self):
        self.state.volume_nanopure_rt = self.data_agent.record_data("Measure water (10mL) -- record vol")
        print(f"Room Temp Nanopure Volume recorded: {self.state.volume_nanopure_rt}mL")
        self.update_step()

    @listen(measure_nanopure_rt)
    @weave.op()
    def weigh_toab(self):
        self.state.mass_toab = self.data_agent.record_data("Weigh TOAB (~0.25g) -- record mass")
        print(f"TOAB mass recorded: {self.state.mass_toab}g")
        self.update_step()

    @listen(weigh_toab)
    @weave.op()
    def measure_toluene(self):
        self.state.volume_toluene = self.data_agent.record_data("Measure toluene (10mL) -- record vol")
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

    @listen(weigh_sulfur)
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