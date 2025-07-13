#!/usr/bin/env python
"""
Test version of main.py with simulated inputs to verify stability
"""
from random import randint
from pydantic import BaseModel
import weave

# Import all the modules
from agents.data_collection_agent import DataCollectionAgent
from agents.lab_control_agent import LabControlAgent
from agents.safety_monitoring_agent import SafetyMonitoringAgent
from utils.chemistry_calculations import (
    calculate_sulfur_amount,
    calculate_nabh4_amount,
    calculate_percent_yield
)

# Initialize W&B Weave (optional - will work without API key)
try:
    weave.init('weavehacks-lab-assistant-test')
    print("Weave initialized successfully")
except Exception as e:
    print(f"Weave initialization failed (this is optional): {e}")
    print("Continuing without W&B logging...")

class MockDataCollectionAgent:
    """Mock agent that simulates data collection without user input"""
    
    def __init__(self):
        self.mock_values = {
            "gold": 0.1576,
            "water": 10.0,
            "toab": 0.25,
            "toluene": 10.0,
            "sulfur": 0.065,
            "nabh4": 0.0151,
            "water_cold": 7.0,
            "final": 0.08
        }
        self.call_count = 0
    
    @weave.op()
    def record_data(self, prompt):
        # Simulate data collection with predetermined values
        self.call_count += 1
        
        # Determine which measurement based on prompt content
        if "HAuCl" in prompt or "gold" in prompt.lower():
            value = self.mock_values["gold"]
        elif "water" in prompt and "cold" in prompt:
            value = self.mock_values["water_cold"]
        elif "water" in prompt or "nanopure" in prompt.lower():
            value = self.mock_values["water"]
        elif "TOAB" in prompt or "toab" in prompt.lower():
            value = self.mock_values["toab"]
        elif "toluene" in prompt.lower():
            value = self.mock_values["toluene"]
        elif "PhCH₂CH₂SH" in prompt or "sulfur" in prompt.lower():
            value = self.mock_values["sulfur"]
        elif "NaBH4" in prompt or "nabh4" in prompt.lower():
            value = self.mock_values["nabh4"]
        elif "final" in prompt.lower() or "nanoparticle" in prompt.lower():
            value = self.mock_values["final"]
        else:
            # Default value
            value = 1.0
        
        print(f"Mock input for '{prompt}': {value}")
        return value

class ExperimentState(BaseModel):
    # experiment tracking
    step_num: int = 0
    exp_status: str = "not started"
    # solids mass
    mass_gold: float = 0.0
    mass_toab: float = 0.0
    mass_sulfur: float = 0.0
    mass_nabh4: float = 0.0
    mass_final: float = 0.0
    # liquids volume
    volume_toluene: float = 0.0
    volume_nanopure_rt: float = 0.0
    volume_nanopure_cold: float = 0.0
    observations: str = ""
    # safety status
    safety_status: str = "safe"

class TestExperimentFlow:
    def __init__(self):
        self.state = ExperimentState()
        self.data_agent = MockDataCollectionAgent()  # Use mock agent
        self.lab_agent = LabControlAgent()
        self.safety_agent = SafetyMonitoringAgent()
        self._setup_workflow()
    
    def _setup_workflow(self):
        """Set up the workflow execution order"""
        self.workflow_steps = [
            self.initialize_experiment,
            self.weigh_gold,
            self.measure_nanopure_rt,
            self.weigh_toab,
            self.measure_toluene,
            self.calculate_sulfur_amount,
            self.weigh_sulfur,
            self.calculate_nabh4_amount,
            self.weigh_nabh4,
            self.measure_nanopure_cold,
            self.weigh_final,
            self.calculate_percent_yield,
            self.finalize_experiment
        ]
    
    def kickoff(self):
        """Execute the workflow steps in sequence"""
        for method in self.workflow_steps:
            method()

    def initialize_experiment(self):
        print("Initializing experiment...")

    @weave.op()
    def update_step(self):
        self.state.step_num += 1
        print(f"Step {self.state.step_num} completed.")
    
    @weave.op()
    def weigh_gold(self):
        self.state.mass_gold = self.data_agent.record_data("Weigh HAuCl₄·3H₂O (0.1576g) -- record mass")
        print(f"Gold mass recorded: {self.state.mass_gold}g")
        self.update_step()

    @weave.op()
    def measure_nanopure_rt(self):
        self.state.volume_nanopure_rt = self.data_agent.record_data("Measure water (10mL) -- record vol")
        print(f"Room Temp Nanopure Volume recorded: {self.state.volume_nanopure_rt}mL")
        self.update_step()

    @weave.op()
    def weigh_toab(self):
        self.state.mass_toab = self.data_agent.record_data("Weigh TOAB (~0.25g) -- record mass")
        print(f"TOAB mass recorded: {self.state.mass_toab}g")
        self.update_step()

    @weave.op()
    def measure_toluene(self):
        self.state.volume_toluene = self.data_agent.record_data("Measure toluene (10mL) -- record vol")
        print(f"Toluene volume recorded: {self.state.volume_toluene}mL")
        self.update_step()

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

    @weave.op()
    def weigh_sulfur(self):
        mass_needed = self.calculate_sulfur_amount()
        prompt = f"Weigh PhCH₂CH₂SH (~{mass_needed:.3f}g) -- record mass"
        self.state.mass_sulfur = self.data_agent.record_data(prompt)
        print(f"Sulfur mass recorded: {self.state.mass_sulfur}g")
        self.update_step()

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

    @weave.op()
    def weigh_nabh4(self):
        mass_needed = self.calculate_nabh4_amount()
        prompt = f"Weigh NaBH4 (~{mass_needed:.3f}g) -- record mass"
        self.state.mass_nabh4 = self.data_agent.record_data(prompt)
        print(f"NaBH4 mass recorded: {self.state.mass_nabh4}g")
        self.update_step()

    @weave.op()
    def measure_nanopure_cold(self):
        self.state.volume_nanopure_cold = self.data_agent.record_data("Measure ice-cold Nanopure water (7mL) -- record vol")
        print(f"Cold Nanopure Volume recorded: {self.state.volume_nanopure_cold}mL")
        self.update_step()
    
    @weave.op()
    def weigh_final(self):
        self.state.mass_final = self.data_agent.record_data("Weigh final Au₂₅ nanoparticles -- record mass")
        print(f"Nanoparticle mass recorded: {self.state.mass_final}g")
        self.update_step()

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

@weave.op()
def test_kickoff():
    print("Starting automated test of experiment flow...")
    experiment_flow = TestExperimentFlow()
    experiment_flow.kickoff()
    print("Test completed successfully!")

if __name__ == "__main__":
    test_kickoff()