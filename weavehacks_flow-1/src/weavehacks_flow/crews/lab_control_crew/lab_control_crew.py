# filepath: /weavehacks_flow/weavehacks_flow/src/weavehacks_flow/crews/lab_control_crew/lab_control_crew.py

from crewai import Crew
from weavehacks_flow.agents.lab_control_agent import LabControlAgent

class LabControlCrew(Crew):
    def __init__(self):
        super().__init__()
        self.lab_agent = LabControlAgent()

    def turn_on_instrument(self, instrument_name):
        self.lab_agent.turn_on(instrument_name)
        print(f"{instrument_name} has been turned on.")

    def turn_off_instrument(self, instrument_name):
        self.lab_agent.turn_off(instrument_name)
        print(f"{instrument_name} has been turned off.")

    def control_instruments(self, commands):
        for command in commands:
            if command['action'] == 'turn_on':
                self.turn_on_instrument(command['instrument'])
            elif command['action'] == 'turn_off':
                self.turn_off_instrument(command['instrument'])