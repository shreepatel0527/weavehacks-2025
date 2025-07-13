# filepath: /weavehacks_flow/weavehacks_flow/src/weavehacks_flow/crews/data_collection_crew/data_collection_crew.py
from crewai import Crew
from weavehacks_flow.agents.data_collection_agent import DataCollectionAgent

class DataCollectionCrew(Crew):
    def __init__(self):
        self.data_agent = DataCollectionAgent()

    def collect_data(self, prompt):
        return self.data_agent.record_data(prompt)

    def clarify_reagent(self, reagent):
        return self.data_agent.clarify_reagent(reagent)

    def manage_data_collection(self):
        # Implement the logic for managing data collection tasks
        pass