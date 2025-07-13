from crewai import Agent
import streamlit as st
from safety_monitoring import sensor_file_reader
# from rohit_prototype import app
from Prototype_1 import claude_interface, gemini_interface, speech_recognition_module

# Try to import whisper for browser audio support
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
# Define the Safety Monitoring Agent
class SafetyMonitoringAgent(Agent):
    def execute(self, input_data):
        """
        Executes the safety monitoring logic.
        """
        print("Executing Safety Monitoring Agent...")
        safety_status = sensor_file_reader.monitor_sensor_file()
        print(f"Safety Monitoring Result: {safety_status}")
        return safety_status
"""
# Define the Data Collection Agent
class DataCollectionAgent(Agent):
     def execute(self, input_data):
        '''
        Executes the data collection logic.
        '''
        print("Executing Data Collection Agent...")
        data_status = app(input_data)
        print(f"Data Collection Result: {data_status}")
        return data_status
"""
# Define the Orchestrator Agent
class OrchestratorAgent(Agent):
    def orchestrate(self):
        """
        Orchestrates the flow between the Safety Monitoring Agent and the Data Collection Agent.
        """
        print("Starting orchestration...")

        # Step 1: Run the Safety Monitoring Agent
        safety_agent = SafetyMonitoringAgent()
        safety_result = safety_agent.execute(input_data={})
        
        # Check if safety is ensured before proceeding
        if safety_result.get("status") == "safe":
            print("Safety confirmed. Proceeding to data collection...")
            
            # Step 2: Run the Data Collection Agent
            data_agent = DataCollectionAgent()
            data_result = data_agent.execute(input_data={})
            
            print("Orchestration completed successfully.")
            return {"safety_result": safety_result, "data_result": data_result}
        else:
            print("Safety concern detected. Halting orchestration.")
            return {"safety_result": safety_result, "data_result": None}

# Main entry point
if __name__ == "__main__":
    # Initialize Streamlit session state
    st.set_page_config(
        page_title="AI Orchestrator",
        page_icon="🤖",
        layout="centered"
    )
    '''
    # Initialize session state variables
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'claude' not in st.session_state:
        st.session_state.claude = claude_interface.ClaudeInterface()
    if 'gemini' not in st.session_state:
        st.session_state.gemini = gemini_interface.GeminiInterface()
    if 'speech_recognizer' not in st.session_state:
        st.session_state.speech_recognizer = speech_recognition_module.SpeechRecognizer(model_size="base")
    if 'whisper_model' not in st.session_state and WHISPER_AVAILABLE:
        with st.spinner("Loading Whisper model..."):
            st.session_state.whisper_model = whisper.load_model("base")
    if 'audio_input_active' not in st.session_state:
        st.session_state.audio_input_active = False
    if 'ai_model' not in st.session_state:
        st.session_state.ai_model = "Claude"
    if 'audio_method' not in st.session_state:
        st.session_state.audio_method = "System Microphone""
    '''

    # Run the orchestrator
    orchestrator = OrchestratorAgent()
    results = orchestrator.orchestrate()
    print(f"Final Results: {results}")