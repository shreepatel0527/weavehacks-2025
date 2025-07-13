try:
    import weave
except ImportError:
    weave = None

try:
    import wandb
except ImportError:
    wandb = None
from .voice_recognition_agent import SpeechRecognizerAgent
import re
import csv

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
    def __init__(self):
        self.voice_agent = SpeechRecognizerAgent()

    def record_data(self, prompt, use_voice=False, experiment_state=None, state_key=None):
        """
        Records data based on the given prompt and updates the ExperimentState.
        Args:
            prompt (str): The prompt describing the data to record.
            use_voice (bool): Whether to use voice input for recording data.
            experiment_state (ExperimentState): The ExperimentState to update.
            state_key (str): The key in the ExperimentState to update.
        Returns:
            float: The recorded data.
        """
        if use_voice:
            print(prompt)
            
            # Check if voice agent is properly initialized
            if not self.voice_agent.audio_initialized:
                print("⚠️  Voice recognition not available. Audio system not properly initialized.")
                print("🔧 To troubleshoot, run: python diagnose_audio.py")
                print("📝 Falling back to manual input...")
                use_voice = False
            else:
                try:
                    success, transcribed_text = self.voice_agent.record_and_transcribe(duration=5.0)
                    if success:
                        print(f"Transcribed Data: {transcribed_text}")
                        extracted_value = self._extract_digit(transcribed_text)
                        if experiment_state is not None and state_key is not None:
                            setattr(experiment_state, state_key, extracted_value)
                        return extracted_value
                    else:
                        print(f"❌ Voice Recognition Error: {transcribed_text}")
                        print("📝 Falling back to manual input...")
                        use_voice = False
                except RuntimeError as e:
                    print(f"❌ Audio System Error: {e}")
                    if "device -1" in str(e).lower():
                        print("🔧 Quick fix: Run 'python diagnose_audio.py' for detailed troubleshooting")
                    print("📝 Falling back to manual input...")
                    use_voice = False
                except Exception as e:
                    print(f"❌ Unexpected voice recognition error: {e}")
                    print("📝 Falling back to manual input...")
                    use_voice = False
        
        if not use_voice:
            print(prompt)
            recorded_data = input("Enter the data: ")
            extracted_value = self._extract_digit(recorded_data)
            if experiment_state is not None and state_key is not None:
                setattr(experiment_state, state_key, extracted_value)
            return extracted_value

    def _extract_digit(self, text):
        """
        Extracts the first float or integer from a text string.
        Args:
            text (str): The input text.
        Returns:
            float: The extracted number, or None if no number is found.
        """
        match = re.search(r"[-+]?\d*\.\d+|\d+", text)
        if match:
            return float(match.group())
        return None

    def state_to_table(self, experiment_state, output_csv_path):
        """
        Converts the ExperimentState into the format of quantitative_observation.csv.
        Args:
            experiment_state (ExperimentState): The ExperimentState to convert.
            output_csv_path (str): The path to save the CSV file.
        """
        # Define the mapping of state keys to CSV rows
        csv_rows = [
            ["Substance", "Mass (g)", "Volume (mL)"],
            ["HAuCl₄·3H₂O", experiment_state.mass_gold, None],
            ["Water (for gold)", None, experiment_state.volume_nanopure_rt],
            ["TOAB", experiment_state.mass_toab, None],
            ["Toluene", None, experiment_state.volume_toluene],
            ["PhCH₂CH₂SH", experiment_state.mass_sulfur, None],
            ["NaBH₄", experiment_state.mass_nabh4, None],
            ["Ice-cold Nanopure water (for NaBH₄)", None, experiment_state.volume_nanopure_cold],
            ["Final mass of nanoparticles", experiment_state.mass_final, None],
        ]

        # Write to CSV
        with open(output_csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(csv_rows)