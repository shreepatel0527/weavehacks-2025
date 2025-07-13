#!/usr/bin/env python3
"""
Experiment Configuration Module
Manages experiment-specific parameters, temperature ranges, pressure ranges, and protocols.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from enum import Enum

class ExperimentType(Enum):
    GOLD_NANOPARTICLE_SYNTHESIS = "gold_nanoparticle_synthesis"
    SILVER_NANOPARTICLE_SYNTHESIS = "silver_nanoparticle_synthesis"
    CATALYSIS_REACTION = "catalysis_reaction"
    CRYSTALLIZATION = "crystallization"
    CUSTOM = "custom"

@dataclass
class ParameterRange:
    min_safe: float
    max_safe: float
    warning_buffer: float
    critical_buffer: float
    units: str
    description: str = ""

@dataclass
class ExperimentConfig:
    name: str
    experiment_type: ExperimentType
    description: str
    temperature_range: ParameterRange
    pressure_range: ParameterRange
    duration_hours: float = 0.0
    special_notes: str = ""

class ExperimentManager:
    """
    Manages experiment configurations with specific temperature and pressure ranges
    for different types of lab procedures.
    """
    
    def __init__(self, config_file: str = "experiment_configs.json"):
        self.config_file = config_file
        self.experiments = self._load_experiment_configs()
        self.current_experiment: Optional[ExperimentConfig] = None
    
    def _load_experiment_configs(self) -> Dict[str, ExperimentConfig]:
        """Load experiment configurations from file."""
        default_experiments = {
            "gold_nanoparticle_room_temp": ExperimentConfig(
                name="Gold Nanoparticle Synthesis (Room Temp)",
                experiment_type=ExperimentType.GOLD_NANOPARTICLE_SYNTHESIS,
                description="Standard gold nanoparticle synthesis at room temperature",
                temperature_range=ParameterRange(
                    min_safe=20.0, max_safe=25.0,
                    warning_buffer=2.0, critical_buffer=5.0,
                    units="°C",
                    description="Room temperature synthesis"
                ),
                pressure_range=ParameterRange(
                    min_safe=100.0, max_safe=102.0,
                    warning_buffer=1.0, critical_buffer=3.0,
                    units="kPa",
                    description="Atmospheric pressure"
                ),
                duration_hours=1.5,
                special_notes="Monitor color change: deep red → faint yellow → clear"
            ),
            "gold_nanoparticle_ice_bath": ExperimentConfig(
                name="Gold Nanoparticle Synthesis (Ice Bath)",
                experiment_type=ExperimentType.GOLD_NANOPARTICLE_SYNTHESIS,
                description="Gold nanoparticle synthesis with ice bath cooling",
                temperature_range=ParameterRange(
                    min_safe=0.0, max_safe=5.0,
                    warning_buffer=1.0, critical_buffer=3.0,
                    units="°C",
                    description="Ice bath temperature for 30 min cooling"
                ),
                pressure_range=ParameterRange(
                    min_safe=100.0, max_safe=102.0,
                    warning_buffer=1.0, critical_buffer=3.0,
                    units="kPa",
                    description="Atmospheric pressure"
                ),
                duration_hours=1.0,
                special_notes="Cool to 0°C in ice bath over 30 min with stirring"
            ),
            "gold_nanoparticle_stirring": ExperimentConfig(
                name="Gold Nanoparticle Synthesis (Vigorous Stirring)",
                experiment_type=ExperimentType.GOLD_NANOPARTICLE_SYNTHESIS,
                description="High-energy mixing phase at elevated temperature",
                temperature_range=ParameterRange(
                    min_safe=25.0, max_safe=35.0,
                    warning_buffer=3.0, critical_buffer=8.0,
                    units="°C",
                    description="Elevated temperature during vigorous stirring"
                ),
                pressure_range=ParameterRange(
                    min_safe=100.0, max_safe=103.0,
                    warning_buffer=1.5, critical_buffer=4.0,
                    units="kPa",
                    description="Slightly elevated pressure from stirring"
                ),
                duration_hours=0.25,
                special_notes="Stir vigorously (~1100 rpm) for ~15 min"
            ),
            "overnight_stirring": ExperimentConfig(
                name="Overnight Stirring Under N₂",
                experiment_type=ExperimentType.GOLD_NANOPARTICLE_SYNTHESIS,
                description="Long-term stirring under nitrogen atmosphere",
                temperature_range=ParameterRange(
                    min_safe=18.0, max_safe=28.0,
                    warning_buffer=2.0, critical_buffer=5.0,
                    units="°C",
                    description="Stable temperature for overnight reaction"
                ),
                pressure_range=ParameterRange(
                    min_safe=99.0, max_safe=103.0,
                    warning_buffer=2.0, critical_buffer=5.0,
                    units="kPa",
                    description="Pressure with N₂ atmosphere"
                ),
                duration_hours=12.0,
                special_notes="Stir overnight under N₂ atmosphere - REQUIRES SAFETY MONITORING"
            ),
            "custom_experiment": ExperimentConfig(
                name="Custom Experiment",
                experiment_type=ExperimentType.CUSTOM,
                description="User-defined experiment parameters",
                temperature_range=ParameterRange(
                    min_safe=15.0, max_safe=40.0,
                    warning_buffer=3.0, critical_buffer=7.0,
                    units="°C",
                    description="Customizable temperature range"
                ),
                pressure_range=ParameterRange(
                    min_safe=95.0, max_safe=110.0,
                    warning_buffer=2.0, critical_buffer=5.0,
                    units="kPa",
                    description="Customizable pressure range"
                ),
                duration_hours=1.0,
                special_notes="User-defined experimental conditions"
            )
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                    # Convert loaded data back to ExperimentConfig objects
                    # This is simplified - in practice you'd want proper deserialization
                    print(f"Loaded experiment configurations from {self.config_file}")
            except Exception as e:
                print(f"Failed to load config file: {e}. Using defaults.")
        else:
            self._save_default_configs(default_experiments)
        
        return default_experiments
    
    def _save_default_configs(self, experiments: Dict[str, ExperimentConfig]):
        """Save default configurations to file."""
        config_data = {
            exp_id: asdict(config) for exp_id, config in experiments.items()
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            print(f"Saved default experiment configurations to {self.config_file}")
        except Exception as e:
            print(f"Failed to save default config: {e}")
    
    def get_experiment_list(self) -> List[str]:
        """Get list of available experiment names."""
        return list(self.experiments.keys())
    
    def get_experiment_config(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """Get specific experiment configuration."""
        return self.experiments.get(experiment_id)
    
    def set_current_experiment(self, experiment_id: str) -> bool:
        """Set the current active experiment."""
        if experiment_id in self.experiments:
            self.current_experiment = self.experiments[experiment_id]
            print(f"Set current experiment to: {self.current_experiment.name}")
            return True
        return False
    
    def get_current_experiment(self) -> Optional[ExperimentConfig]:
        """Get the currently active experiment."""
        return self.current_experiment
    
    def get_safety_parameters_for_current_experiment(self) -> Optional[Dict]:
        """Get safety parameters for the current experiment."""
        if not self.current_experiment:
            return None
        
        return {
            "temperature": {
                "min_safe": self.current_experiment.temperature_range.min_safe,
                "max_safe": self.current_experiment.temperature_range.max_safe,
                "warning_buffer": self.current_experiment.temperature_range.warning_buffer,
                "critical_buffer": self.current_experiment.temperature_range.critical_buffer,
                "units": self.current_experiment.temperature_range.units
            },
            "pressure": {
                "min_safe": self.current_experiment.pressure_range.min_safe,
                "max_safe": self.current_experiment.pressure_range.max_safe,
                "warning_buffer": self.current_experiment.pressure_range.warning_buffer,
                "critical_buffer": self.current_experiment.pressure_range.critical_buffer,
                "units": self.current_experiment.pressure_range.units
            }
        }
    
    def create_custom_experiment(self, name: str, temp_min: float, temp_max: float, 
                                pressure_min: float, pressure_max: float,
                                duration: float = 1.0, notes: str = "") -> str:
        """Create a custom experiment configuration."""
        experiment_id = f"custom_{name.lower().replace(' ', '_')}"
        
        custom_config = ExperimentConfig(
            name=name,
            experiment_type=ExperimentType.CUSTOM,
            description=f"Custom experiment: {name}",
            temperature_range=ParameterRange(
                min_safe=temp_min, max_safe=temp_max,
                warning_buffer=2.0, critical_buffer=5.0,
                units="°C",
                description="Custom temperature range"
            ),
            pressure_range=ParameterRange(
                min_safe=pressure_min, max_safe=pressure_max,
                warning_buffer=1.5, critical_buffer=4.0,
                units="kPa",
                description="Custom pressure range"
            ),
            duration_hours=duration,
            special_notes=notes
        )
        
        self.experiments[experiment_id] = custom_config
        return experiment_id

def main():
    """Test the experiment manager."""
    manager = ExperimentManager()
    
    print("🧪 Available Experiments:")
    for exp_id in manager.get_experiment_list():
        config = manager.get_experiment_config(exp_id)
        print(f"  {exp_id}: {config.name}")
        print(f"    Temperature: {config.temperature_range.min_safe}-{config.temperature_range.max_safe}{config.temperature_range.units}")
        print(f"    Pressure: {config.pressure_range.min_safe}-{config.pressure_range.max_safe}{config.pressure_range.units}")
        print(f"    Duration: {config.duration_hours} hours")
        print()

if __name__ == "__main__":
    main()