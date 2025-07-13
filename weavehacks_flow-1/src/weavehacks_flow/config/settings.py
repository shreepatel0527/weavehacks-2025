"""
Centralized configuration management for Lab Assistant
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import json
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ChemistryConfig:
    """Chemistry-related configuration"""
    # Molecular weights
    MW_HAuCl4: float = 339.79  # g/mol
    MW_HAuCl4_3H2O: float = 393.83  # g/mol
    MW_TOAB: float = 546.78  # g/mol
    MW_PhCH2CH2SH: float = 138.23  # g/mol
    MW_NaBH4: float = 37.83  # g/mol
    MW_Au: float = 196.97  # g/mol
    
    # Reaction parameters
    DEFAULT_EQUIVALENTS_SULFUR: int = 3
    DEFAULT_EQUIVALENTS_NABH4: int = 10
    DEFAULT_SOLVENT_VOLUME_ML: int = 125
    
    # Concentration
    DEFAULT_TOAB_CONCENTRATION: float = 0.05  # M

@dataclass
class SafetyConfig:
    """Safety monitoring configuration"""
    # Temperature thresholds (Celsius)
    TEMP_MIN: float = 0.0
    TEMP_MAX: float = 30.0
    TEMP_WARNING: float = 25.0
    
    # Pressure thresholds (kPa)
    PRESSURE_MIN: float = 90.0
    PRESSURE_MAX: float = 110.0
    PRESSURE_WARNING: float = 105.0
    
    # Stirring speed (RPM)
    STIRRING_MIN: int = 500
    STIRRING_MAX: int = 1500
    STIRRING_OPTIMAL: int = 1100
    
    # pH thresholds
    PH_MIN: float = 6.0
    PH_MAX: float = 8.0
    PH_OPTIMAL: float = 7.0
    
    # Alert settings
    ALERT_COOLDOWN_SECONDS: int = 60
    MAX_CONSECUTIVE_VIOLATIONS: int = 3
    MONITORING_INTERVAL_SECONDS: float = 1.0

@dataclass
class APIConfig:
    """External API configuration"""
    # OpenAI
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv('OPENAI_API_KEY', ''))
    OPENAI_MODEL: str = "gpt-4"
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_MAX_TOKENS: int = 2000
    
    # Anthropic
    ANTHROPIC_API_KEY: str = field(default_factory=lambda: os.getenv('ANTHROPIC_API_KEY', ''))
    CLAUDE_MODEL: str = "claude-3-opus-20240229"
    
    # Google
    GOOGLE_API_KEY: str = field(default_factory=lambda: os.getenv('GOOGLE_API_KEY', ''))
    GEMINI_MODEL: str = "gemini-pro"
    
    # W&B Weave
    WEAVE_API_KEY: str = field(default_factory=lambda: os.getenv('WEAVE_API_KEY', ''))
    WEAVE_PROJECT: str = field(default_factory=lambda: os.getenv('WEAVE_PROJECT', 'lab-assistant'))
    
    # Rate limiting
    RATE_LIMIT_CALLS: int = 100
    RATE_LIMIT_PERIOD: int = 3600  # seconds

@dataclass
class UIConfig:
    """User interface configuration"""
    # Streamlit settings
    PAGE_TITLE: str = "Lab Assistant"
    PAGE_ICON: str = "🧪"
    LAYOUT: str = "wide"
    INITIAL_SIDEBAR_STATE: str = "expanded"
    
    # Refresh settings
    AUTO_REFRESH_INTERVAL: int = 1000  # milliseconds
    MAX_REFRESH_COUNT: int = 1000
    
    # Chart settings
    CHART_HEIGHT: int = 400
    CHART_COLOR_SCHEME: str = "plotly"
    
    # Data display
    MAX_TABLE_ROWS: int = 100
    DECIMAL_PLACES: int = 2

@dataclass
class PathConfig:
    """File and directory paths"""
    # Base paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    LOG_DIR: Path = BASE_DIR / "logs"
    CONFIG_DIR: Path = BASE_DIR / "config"
    
    # Data files
    SENSOR_DATA_FILE: Path = DATA_DIR / "sensor_data.json"
    EXPERIMENT_LOG_FILE: Path = DATA_DIR / "experiment_log.json"
    
    # Model files
    MODEL_DIR: Path = BASE_DIR / "models"
    VOICE_MODEL_PATH: Path = MODEL_DIR / "whisper_base.pt"
    
    # Media files
    RECORDINGS_DIR: Path = BASE_DIR / "recordings"
    SCREENSHOTS_DIR: Path = BASE_DIR / "screenshots"
    
    def ensure_directories(self):
        """Create all required directories"""
        for path in [self.DATA_DIR, self.LOG_DIR, self.CONFIG_DIR, 
                     self.MODEL_DIR, self.RECORDINGS_DIR, self.SCREENSHOTS_DIR]:
            path.mkdir(parents=True, exist_ok=True)

@dataclass
class ExperimentConfig:
    """Experiment-specific configuration"""
    # Protocol settings
    DEFAULT_PROTOCOL: str = "nanoparticle_synthesis"
    PROTOCOL_TIMEOUT_HOURS: int = 4
    
    # Data collection
    SAMPLING_RATE_HZ: float = 10.0
    DATA_BUFFER_SIZE: int = 1000
    
    # Experiment phases
    PHASES: list = field(default_factory=lambda: [
        "preparation",
        "reagent_addition",
        "synthesis",
        "purification",
        "analysis",
        "cleanup"
    ])
    
    # Quality thresholds
    MIN_YIELD_PERCENT: float = 30.0
    TARGET_YIELD_PERCENT: float = 40.0
    MAX_CONTAMINATION_PPM: float = 10.0

class Settings:
    """Main settings class that combines all configurations"""
    
    def __init__(self, config_file: Optional[str] = None):
        # Load configurations
        self.chemistry = ChemistryConfig()
        self.safety = SafetyConfig()
        self.api = APIConfig()
        self.ui = UIConfig()
        self.paths = PathConfig()
        self.experiment = ExperimentConfig()
        
        # Ensure directories exist
        self.paths.ensure_directories()
        
        # Load custom config if provided
        if config_file:
            self.load_config(config_file)
        
        # Validate configuration
        self._validate()
    
    def load_config(self, config_file: str):
        """Load configuration from file"""
        config_path = Path(config_file)
        
        if config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        elif config_path.suffix in ['.yml', '.yaml']:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        # Update configurations
        self._update_from_dict(config_data)
    
    def _update_from_dict(self, config_data: Dict[str, Any]):
        """Update configurations from dictionary"""
        for section, values in config_data.items():
            if hasattr(self, section):
                config_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    def _validate(self):
        """Validate configuration values"""
        # Validate API keys
        if not self.api.WEAVE_API_KEY:
            print("Warning: WEAVE_API_KEY not set. W&B logging will be disabled.")
        
        # Validate thresholds
        assert self.safety.TEMP_MIN < self.safety.TEMP_MAX, "Invalid temperature range"
        assert self.safety.PRESSURE_MIN < self.safety.PRESSURE_MAX, "Invalid pressure range"
        assert self.safety.PH_MIN < self.safety.PH_MAX, "Invalid pH range"
        
        # Validate paths
        if not self.paths.BASE_DIR.exists():
            raise ValueError(f"Base directory does not exist: {self.paths.BASE_DIR}")
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return {
            'chemistry': self.chemistry.__dict__,
            'safety': self.safety.__dict__,
            'api': {k: v for k, v in self.api.__dict__.items() 
                   if not k.endswith('_KEY')},  # Don't expose keys
            'ui': self.ui.__dict__,
            'paths': {k: str(v) if isinstance(v, Path) else v 
                     for k, v in self.paths.__dict__.items()},
            'experiment': self.experiment.__dict__
        }
    
    def save_config(self, filepath: str):
        """Save current configuration to file"""
        config_dict = self.get_config_dict()
        
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif filepath.endswith(('.yml', '.yaml')):
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)

# Global settings instance
settings = Settings()

# Convenience functions
def get_chemistry_config() -> ChemistryConfig:
    """Get chemistry configuration"""
    return settings.chemistry

def get_safety_config() -> SafetyConfig:
    """Get safety configuration"""
    return settings.safety

def get_api_config() -> APIConfig:
    """Get API configuration"""
    return settings.api

def get_ui_config() -> UIConfig:
    """Get UI configuration"""
    return settings.ui

def get_path_config() -> PathConfig:
    """Get path configuration"""
    return settings.paths

def get_experiment_config() -> ExperimentConfig:
    """Get experiment configuration"""
    return settings.experiment