import os
import sys
from typing import Optional, Tuple
from pathlib import Path
import logging

# Set up stderr logging for verbose AI interactions
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Install with: pip install google-generativeai")


class GeminiInterface:
    """Interface to interact with Google Gemini 2.5 Pro via API.
    
    Uses the stable gemini-2.5-pro model for all interactions.
    """
    
    def __init__(self):
        self.api_key = None
        self.model = None
        self.model_name = "gemini-2.5-pro"  # Google Gemini 2.5 Pro stable version
        self._load_api_key()
        
    def _load_api_key(self):
        """Securely load API key from RAM disk location."""
        api_key_path = Path.home() / "Documents" / "Ephemeral" / "gapi"
        
        try:
            logger.info(f"Looking for API key at: {api_key_path}")
            if api_key_path.exists():
                with open(api_key_path, 'r') as f:
                    self.api_key = f.read().strip()
                logger.info(f"API key loaded from secure location (length: {len(self.api_key)})")
                
                if GENAI_AVAILABLE and self.api_key:
                    try:
                        genai.configure(api_key=self.api_key)
                        self.model = genai.GenerativeModel(self.model_name)
                        logger.info(f"Successfully initialized Gemini model: {self.model_name}")
                    except Exception as model_error:
                        logger.error(f"Error initializing Gemini model: {str(model_error)}")
                        self.model = None
                else:
                    if not GENAI_AVAILABLE:
                        logger.error("google-generativeai library not available")
                    if not self.api_key:
                        logger.error("API key is empty")
            else:
                logger.error(f"API key file not found at: {api_key_path}")
                self.api_key = None
        except Exception as e:
            logger.error(f"Error loading API key: {str(e)}")
            self.api_key = None
    
    def send_message(self, message: str) -> Tuple[bool, str]:
        """
        Send a message to Gemini and return the response.
        
        Args:
            message: The user's input message
            
        Returns:
            Tuple of (success: bool, response: str)
        """
        if not GENAI_AVAILABLE:
            return False, "Error: google-generativeai library not installed"
        
        if not self.api_key:
            return False, "Error: API key not found at $HOME/Documents/Ephemeral/gapi"
        
        if not self.model:
            return False, "Error: Gemini model not initialized"
        
        try:
            # Log the query to stderr
            logger.info("=" * 60)
            logger.info("GEMINI QUERY:")
            logger.info(f"Model: {self.model_name}")
            logger.info(f"Message: {message}")
            logger.info("=" * 60)
            
            # Send the message to Gemini
            response = self.model.generate_content(message)
            response_text = response.text
            
            # Log the response to stderr
            logger.info("GEMINI RESPONSE:")
            logger.info(response_text)
            logger.info("=" * 60)
            
            return True, response_text
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"GEMINI ERROR: {error_msg}")
            return False, error_msg
    
    def test_connection(self) -> bool:
        """Test if the Gemini API is properly configured."""
        if not GENAI_AVAILABLE:
            logger.error("Google generativeai library not available")
            return False
        
        if not self.api_key:
            logger.error("API key not loaded")
            return False
            
        if not self.model:
            logger.error("Gemini model not initialized")
            return False
        
        try:
            # Simple test query
            logger.info("Testing Gemini connection...")
            response = self.model.generate_content("Say 'OK' if you can hear me.")
            result = bool(response.text)
            logger.info(f"Gemini connection test result: {result}")
            return result
        except Exception as e:
            logger.error(f"Gemini connection test failed: {str(e)}")
            return False