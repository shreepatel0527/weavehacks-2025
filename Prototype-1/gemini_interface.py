#!/usr/bin/env python3
"""
Gemini Interface Module
Handles communication with Google Gemini AI.
"""

import os
import logging
from pathlib import Path

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiInterface:
    """Interface for communicating with Google Gemini AI."""
    
    def __init__(self):
        self.model = None
        self.available = False
        
        if GEMINI_AVAILABLE:
            self.available = self._setup_gemini()
    
    def _setup_gemini(self) -> bool:
        """Setup Gemini API connection."""
        try:
            # Try to get API key from expected location
            api_key_path = Path.home() / "Documents" / "Ephemeral" / "gapi"
            
            if api_key_path.exists():
                api_key = api_key_path.read_text().strip()
            else:
                # Try environment variable
                api_key = os.getenv('GEMINI_API_KEY')
            
            if not api_key:
                logger.warning("Gemini API key not found")
                return False
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Gemini: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test if Gemini API is available."""
        return self.available and self.model is not None
    
    def send_message(self, message: str) -> tuple[bool, str]:
        """Send a message to Gemini and get response."""
        if not self.available or not self.model:
            return False, "Gemini API not available"
        
        try:
            response = self.model.generate_content(message)
            if response.text:
                return True, response.text.strip()
            else:
                return False, "No response from Gemini"
                
        except Exception as e:
            return False, f"Error communicating with Gemini: {str(e)}"