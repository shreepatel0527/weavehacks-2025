#!/usr/bin/env python3
"""
Claude Interface Module
Handles communication with Claude AI via CLI.
"""

import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaudeInterface:
    """Interface for communicating with Claude AI."""
    
    def __init__(self):
        self.available = self.test_connection()
    
    def test_connection(self) -> bool:
        """Test if Claude CLI is available."""
        try:
            result = subprocess.run(['claude', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def send_message(self, message: str) -> tuple[bool, str]:
        """Send a message to Claude and get response."""
        if not self.available:
            return False, "Claude CLI not available"
        
        try:
            # Use Claude CLI to send message
            result = subprocess.run(['claude', '-p', message], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                return False, f"Claude error: {error_msg}"
                
        except subprocess.TimeoutExpired:
            return False, "Claude request timed out"
        except Exception as e:
            return False, f"Error communicating with Claude: {str(e)}"