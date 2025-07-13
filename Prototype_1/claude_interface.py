import subprocess
import json
import sys
from typing import Optional, Tuple
import logging

# Set up stderr logging for verbose AI interactions
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class ClaudeInterface:
    """Minimal interface to interact with the claude -p command line tool."""
    
    def __init__(self):
        self.command = ["claude", "-p"]
    
    def send_message(self, message: str) -> Tuple[bool, str]:
        """
        Send a message to Claude via command line and return the response.
        
        Args:
            message: The user's input message
            
        Returns:
            Tuple of (success: bool, response: str)
        """
        try:
            # Log the query to stderr
            logger.info("=" * 60)
            logger.info("CLAUDE QUERY:")
            logger.info(f"Command: {' '.join(self.command)}")
            logger.info(f"Message: {message}")
            logger.info("=" * 60)
            
            # Run the claude command with the message
            process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Send the message and get the response
            stdout, stderr = process.communicate(input=message)
            
            if process.returncode == 0:
                response = stdout.strip()
                
                # Log the response to stderr
                logger.info("CLAUDE RESPONSE:")
                logger.info(response)
                logger.info("=" * 60)
                
                return True, response
            else:
                error_msg = f"Error: {stderr.strip()}"
                logger.error(f"CLAUDE ERROR: {error_msg}")
                return False, error_msg
                
        except FileNotFoundError:
            error_msg = "Error: 'claude' command not found. Please ensure Claude CLI is installed and in PATH."
            logger.error(f"CLAUDE ERROR: {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"CLAUDE ERROR: {error_msg}")
            return False, error_msg
    
    def test_connection(self) -> bool:
        """Test if the claude command is available."""
        try:
            subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                check=True
            )
            logger.info("Claude CLI connection test successful")
            return True
        except:
            logger.error("Claude CLI connection test failed")
            return False