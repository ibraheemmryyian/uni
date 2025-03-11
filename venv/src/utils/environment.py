"""Environment setup utilities."""

import sys
import subprocess
import logging
from pathlib import Path

def setup_python_environment():
    """Set up Python environment with required packages."""
    try:
        # Ensure pip is available
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
        
        # Upgrade pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        requirements_path = Path(__file__).parent.parent.parent / "requirements.txt"
        if requirements_path.exists():
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_path)
            ])
        
        return True
    except Exception as e:
        logging.error(f"Failed to set up Python environment: {str(e)}")
        return False