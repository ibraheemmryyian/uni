"""Script to run the application."""

import os
import sys
from pathlib import Path
import asyncio  # Import asyncio to run the async function

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.main import main  # Import your main function from the appropriate module

if __name__ == "__main__":
    main()