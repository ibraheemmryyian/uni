"""Constants used throughout the application."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_STORE_DIR = DATA_DIR / "vectorstore"
LOG_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [DATA_DIR, VECTOR_STORE_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)