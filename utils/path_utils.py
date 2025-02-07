import os
import sys
from pathlib import Path

def add_project_root_to_path():
    """Add the project root directory to Python path."""
    # Get the directory containing this file
    current_dir = Path(__file__).parent
    # Get the project root (parent of utils/)
    project_root = current_dir.parent
    # Add to Python path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root)) 